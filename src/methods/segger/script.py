import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import spatialdata as sd
import torch
import xarray as xr
from shapely.geometry import MultiPoint, Polygon
from spatialdata.models import Labels2DModel
from spatialdata.transformations import get_transformation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)
if device.type != "cuda":
    raise RuntimeError(
        "segger requires a CUDA GPU end-to-end (cudf, cuspatial, GPU "
        "kernels) and none is available. Run this component on a "
        "GPU-equipped host."
    )

## VIASH START
par = {
    "input": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr",
    "output": "prediction.zarr",
    "init_segmentation": "auto",
    "n_epochs": 20,
    "prediction_expansion_ratio": 0.5,
    "prediction_mode": "nucleus",
    "cellpose_diameter": 30.0,
}
meta = {"name": "segger", "temp_dir": "/tmp"}
## VIASH END


_UNASSIGNED_CELL_ID_STRINGS = {"-1", "UNASSIGNED", "NONE", "NAN", ""}


def _valid_cell_id_mask(cell_id: pd.Series) -> pd.Series:
    """Pandas analog of segger.validation.quick_metrics.valid_cell_id_expr:
    reject null, -1, "UNASSIGNED", "NONE" (case-insensitive). Keeps the
    "is this transcript assigned?" predicate in sync with the upstream
    segger pipeline so the same sentinels are treated identically here."""
    if pd.api.types.is_numeric_dtype(cell_id):
        return cell_id.notna() & (cell_id != -1)
    s = cell_id.astype(str).str.strip().str.upper()
    return cell_id.notna() & ~s.isin(_UNASSIGNED_CELL_ID_STRINGS)


def _to_lower_uint(arr: np.ndarray) -> np.ndarray:
    m = int(arr.max()) if arr.size else 0
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if m <= np.iinfo(dtype).max:
            return arr.astype(dtype)
    return arr.astype(np.uint64)


def _affine_global_to_pixel(image_element) -> np.ndarray:
    """3x3 homogeneous matrix mapping (x_global, y_global, 1) -> (x_pixel, y_pixel, 1)."""
    t = get_transformation(image_element, "global")
    M = np.asarray(
        t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    )
    return np.linalg.inv(M)


def _apply_affine(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    homog = np.column_stack([xy, np.ones(len(xy))])
    return (homog @ M.T)[:, :2]


def _polygons_from_cell_ids(tx_pd: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convex hull per prior cell id value, in global coordinates."""
    if "cell_id" not in tx_pd.columns:
        raise ValueError("transcripts table has no `cell_id` column")
    records = []
    valid = tx_pd[_valid_cell_id_mask(tx_pd["cell_id"])]
    for cid, group in valid.groupby("cell_id"):
        if len(group) < 3:
            continue
        pts = group[["x", "y"]].to_numpy()
        hull = MultiPoint(pts).convex_hull
        if hull.geom_type != "Polygon":
            continue
        records.append({"cell_id": cid, "geometry": hull})
    if not records:
        raise ValueError("No usable cell_id groups (need >=3 transcripts each)")
    return gpd.GeoDataFrame(records, geometry="geometry")


def _polygons_from_cellpose(image_element, diameter: float) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """Run Cellpose on the morphology image and polygonize the masks.

    Returns the pixel-space label image and a GeoDataFrame of polygons in
    global coordinates (so segger consumes them through the SpatialData
    affine, just like Xenium's native nucleus_boundaries).
    """
    from cellpose.models import CellposeModel
    from rasterio.features import shapes as rio_shapes

    arr = image_element.compute().to_numpy()
    if arr.ndim == 3:
        arr = arr[0]
    print(f"Cellpose input image shape: {arr.shape}", flush=True)
    model = CellposeModel(gpu=torch.cuda.is_available())
    masks, _, _ = model.eval(arr, diameter=diameter, niter=10, flow_threshold=0, min_size=0, resample=False)
    masks = masks.astype(np.int32)

    # Polygonize. shapes() yields (geom_dict, label) for connected regions.
    M_px_to_global = np.linalg.inv(_affine_global_to_pixel(image_element))
    records = []
    for geom_dict, label in rio_shapes(masks, mask=masks > 0):
        coords_px = np.asarray(geom_dict["coordinates"][0])  # exterior ring
        if len(coords_px) < 4:
            continue
        coords_global = _apply_affine(M_px_to_global, coords_px)
        poly = Polygon(coords_global)
        if not poly.is_valid or poly.area == 0:
            continue
        records.append({"cell_id": int(label), "geometry": poly})
    if not records:
        raise RuntimeError("Cellpose produced no usable masks")
    gdf = gpd.GeoDataFrame(records, geometry="geometry")
    return masks, gdf


def _rasterize_polygons(
    gdf: gpd.GeoDataFrame, image_element, label_col: str = "cell_id"
) -> np.ndarray:
    """Rasterize global-coord polygons onto the morphology image grid."""
    from skimage.draw import polygon as draw_polygon

    H, W = image_element.shape[-2:]
    M_g2p = _affine_global_to_pixel(image_element)
    labels = np.zeros((H, W), dtype=np.uint32)
    for cid, geom in zip(gdf[label_col].to_numpy(), gdf.geometry.to_numpy()):
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        ring = np.asarray(geom.exterior.coords)
        pix = _apply_affine(M_g2p, ring)
        rr, cc = draw_polygon(pix[:, 1], pix[:, 0], shape=labels.shape)
        labels[rr, cc] = int(cid)
    return labels


def _shapes_to_xenium_vertices(shapes_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """Xenium boundary parquet schema: one row per polygon vertex with
    columns (cell_id, vertex_x, vertex_y)."""
    rows = []
    for cid, geom in zip(shapes_gdf["cell_id"].to_numpy(), shapes_gdf.geometry.to_numpy()):
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        for vx, vy in np.asarray(geom.exterior.coords):
            rows.append((int(cid), float(vx), float(vy)))
    return pl.DataFrame(rows, schema=["cell_id", "vertex_x", "vertex_y"], orient="row")


def _initial_cell_id_per_transcript(
    tx_pd: pd.DataFrame, initial_labels: np.ndarray, image_element
) -> np.ndarray:
    """For each transcript, return the integer id of the initial mask region
    it lands in, or 0 (segger's UNASSIGNED) when it falls on background."""
    H, W = initial_labels.shape
    M = _affine_global_to_pixel(image_element)
    xy_px = _apply_affine(M, tx_pd[["x", "y"]].to_numpy())
    ys = np.clip(np.round(xy_px[:, 1]).astype(int), 0, H - 1)
    xs = np.clip(np.round(xy_px[:, 0]).astype(int), 0, W - 1)
    return initial_labels[ys, xs].astype(np.int64)


def _build_xenium_layout(
    tx_pd: pd.DataFrame,
    shapes_gdf: gpd.GeoDataFrame,
    initial_labels: np.ndarray,
    image_element,
    out_dir: Path,
) -> Path:
    """Materialize a directory that segger's `10x_xenium` preprocessor can
    consume: `transcripts.parquet` + matching `cell_boundaries.parquet` /
    `nucleus_boundaries.parquet`. We don't have a real cell vs nucleus split,
    so the same polygon set is written to both — segger then drives the
    prediction graph off whichever one `--prediction-mode` selects."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tx_init = _initial_cell_id_per_transcript(tx_pd, initial_labels, image_element)
    cell_id_str = np.where(tx_init > 0, tx_init.astype(str), "UNASSIGNED")

    tx_out = pd.DataFrame({
        "transcript_id": (
            tx_pd["transcript_id"].to_numpy()
            if "transcript_id" in tx_pd.columns
            else np.arange(len(tx_pd), dtype=np.int64)
        ),
        "x_location": tx_pd["x"].to_numpy().astype(np.float32),
        "y_location": tx_pd["y"].to_numpy().astype(np.float32),
        "feature_name": tx_pd["feature_name"].astype(str).to_numpy(),
        "cell_id": cell_id_str,
        "qv": (
            tx_pd["qv"].to_numpy().astype(np.float32)
            if "qv" in tx_pd.columns
            else np.full(len(tx_pd), 40.0, dtype=np.float32)
        ),
        # Treat the entire mask as nucleus (overlaps_nucleus=1); transcripts
        # off the mask get 0. segger only uses this when prediction_mode
        # distinguishes cell vs nucleus.
        "overlaps_nucleus": (tx_init > 0).astype(np.int8),
    })
    if "z" in tx_pd.columns:
        tx_out.insert(3, "z_location", tx_pd["z"].to_numpy().astype(np.float32))
    tx_out.to_parquet(out_dir / "transcripts.parquet", index=False)

    verts = _shapes_to_xenium_vertices(shapes_gdf)
    verts.write_parquet(out_dir / "nucleus_boundaries.parquet")
    verts.write_parquet(out_dir / "cell_boundaries.parquet")
    return out_dir


def _run_segger(xenium_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "segger", "segment",
        "-i", str(xenium_dir),
        "-o", str(output_dir),
        "--n-epochs", str(par["n_epochs"]),
        "--prediction-expansion-ratio", str(par["prediction_expansion_ratio"]),
        "--prediction-mode", par["prediction_mode"],
    ]
    # cudf's `validate_setup` aborts `import cudf` on hosts without a
    # usable NVIDIA driver (cudaErrorInsufficientDriver). segger imports
    # cudf eagerly in tiling.py, so the subprocess dies before it can
    # use its documented CPU fallbacks. RAPIDS_NO_INITIALIZE / *_NO_INITIALIZE
    # skip that validation; cudf functions still attempt GPU calls if
    # invoked, but on small test fixtures segger stays on CPU paths.
    env = os.environ.copy()
    env.setdefault("RAPIDS_NO_INITIALIZE", "1")
    env.setdefault("CUDF_NO_INITIALIZE", "1")
    env.setdefault("RMM_NO_INITIALIZE", "1")
    print("Running segger:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    pq = output_dir / "segger_segmentation.parquet"
    if not pq.exists():
        raise RuntimeError(f"Expected segger output not found: {pq}")
    return pq


def _relabel_initial_mask(
    initial_labels: np.ndarray,
    tx_pixel_xy: np.ndarray,
    tx_segger_cell: np.ndarray,
) -> np.ndarray:
    """For each label in `initial_labels`, replace it by the majority segger
    cell id of the transcripts that fall on top of it. Pixels whose initial
    label has no covering transcripts are left at 0 (background)."""
    H, W = initial_labels.shape
    ys = np.clip(np.round(tx_pixel_xy[:, 1]).astype(int), 0, H - 1)
    xs = np.clip(np.round(tx_pixel_xy[:, 0]).astype(int), 0, W - 1)
    tx_init = initial_labels[ys, xs]
    relabel = {0: 0}
    for init_id in np.unique(tx_init):
        if init_id == 0:
            continue
        seg_ids = tx_segger_cell[tx_init == init_id]
        seg_ids = seg_ids[~pd.isna(seg_ids)]
        if seg_ids.size == 0:
            continue
        most = Counter(seg_ids.astype(np.int64).tolist()).most_common(1)[0][0]
        relabel[int(init_id)] = int(most)
    out = np.zeros_like(initial_labels, dtype=np.int64)
    for k, v in relabel.items():
        out[initial_labels == k] = v
    return out


# ----------------------------- main -------------------------------- #

input_path = Path(par["input"])
output_path = Path(par["output"])
work_root = Path(meta.get("temp_dir") or "/tmp") / f"segger_{os.getpid()}"
work_root.mkdir(parents=True, exist_ok=True)
xenium_dir = work_root / "xenium_input"
segger_out_dir = work_root / "segger_output"

print(f"Reading input: {input_path}", flush=True)
sdata = sd.read_zarr(str(input_path))
print("Input: ", sdata, flush=True)
image_el = sdata["image"]["scale0"].image
image_transform = image_el.transform.copy()

# Bring transcripts into a pandas frame once (we need them again for relabeling).
tx_pd = sdata.points["transcripts"].compute().reset_index(drop=True)

# --- Step 1: produce initial boundaries ---
# Use the vendor `cell_id` prior on the transcripts when present; otherwise
# run Cellpose on the morphology image.
has_prior = "cell_id" in tx_pd.columns and tx_pd["cell_id"].notna().any()
mode = par["init_segmentation"]
if mode == "auto":
    mode = "transcript_cell_id" if has_prior else "cellpose"
print(f"Init segmentation mode: {mode}", flush=True)

initial_labels: np.ndarray
shapes_gdf: gpd.GeoDataFrame
if mode == "transcript_cell_id":
    shapes_gdf = _polygons_from_cell_ids(tx_pd)
    initial_labels = _rasterize_polygons(shapes_gdf, image_el, label_col="cell_id")
elif mode == "cellpose":
    initial_labels, shapes_gdf = _polygons_from_cellpose(image_el, par["cellpose_diameter"])
else:
    raise ValueError(f"Unknown init_segmentation mode: {mode}")
print(f"Initial segmentation: {len(shapes_gdf)} polygons", flush=True)

# --- Step 2: materialize a Xenium-layout dir and run segger ---
# segger@main only knows how to consume Xenium/Merscope/CosMx directories,
# so we synthesize a minimal Xenium dataset from the SpatialData input and
# our just-computed nucleus polygons. transcripts.parquet carries the
# initial mask id as `cell_id` (segger's training signal), and the polygons
# are written to both cell_boundaries.parquet and nucleus_boundaries.parquet
# since segger expects both Xenium files to exist.
_build_xenium_layout(tx_pd, shapes_gdf, initial_labels, image_el, xenium_dir)
seg_pq = _run_segger(xenium_dir, segger_out_dir)
seg = pl.read_parquet(seg_pq)
print(f"segger emitted {seg.height} rows", flush=True)

# Keep only confident assignments. Mirror segger's canonical
# valid_cell_id_expr (null / "-1" / "UNASSIGNED" / "NONE") so that the
# -1 unassigned sentinel segger emits doesn't reach _relabel_initial_mask
# and ultimately the uint32 labels image.
_seg_id_str = pl.col("segger_cell_id").cast(pl.Utf8).str.to_uppercase()
seg = seg.filter(
    pl.col("keep")
    & pl.col("segger_cell_id").is_not_null()
    & (_seg_id_str != "-1")
    & (_seg_id_str != "UNASSIGNED")
    & (_seg_id_str != "NONE")
)
print(f"kept assignments: {seg.height}", flush=True)
if seg.height == 0:
    print("WARNING: segger kept no transcript assignments — falling back to the initial mask.", flush=True)
    final_labels = initial_labels.astype(np.int64)
else:
    row_idx = seg["row_index"].to_numpy()
    segger_cell = seg["segger_cell_id"].to_numpy()
    xy_global = tx_pd.loc[row_idx, ["x", "y"]].to_numpy()
    xy_pixel = _apply_affine(_affine_global_to_pixel(image_el), xy_global)
    final_labels = _relabel_initial_mask(initial_labels, xy_pixel, segger_cell)

final_labels = _to_lower_uint(final_labels)

# --- Step 3: write the SpatialData prediction ---
sd_out = sd.SpatialData(
    labels={
        "segmentation": Labels2DModel.parse(
            xr.DataArray(final_labels, name="segmentation", dims=("y", "x")),
            transformations=image_transform,
        ),
    },
    tables={
        "table": ad.AnnData(
            uns={
                "dataset_id": sdata.tables["table"].uns["dataset_id"],
                "method_id": meta["name"],
            }
        ),
    },
)

print(f"Saving output: {output_path}", flush=True)
if output_path.exists():
    shutil.rmtree(output_path)
sd_out.write(str(output_path))

# best-effort cleanup so the worker doesn't accumulate state across runs
try:
    shutil.rmtree(work_root)
except OSError as e:
    print(f"(non-fatal) cleanup of {work_root} failed: {e}", flush=True)
