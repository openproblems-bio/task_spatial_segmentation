"""
Assignment Specificity Ratio (ASR) metric.

Measures how specifically a segmentation method assigns hepatocyte transcripts
to hepatocytes (vs. neighbouring sinusoidal endothelial cells) by comparing the
zonal gene-expression gradients of the two cell types relative to their distance
from the nearest Portal Area (PA) boundary in the liver lobule.

Formula
-------
For each target gene:

    ASR = β_hep / (β_sin + ε)

where
    β_hep  = OLS regression slope of hepatocyte log-norm expression vs dist_PA
    β_sin  = OLS regression slope of sinusoidal log-norm expression vs dist_PA
    ε      = small stabilising constant (par["epsilon"], default 1e-6)

Interpretation
--------------
    ASR ≫ 1  — gradient confined to hepatocytes; correct assignment
    ASR ≈ 1  — sinusoidal cells show an equal gradient; mis-assignment detected
    ASR < 1  — catastrophic mis-assignment; sinusoidal cells carry more signal

Inputs
------
    input_prediction  : processed_prediction.zarr
                        (labels/segmentation, tables/table with CxG matrix)
    input_solution    : spatial_solution.zarr
                        (shapes/pa_boundaries, shapes/cv_boundaries,
                         points/transcripts with ground-truth coordinates)

KNOWN ISSUE — dataset specificity
----------------------------------
This metric requires pa_boundaries shapes in spatial_solution.zarr.  These are
present ONLY for liver datasets prepared with liver_zonation/migrate_annotations.py.
The standard mouse-brain test dataset (resources_test/…/mouse_brain_combined)
does NOT contain pa_boundaries / cv_boundaries.

The guard block below detects this and writes NaN metric values, allowing the
metric to run on all benchmark datasets without crashing.  However, NaN scores
are not meaningful for benchmarking non-liver datasets.

Maintainers: a dataset-specific benchmark configuration or dataset-level filter
is recommended before full production deployment.  See also:
  src/api/file_common_ist.yaml   — schema note on pa_boundaries / cv_boundaries
  src/metrics/asr/config.vsh.yaml — description and KNOWN ISSUE section
"""

import numpy as np
import pandas as pd
import anndata as ad
import spatialdata as sd
import xarray as xr
from shapely.ops import unary_union
import geopandas as gpd

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution":   "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output":           "output.h5ad",
    "genes":            ["MET", "CAV1"],
    "epsilon":          1e-6,
    "min_cells":        10,
    "landmark_dist_threshold": 4706.0,   # px ≈ 1000 µm at 0.2125 µm/px
}
meta = {
    "name": "asr"
}
## VIASH END


# ── Marker gene sets for simple cell-type assignment ─────────────────────────
# These sets are intentionally small and focused on unambiguous markers so that
# the scoring works even on panels with a limited number of genes (~300-400
# is typical for Xenium).  Adjust if the gene panel changes.
#
# hepatocyte markers  — periportal to panlobular expression
HEPATOCYTE_MARKERS = ["ALB", "APOA1", "CYP3A4", "GLUL", "HPX"]
# sinusoidal endothelial cell markers
SINUSOIDAL_MARKERS = ["PECAM1", "VWF", "CD34", "FCGR2A"]

MARKER_GROUPS = {
    "hepatocyte": HEPATOCYTE_MARKERS,
    "sinusoidal": SINUSOIDAL_MARKERS,
}

# Minimum mean log-norm expression for a cell type assignment to be kept;
# cells below this threshold across ALL groups are labelled "unassigned".
MIN_SCORE_THRESHOLD = 0.05


# ── Helper: look up label image values at transcript coordinates ──────────────
def _lookup_labels(label_element, transcripts_global):
    """Return label-image values at each transcript's (x, y) position.

    Mirrors the coordinate-handling pattern from metrics/ari/script.py:28-42.
    Applies the label element's inverse global transform to convert transcript
    global coordinates to label-array pixel indices, then clips to bounds.

    Parameters
    ----------
    label_element     : xr.DataTree | xr.DataArray  — the segmentation labels
    transcripts_global: dask DataFrame               — transcripts in global CRS

    Returns
    -------
    np.ndarray of int  — predicted cell_id per transcript (0 = background)
    """
    # Get the inverse transform: global → label intrinsic (pixel indices)
    trans = sd.transformations.get_transformation(
        label_element, get_all=True
    )["global"].inverse()
    transcripts_local = sd.transform(transcripts_global, trans, "global")

    y = transcripts_local.y.compute().to_numpy(dtype=np.int64)
    x = transcripts_local.x.compute().to_numpy(dtype=np.int64)

    # Handle multiscale DataTree (use full-resolution scale0)
    if isinstance(label_element, xr.DataTree):
        img = label_element["scale0"].image.to_numpy()
    else:
        img = label_element.to_numpy()

    # Clip to valid array bounds (guards against floating-point rounding at edges)
    y = np.clip(y, 0, img.shape[0] - 1)
    x = np.clip(x, 0, img.shape[1] - 1)

    return img[y, x]


# ── Helper: write NaN score file and exit ─────────────────────────────────────
def _write_nan_output(par, meta, dataset_id, method_id, genes, reason):
    """Write a well-formed file_score.h5ad with NaN metric values.

    Used when the metric cannot be computed (e.g. no PA/CV shapes in solution,
    gene not in panel, insufficient cells).  The output still conforms to the
    file_score.yaml schema so the benchmark pipeline does not fail.
    """
    print(f"WARNING: {reason}  Writing NaN metric values.", flush=True)
    metric_ids    = [f"asr_{g}" for g in genes]
    metric_values = [float("nan")] * len(genes)
    output = ad.AnnData(uns={
        "dataset_id":       dataset_id,
        "normalization_id": "normalized_log",
        "method_id":        method_id,
        "metric_ids":       metric_ids,
        "metric_values":    metric_values,
    })
    output.write_h5ad(par["output"], compression="gzip")
    print(f"  metric_ids:    {metric_ids}", flush=True)
    print(f"  metric_values: {metric_values}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

print(">> Reading input files", flush=True)
sdata_pred = sd.read_zarr(par["input_prediction"])
sdata_sol  = sd.read_zarr(par["input_solution"])

dataset_id = sdata_sol.tables["table"].uns["dataset_id"]
method_id  = sdata_pred.tables["table"].uns["method_id"]

# ── Guard: PA/CV shapes must be present in the solution ──────────────────────
#
# KNOWN ISSUE: non-liver datasets (e.g. mouse brain) will always hit this path
# because their spatial_solution.zarr does not contain pa_boundaries.
# This is intentional graceful degradation.  See module docstring for details.
if "pa_boundaries" not in sdata_sol.shapes:
    _write_nan_output(
        par, meta, dataset_id, method_id, par["genes"],
        "pa_boundaries not found in spatial_solution.zarr.  "
        "This is a non-liver dataset, or migrate_annotations.py has not been "
        "run on the upstream zarr.  ASR requires PA/CV landmark annotations."
    )
    raise SystemExit(0)

# ── Build PA union polygon in global (pixel) coordinate space ─────────────────
# sd.transform with to_coordinate_system="global" applies the shape's scale
# transform (1/pixel_size ≈ 4.706) and returns geometries in pixel space,
# consistent with the morphology image and label arrays.
print(">> Building PA boundary union in global (pixel) space", flush=True)
pa_global = sd.transform(sdata_sol["pa_boundaries"], to_coordinate_system="global")
pa_union  = unary_union(pa_global.geometry)
print(
    f"   PA union bbox (pixels): "
    f"x=[{pa_global.geometry.total_bounds[0]:.0f}, {pa_global.geometry.total_bounds[2]:.0f}]  "
    f"y=[{pa_global.geometry.total_bounds[1]:.0f}, {pa_global.geometry.total_bounds[3]:.0f}]",
    flush=True,
)

# ── Load solution transcripts in global (pixel) space ────────────────────────
# Transcript coordinates from spatial_solution are used to:
#   (a) look up the predicted cell_id in the segmentation label image, and
#   (b) compute per-cell centroid as the mean (x, y) of its transcripts.
# This avoids loading a second copy of the label image for regionprops and
# is consistent with how metrics/ari/script.py handles coordinates.
print(">> Loading solution transcripts in global space", flush=True)
transcripts_global = sd.transform(
    sdata_sol["transcripts"], to_coordinate_system="global"
)

# ── Look up predicted cell ID for each transcript ─────────────────────────────
# Applies the inverse of the segmentation's global transform so that transcript
# global coordinates become pixel indices into the label array.
print(">> Looking up predicted cell IDs from segmentation label image", flush=True)
pred_cell_ids = _lookup_labels(sdata_pred["segmentation"], transcripts_global)

# ── Compute per-transcript global (x, y) and distance to nearest PA boundary ──
print(">> Computing PA boundary distances for each transcript", flush=True)
tx = transcripts_global.x.compute().to_numpy()   # global x (pixels)
ty = transcripts_global.y.compute().to_numpy()   # global y (pixels)

# geopandas GeoSeries.distance() uses vectorised GEOS distance — far faster
# than a Python Shapely loop over millions of transcripts.
trans_points  = gpd.GeoSeries.from_xy(pd.Series(tx), pd.Series(ty))
dist_pa_trans = trans_points.distance(pa_union).to_numpy()

# ── Aggregate transcripts to cell level ───────────────────────────────────────
# For each predicted cell:
#   - centroid (cx, cy) = mean transcript (x, y)  [in global / pixel units]
#   - dist_pa           = mean transcript distance to PA boundary
#                         (good approximation to centroid distance for typical
#                          hepatocyte sizes ~20-40 µm)
print(">> Aggregating transcripts to per-cell summaries", flush=True)
trans_df = pd.DataFrame({
    "cell_id": pred_cell_ids,
    "x":       tx,
    "y":       ty,
    "dist_pa": dist_pa_trans,
})
# Drop background (cell_id == 0 means the transcript was not assigned to any cell)
trans_df = trans_df[trans_df["cell_id"] != 0]

cell_df = trans_df.groupby("cell_id").agg(
    cx      = ("x",       "mean"),
    cy      = ("y",       "mean"),
    dist_pa = ("dist_pa", "mean"),
).reset_index()

print(f"   Predicted cells with ≥1 transcript: {len(cell_df):,}", flush=True)

# ── Merge with expression table ───────────────────────────────────────────────
# processed_prediction table obs["cell_id"] is a string; cell_df["cell_id"] is
# an integer from the label image.  Convert for the join.
print(">> Merging with expression table", flush=True)
table = sdata_pred.tables["table"]

cell_df["cell_id_str"] = cell_df["cell_id"].astype(str)
obs_df = table.obs[["cell_id"]].copy()    # cell_id as string

merged = obs_df.merge(
    cell_df[["cell_id_str", "dist_pa"]],
    left_on="cell_id", right_on="cell_id_str",
    how="left",
)
merged.index = obs_df.index    # preserve AnnData obs index alignment

dist_all = merged["dist_pa"].to_numpy(dtype=float)   # NaN for cells with no transcripts

# ── Assign cell types from marker gene scores ─────────────────────────────────
# Uses mean log-normalised expression of curated marker sets.
# No clustering: straightforward, fast, works on any gene panel size.
# Falls back to "unassigned" for cells below MIN_SCORE_THRESHOLD.
print(">> Assigning cell types from marker gene scores", flush=True)

# Extract log-normalised expression matrix as a dense DataFrame (cells × genes)
lognorm = table.layers["normalized_log"]
if hasattr(lognorm, "toarray"):
    lognorm = lognorm.toarray()
else:
    lognorm = np.array(lognorm)

expr_df = pd.DataFrame(lognorm, index=table.obs_names, columns=table.var_names)

scores = {}
for ct, markers in MARKER_GROUPS.items():
    avail = [m for m in markers if m in expr_df.columns]
    missing = [m for m in markers if m not in expr_df.columns]
    if missing:
        print(f"   {ct}: {len(avail)} markers available, missing: {missing}", flush=True)
    if avail:
        scores[ct] = expr_df[avail].mean(axis=1)
    else:
        print(f"   WARNING: no markers available for '{ct}' — assigning score 0.", flush=True)
        scores[ct] = pd.Series(0.0, index=expr_df.index)

score_df  = pd.DataFrame(scores)
cell_type = score_df.idxmax(axis=1)
max_score = score_df.max(axis=1)
cell_type[max_score < MIN_SCORE_THRESHOLD] = "unassigned"

hep_mask = (cell_type == "hepatocyte").to_numpy()
sin_mask = (cell_type == "sinusoidal").to_numpy()

n_hep = hep_mask.sum()
n_sin = sin_mask.sum()
print(f"   Hepatocytes: {n_hep:,}   Sinusoidal: {n_sin:,}", flush=True)

# ── Compute ASR per gene ───────────────────────────────────────────────────────
print(">> Computing ASR per gene", flush=True)

metric_ids    = []
metric_values = []

for gene in par["genes"]:
    metric_ids.append(f"asr_{gene}")

    # Gene not in gene panel → NaN
    if gene not in expr_df.columns:
        print(f"   {gene}: absent from gene panel — NaN", flush=True)
        metric_values.append(float("nan"))
        continue

    gene_expr = expr_df[gene].to_numpy()

    slopes = {}
    for ct_label, mask in [("hepatocyte", hep_mask), ("sinusoidal", sin_mask)]:
        # Restrict to cells that:
        #   (a) belong to this cell type
        #   (b) have a valid (non-NaN) PA distance
        #   (c) are within the landmark distance threshold
        valid = (
            mask
            & np.isfinite(dist_all)
            & (dist_all <= par["landmark_dist_threshold"])
        )
        n = int(valid.sum())

        if n < par["min_cells"]:
            print(
                f"   {gene}/{ct_label}: only {n} cells within threshold "
                f"(need ≥ {par['min_cells']}) — NaN slope",
                flush=True,
            )
            slopes[ct_label] = None
        else:
            d = dist_all[valid]
            e = gene_expr[valid]
            # np.polyfit returns [slope, intercept] for degree-1 polynomial
            slope = float(np.polyfit(d, e, 1)[0])
            slopes[ct_label] = slope
            print(
                f"   {gene}/{ct_label}: slope = {slope:.6f}  n = {n:,}",
                flush=True,
            )

    slope_hep = slopes.get("hepatocyte")
    slope_sin = slopes.get("sinusoidal")

    if slope_hep is None or slope_sin is None:
        # Insufficient cells for one or both types
        asr = float("nan")
    else:
        # Core ASR formula:  β_hep / (β_sin + ε)
        asr = slope_hep / (slope_sin + par["epsilon"])

    print(f"   {gene}: ASR = {asr}", flush=True)
    metric_values.append(float(asr) if np.isfinite(asr) else float("nan"))

# ── Write score file ──────────────────────────────────────────────────────────
# Output conforms to file_score.yaml:
#   uns.dataset_id, uns.normalization_id, uns.method_id,
#   uns.metric_ids (list[str]), uns.metric_values (list[float])
print(">> Writing output score file", flush=True)
output = ad.AnnData(uns={
    "dataset_id":       dataset_id,
    "normalization_id": "normalized_log",
    "method_id":        method_id,
    "metric_ids":       metric_ids,
    "metric_values":    metric_values,
})
output.write_h5ad(par["output"], compression="gzip")

print("Done.", flush=True)
print(f"  metric_ids:    {metric_ids}",    flush=True)
print(f"  metric_values: {metric_values}", flush=True)
