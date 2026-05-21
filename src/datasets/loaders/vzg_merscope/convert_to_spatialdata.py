"""Convert MERSCOPE regions to SpatialData .zarr files.

Produces one .zarr per region with:
    images:  'morphology_mip'
    points:  'transcripts'
    shapes:  'cell_boundaries'  (from cell_boundaries.parquet in region folder)
    tables:  'table'            (obs × var counts, linked to cell_boundaries)
    coordinate_systems: 'global'
"""

import shutil
from pathlib import Path

import spatialdata as sd
import yaml
from spatialdata_io import merscope

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("metadata.yml") as f:
    cfg = yaml.safe_load(f)

BASE_DIR = Path(cfg["base_dir"])
OUTPUT_DIR = Path(cfg["output_dir"])
REGIONS = cfg["regions"]
SLIDE_NAME = BASE_DIR.name
Z_LAYER = cfg["z_layer"]
DATASET_METADATA = cfg["dataset"]

# ---------------------------------------------------------------------------
# Load & convert each region
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for region in REGIONS:
    region_dir = BASE_DIR / region
    dataset_id = f"{SLIDE_NAME}_{region}"
    print(f"\n{'='*60}")
    print(f"Processing {region}  (dataset_id={dataset_id})")
    print(f"{'='*60}")

    # --- Load with spatialdata-io merscope reader ---
    sdata = merscope(
        path=region_dir,
        z_layers=Z_LAYER,
        region_name=region,
        slide_name=SLIDE_NAME,
        backend=None,
        transcripts=True,
        cells_boundaries=True,
        cells_table=True,
        mosaic_images=True,
    )
    print(f"[load] {sdata}")

    # --- Rename elements to canonical names ---

    # Images: {dataset_id}_z{Z_LAYER} → morphology_mip
    image_key = f"{dataset_id}_z{Z_LAYER}"
    if image_key in sdata.images:
        sdata.images["morphology_mip"] = sdata.images.pop(image_key)

    # Points: {dataset_id}_transcripts → transcripts
    points_key = f"{dataset_id}_transcripts"
    if points_key in sdata.points:
        sdata.points["transcripts"] = sdata.points.pop(points_key)

    # Shapes: {dataset_id}_polygons → cell_boundaries
    polygons_key = f"{dataset_id}_polygons"
    if polygons_key in sdata.shapes:
        sdata.shapes["cell_boundaries"] = sdata.shapes.pop(polygons_key)

    # --- Update uns metadata on table ---
    DATASET_METADATA["dataset_id"] = dataset_id
    for key, value in DATASET_METADATA.items():
        sdata.tables["table"].uns[key] = value

    print(f"[done] {sdata}")

    # --- Write output ---
    out_path = OUTPUT_DIR / f"{region}.zarr"
    if out_path.exists():
        shutil.rmtree(out_path)
    sdata.write(out_path)
    print(f"[write] {out_path}")

print("\nDone.")
