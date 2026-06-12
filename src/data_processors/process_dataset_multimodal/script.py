import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
import scanpy as sc
import os
import shutil

def _compute_protein_mean(protein_table, sp_data, input_image_name, matched):
    """Compute mean image intensity per cell per matched protein channel."""
    spatialdata_attrs = protein_table.uns.get('spatialdata_attrs', {})
    region = spatialdata_attrs.get('region', None)
    if isinstance(region, list):
        region = region[0]
    if region is None or region not in sp_data.labels:
        raise ValueError(
            f"Cannot compute protein_mean: no valid region in spatialdata_attrs "
            f"(got '{region}'). Available labels: {list(sp_data.labels.keys())}"
        )

    labels_elem = sp_data.labels[region]
    labels_arr = (
        labels_elem['scale0'].values.squeeze()
        if hasattr(labels_elem, 'scale0')
        else labels_elem.values.squeeze()
    )

    image_elem = sp_data.images[input_image_name]
    image_arr = (
        image_elem['scale0'].values
        if hasattr(image_elem, 'scale0')
        else image_elem.values
    )  # shape: (C, Y, X)

    all_channels = list(sp_data.images[input_image_name]['scale0'].coords['c'].values)
    channel_indices = [all_channels.index(ch) for ch in matched]

    instance_key = spatialdata_attrs.get('instance_key', 'cell_id')
    cell_ids = protein_table.obs[instance_key].values
    cell_id_to_row = {cid: i for i, cid in enumerate(cell_ids)}

    intensities = np.zeros((len(protein_table), len(matched)), dtype=np.float32)
    unique_labels = np.unique(labels_arr)
    for col_idx, (ch_idx, channel) in enumerate(zip(channel_indices, matched)):
        print(f"  [{col_idx + 1}/{len(matched)}] Computing mean for channel '{channel}'", flush=True)
        for label_val in unique_labels:
            if label_val == 0 or label_val not in cell_id_to_row:
                continue
            mask = labels_arr == label_val
            row_idx = cell_id_to_row[label_val]
            intensities[row_idx, col_idx] = image_arr[ch_idx][mask].mean()

    return intensities

## VIASH START
par = {
    'input_sp': 'resources_test/common/Xenium_V1_Human_Kidney_FFPE/Xenium_V1_Human_Kidney_FFPE_crop.zarr',
    'output_spatial_unlabelled': 'spatial_unlabelled.zarr',
    'output_spatial_solution': 'spatial_solution.zarr',
    'span': 0.3,
    'seed': 123,
    'dataset_id': 'Xenium_V1_Human_Kidney_FFPE',
    'dataset_name': 'Xenium V1 Human Kidney FFPE (with proteins)',
    'dataset_url': '',
    'dataset_summary': '',
    'dataset_description': '',
    'dataset_reference': [],
    'dataset_organism': 'Homo sapiens',
}
## VIASH END

# read input_sp
print(">> Read spatial data", flush=True)
sp_data = sd.read_zarr(par["input_sp"])
print(f"spatial data: {sp_data}")

dataset_uns = {
    "dataset_id": par["dataset_id"],
    "dataset_name": par["dataset_name"],
    "dataset_url": par["dataset_url"],
    "dataset_summary": par["dataset_summary"],
    "dataset_description": par["dataset_description"],
    "dataset_reference": par["dataset_reference"],
    "dataset_organism": par["dataset_organism"],
    "orig_dataset_id": sp_data.tables["table"].uns.get("dataset_id", None),
}

# ---------------------------------------------------------------
# output_spatial_dataset: image + transcripts (no ground truth)
# ---------------------------------------------------------------
print(">> Building spatial dataset for methods (no ground truth)", flush=True)

# Keep only the columns defined in the file_spatial_unlabelled schema.
# Any extra columns (e.g. ground-truth nucleus_id, cell_type) are dropped by
# only selecting schema-defined columns rather than by explicit exclusion.
_TRANSCRIPT_COLS_REQUIRED = ["x", "y", "feature_name", "transcript_id"]
_TRANSCRIPT_COLS_OPTIONAL = ["z", "qv", "overlaps_nucleus", "cell_id"]
_TRANSCRIPT_COLS_SCHEMA = set(_TRANSCRIPT_COLS_REQUIRED + _TRANSCRIPT_COLS_OPTIONAL)
transcripts = sp_data.points["transcripts"]
for col in _TRANSCRIPT_COLS_REQUIRED:
    assert col in transcripts.columns, f"Required transcript column '{col}' is missing from the input data"
clean_transcript_cols = [c for c in transcripts.columns if c in _TRANSCRIPT_COLS_SCHEMA]
clean_transcripts = transcripts[clean_transcript_cols]

# Build var from unique feature names in transcripts, mapping to feature_ids from metadata
feature_names = transcripts["feature_name"].compute().unique().tolist()
var_df = pd.DataFrame({"feature_name": feature_names}, index=feature_names)
var_df.index.name = "feature_name"
if "metadata" in sp_data.tables and "gene_ids" in sp_data.tables["metadata"].var.columns:
    id_map = sp_data.tables["metadata"].var["gene_ids"]
    var_df["feature_id"] = var_df.index.map(id_map)

# Minimal table: dataset metadata in uns, gene list in var
minimal_table = ad.AnnData(var=var_df, uns=dataset_uns)

if "image" in sp_data.images:
    input_image_name = "image"
elif "morphology_mip" in sp_data.images:
    print("WARNING: 'morphology_mip' image found but expected 'image'. Using 'morphology_mip' as fallback.", flush=True)
    input_image_name = "morphology_mip"
elif "morphology_focus" in sp_data.images:
    print("WARNING: 'morphology_focus' image found but expected 'image'. Using 'morphology_focus' as fallback.", flush=True)
    input_image_name = "morphology_focus"
else:
    raise ValueError("No suitable image found in spatial data. Expected 'image' or 'morphology_mip/focus'.")

### preparing the protein reference table:
protein_table = sp_data.tables['protein']

### checking that var_names match the channels in the image:
image_channels = list(sp_data.images[input_image_name]['scale0'].coords['c'].values)
protein_var_names = protein_table.var_names.tolist()

matched = [v for v in protein_var_names if v in image_channels]
unmatched = [v for v in protein_var_names if v not in image_channels]

if unmatched:
    print(
        f"WARNING: {len(unmatched)} protein var_names not found in image channels and will be dropped: {unmatched}",
        flush=True,
    )

if not matched:
    raise ValueError(
        f"No protein var_names match any image channel. "
        f"protein var_names: {protein_var_names}, image channels: {image_channels}"
    )

protein_table = protein_table[:, matched].copy()
print(f"Filtered protein_table to {len(matched)} channel(s) matching image: {matched}", flush=True)

### keep only matched_gene in var and filter to genes present in minimal_table:
assert 'matched_gene' in protein_table.var.columns, \
    "Column 'matched_gene' not found in protein_table.var"
protein_table.var = protein_table.var[['matched_gene']]

valid_gene_mask = protein_table.var['matched_gene'].isin(minimal_table.var_names)
n_dropped = (~valid_gene_mask).sum()
if n_dropped:
    dropped_channels = protein_table.var_names[~valid_gene_mask].tolist()
    print(
        f"WARNING: {n_dropped} protein channel(s) have 'matched_gene' not in minimal_table.var_names and will be dropped: {dropped_channels}",
        flush=True,
    )
protein_table = protein_table[:, valid_gene_mask.values].copy()
matched = protein_table.var_names.tolist()
print(f"Retained {len(matched)} protein channel(s) with matched_gene in minimal_table", flush=True)



### ensure protein_mean is in layers:
if 'protein_mean' in protein_table.layers:
    print("'protein_mean' found in layers", flush=True)
elif protein_table.X is not None:
    print("'protein_mean' not in layers — moving X to layers['protein_mean'] and clearing X", flush=True)
    protein_table.layers['protein_mean'] = protein_table.X.copy()
    protein_table.X = None
else:
    print("'protein_mean' absent — computing mean intensity per cell per channel from image", flush=True)
    protein_table.layers['protein_mean'] = _compute_protein_mean(
        protein_table, sp_data, input_image_name, matched
    )
    print(
        f"Computed protein_mean for {len(protein_table)} cells x {len(matched)} channels",
        flush=True,
    )

### drop all obs columns:
protein_table.obs = protein_table.obs[[]]

### remove spatialdata_attrs so the table is not region-annotated in the unlabelled output:
protein_table.uns.pop('spatialdata_attrs', None)

output_spatial = sd.SpatialData(
    images={"image": sp_data.images[input_image_name]},
    points={"transcripts": clean_transcripts},
    tables={"table": minimal_table,
            "protein": protein_table},
)

print(">> Writing spatial unlabelled dataset", flush=True)
# remove if output exists
if os.path.exists(par["output_spatial_unlabelled"]):
    if os.path.isdir(par["output_spatial_unlabelled"]):
        shutil.rmtree(par["output_spatial_unlabelled"])
    else:
        os.remove(par["output_spatial_unlabelled"])
output_spatial.write(par["output_spatial_unlabelled"], overwrite=True)

# ---------------------------------------------------------------
# output_spatial_solution: ground truth labels, shapes, reference table
# ---------------------------------------------------------------
print(">> Building spatial solution (ground truth)", flush=True)

ref_table = sp_data.tables["table"]
_spatialdata_attrs = ref_table.uns.get("spatialdata_attrs", {})
_instance_key = _spatialdata_attrs.get("instance_key", "cell_id")
_region_key = _spatialdata_attrs.get("region_key", "region")
_required_obs_cols = [c for c in [_instance_key, _region_key] if c in ref_table.obs.columns]
solution_obs = ref_table.obs[_required_obs_cols].copy()
for extra_col in ["cell_labels", "cell_area", "transcript_counts"]:
    if extra_col in ref_table.obs.columns and extra_col not in solution_obs.columns:
        solution_obs[extra_col] = ref_table.obs[extra_col]

solution_table = ad.AnnData(
    obs=solution_obs,
    var=var_df,
    uns={
        "dataset_id": par["dataset_id"],
        "dataset_name": par["dataset_name"],
        "dataset_url": par["dataset_url"],
        "dataset_summary": par["dataset_summary"],
        "dataset_description": par["dataset_description"],
        "dataset_reference": par["dataset_reference"],
        "dataset_organism": par["dataset_organism"],
        "orig_dataset_id": sp_data.tables["table"].uns.get("dataset_id", None),
        "spatialdata_attrs": ref_table.uns["spatialdata_attrs"],
    },
)

# Keep only the columns needed for the solution (ground truth assignments)
_SOLUTION_TRANSCRIPT_COLS = ["x", "y", "feature_name", "cell_id", "transcript_id"]
if "z" in transcripts.columns:
    _SOLUTION_TRANSCRIPT_COLS = ["x", "y", "z"] + _SOLUTION_TRANSCRIPT_COLS[2:]
solution_transcripts = transcripts[[c for c in _SOLUTION_TRANSCRIPT_COLS if c in transcripts.columns]]

output_solution = sd.SpatialData(
    points={"transcripts": solution_transcripts},
    labels={k: v for k, v in sp_data.labels.items()},
    shapes={k: v for k, v in sp_data.shapes.items()},
    tables={"table": solution_table},
)

print(">> Writing spatial solution", flush=True)
if os.path.exists(par["output_spatial_solution"]):
    if os.path.isdir(par["output_spatial_solution"]):
        shutil.rmtree(par["output_spatial_solution"])
    else:
        os.remove(par["output_spatial_solution"])
output_solution.write(par["output_spatial_solution"], overwrite=True)
