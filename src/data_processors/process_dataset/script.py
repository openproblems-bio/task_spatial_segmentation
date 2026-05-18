import anndata as ad
import pandas as pd
import spatialdata as sd
import scanpy as sc
import os
import shutil

## VIASH START
par = {
    'input_sp': 'resources_test/common/2023_10x_mouse_brain_xenium_rep1/dataset.zarr',
    'input_sc': 'resources_test/common/2023_yao_mouse_brain_scrnaseq_10xv2/dataset.h5ad',
    'output_spatial_unlabelled': 'spatial_unlabelled.zarr',
    'output_spatial_solution': 'spatial_solution.zarr',
    'output_scrnaseq_reference': 'scrnaseq_reference.h5ad',
    'span': 0.3,
    'seed': 123,
    'n_top_genes': 3000,
    'dataset_id': 'mouse_brain_combined',
    'dataset_name': 'Mouse brain combined dataset',
    'dataset_url': '',
    'dataset_summary': '',
    'dataset_description': '',
    'dataset_reference': [],
    'dataset_organism': 'Mus musculus',
}
## VIASH END

def sc_processing(adata):
    if "counts" not in adata.layers and adata.X != None:
        print(">> Save raw counts in .layer", flush=True)
        adata.layers["counts"] = adata.X.copy()
        
    if "normalized" not in adata.layers:
        print(">> Perform standard normalization", flush=True)
        adata.layers["normalized"] = adata.layers["counts"].copy()
        sc.pp.normalize_total(adata, layer="normalized", inplace=True)

    if "normalized_log" not in adata.layers:
        print(">> Perform log1p normalization", flush=True)
        adata.layers["normalized_log"] = adata.layers["normalized"].copy()
        sc.pp.normalize_total(adata, layer="normalized_log", inplace=True)

    if "normalized_log_scaled" not in adata.layers:
        print(">> Perform 0 mean and standard variance normalization", flush=True)
        adata.layers["normalized_log_scaled"] = adata.layers["normalized_log"].copy()
        sc.pp.normalize_total(adata, layer="normalized_log_scaled", inplace=True)

    if "hvg" not in adata.var:
        print(">> Compute highly variable genes", flush=True)
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            layer="counts",
            span=par['span'],
            n_top_genes=par['n_top_genes']
        )
        adata.var.rename(columns={"highly_variable": "hvg"}, inplace=True)

print(">> Load data", flush=True)
sc_data = ad.read_h5ad(par["input_sc"])
print(f"single cell data: {sc_data}")

print(">> Processing sc_data", flush=True)
sc_processing(sc_data)

print(">> Override dataset metadata in .uns", flush=True)
sc_data.uns["orig_dataset_id"] = sc_data.uns.get("dataset_id", None)
for key in ["dataset_id", "dataset_name", "dataset_url", "dataset_summary", "dataset_description", "dataset_organism", "dataset_reference"]:
    sc_data.uns[key] = par[key]

print(">> Writing scrnaseq reference", flush=True)
sc_data.write_h5ad(par["output_scrnaseq_reference"], compression="gzip")

# read input_sp
print(">> Read spatial data", flush=True)
sp_data = sd.read_zarr(par["input_sp"])
print(f"spatial data: {sp_data}")

# TODO: make sure user provides the groundtruth instead of copying it from the vendor labels
if "groundtruth_cell_labels" not in sp_data.labels:
    if "cell_labels" in sp_data.labels:
        print("=" * 80, flush=True)
        print("WARNING: 'groundtruth_cell_labels' not found in spatial data!", flush=True)
        print("Temporarily falling back to vendor-provided 'cell_labels' as ground truth.", flush=True)
        print("=" * 80, flush=True)
        sp_data["groundtruth_cell_labels"] = sp_data.labels["cell_labels"]
    else:
        raise ValueError("Neither 'groundtruth_cell_labels' nor 'cell_labels' found in spatial data.")

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
else:
    raise ValueError("No suitable image found in spatial data. Expected 'image' or 'morphology_mip'.")

unlabelled_labels = {k: sp_data.labels[k] for k in ["cell_labels", "nucleus_labels"] if k in sp_data.labels}

output_spatial_unlabelled = sd.SpatialData(
    images={"image": sp_data.images[input_image_name]},
    labels=unlabelled_labels,
    points={"transcripts": clean_transcripts},
    tables={"table": minimal_table},
)

print(">> Writing spatial unlabelled dataset", flush=True)
print("Format: ", output_spatial_unlabelled, flush=True)
# remove if output exists
if os.path.exists(par["output_spatial_unlabelled"]):
    if os.path.isdir(par["output_spatial_unlabelled"]):
        shutil.rmtree(par["output_spatial_unlabelled"])
    else:
        os.remove(par["output_spatial_unlabelled"])
output_spatial_unlabelled.write(par["output_spatial_unlabelled"], overwrite=True)

# ---------------------------------------------------------------
# output_spatial_solution: ground truth labels, shapes, reference table
# ---------------------------------------------------------------
print(">> Building spatial solution (ground truth)", flush=True)

ref_table = sp_data.tables["table"]
solution_obs = ref_table.obs[["cell_id", "region"]].copy()
for extra_col in ["cell_area", "transcript_counts"]:
    if extra_col in ref_table.obs.columns:
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

solution_labels = {
    "groundtruth_cell_labels": sp_data.labels["groundtruth_cell_labels"],
}
if "cell_labels" in sp_data.labels:
    solution_labels["cell_labels"] = sp_data.labels["cell_labels"]
if "nucleus_labels" in sp_data.labels:
    solution_labels["nucleus_labels"] = sp_data.labels["nucleus_labels"]

output_solution = sd.SpatialData(
    points={"transcripts": solution_transcripts},
    labels=solution_labels,
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
