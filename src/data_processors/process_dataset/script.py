import random
import anndata as ad
import spatialdata as sd
import scanpy as sc

## VIASH START
par = {
    'input_sp': 'resources_test/common/2023_10x_mouse_brain_xenium_rep1/dataset.zarr',
    'input_sc': 'resources_test/common/2023_yao_mouse_brain_scrnaseq_10xv2/dataset.h5ad',
    'output_spatial_dataset': 'resources_test/task_spatial_segmentation/mouse_brain_combined/output_spatial_dataset.zarr',
    'output_scrnaseq_reference': 'resources_test/task_spatial_segmentation/mouse_brain_combined/output_scrnaseq_reference.h5ad',
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


# set seed if need be
if par["seed"]:
    print(f">> Setting seed to {par['seed']}")
    random.seed(par["seed"])

print(">> Load data", flush=True)
sc_data = ad.read_h5ad(par["input_sc"])
print(f"single cell data: {sc_data}")

print(">> Processing sc_data", flush=True)
sc_processing(sc_data)

print(">> Override dataset metadata in .uns", flush=True)
sc_data.uns["orig_dataset_id"] = sc_data.uns.get("dataset_id", None)
for key in ["dataset_id", "dataset_name", "dataset_url", "dataset_summary", "dataset_description", "dataset_reference", "dataset_organism"]:
    sc_data.uns[key] = par[key]

print(">> Writing data", flush=True)
sc_data.write_h5ad(par["output_scrnaseq_reference"], compression="gzip")

# read input_sp
print(">> Read spatial data", flush=True)
sp_data = sd.read_zarr(par["input_sp"])
print(f"spatial data: {sp_data}")

print(">> Processing spatial data", flush=True)
sp_data_table = sp_data.tables['table']
print(f"single cell part of spatial data: {sp_data_table}")
sc_processing(sp_data_table)

if "cell_area" not in sp_data_table.obs:
    print(">> Perform scanpy qc for cell area", flush=True)
    sc.pp.calculate_qc_metrics(sp_data_table, layer="counts", inplace=True)

for x in ["transcript_counts", "n_genes_by_counts"]:
    if f"ca_normalized_{x}" not in sp_data_table.obs and x in sp_data_table.obs:
        print(f">> Perform cell area normalization for {x}", flush=True)
        sp_data_table.obs[f'ca_normalized_{x}'] = sp_data_table.obs[f"{x}"] / sp_data_table.obs["cell_area"]

print(">> Override dataset metadata in .uns", flush=True)
sp_data_table.uns["orig_dataset_id"] = sp_data_table.uns.get("dataset_id", None)
for key in ["dataset_id", "dataset_name", "dataset_url", "dataset_summary", "dataset_description", "dataset_reference", "dataset_organism"]:
    sp_data_table.uns[key] = par[key]

print(f"spatial data: {sp_data}")
print(f"spatial data tables['table']: {sp_data.tables['table']}")

print(">> Writing spatial data", flush=True)
sp_data.write(par["output_spatial_dataset"], overwrite=True)
