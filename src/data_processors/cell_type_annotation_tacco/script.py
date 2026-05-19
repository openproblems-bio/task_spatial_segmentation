import anndata as ad
import numpy as np
import spatialdata as sd
import tacco

## VIASH START
par = {
    'input_processed_prediction': 'resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr',
    'input_scrnaseq_reference': 'resources_test/task_spatial_segmentation/mouse_brain_combined/scrnaseq_reference.h5ad',
    'output': 'output.h5ad',
}
meta = {
    'name': 'tacco',
}
## VIASH END

print('Reading inputs', flush=True)
sdata_pred = sd.read_zarr(par['input_processed_prediction'])
adata_sc = ad.read_h5ad(par['input_scrnaseq_reference'])

table = sdata_pred.tables['table']

if table.n_obs == 0:
    print('No cells detected in prediction — skipping annotation', flush=True)
    cell_types = []
else:
    # remap Ensembl IDs to gene symbols in-place if needed
    if 'feature_name' in adata_sc.var.columns:
        adata_sc.var_names = adata_sc.var['feature_name'].values
        adata_sc = adata_sc[:, ~adata_sc.var_names.duplicated()].copy()

    if 'counts' not in adata_sc.layers:
        raise ValueError("scRNA-seq reference is missing the 'counts' layer.")

    common_genes = sorted(set(table.var_names) & set(adata_sc.var_names))
    if len(common_genes) == 0:
        raise ValueError('No common genes between prediction cells and scRNA-seq reference.')
    print(f'Using {len(common_genes)} common genes', flush=True)

    adata_sp_sub = table[:, common_genes].copy()
    adata_sp_sub.X = adata_sp_sub.layers['counts']
    adata_sc_sub = adata_sc[:, common_genes].copy()
    adata_sc_sub.X = adata_sc_sub.layers['counts']

    print('Running TACCO annotation', flush=True)
    cell_type_annotation = tacco.tl.annotate(
        adata=adata_sp_sub,
        reference=adata_sc_sub,
        annotation_key='cell_type',
    )
    best_type_idx = np.argmax(cell_type_annotation.values, axis=1)
    cell_types = cell_type_annotation.columns[best_type_idx].tolist()

print('Writing output', flush=True)
output = ad.AnnData(obs={'cell_type': cell_types, 'cell_id': table.obs['cell_id'].values})
output.uns['dataset_id'] = table.uns['dataset_id']
output.uns['method_id'] = table.uns['method_id']
output.write_h5ad(par['output'], compression='gzip')