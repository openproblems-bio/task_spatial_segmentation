import anndata as ad
import spatialdata as sd

## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr',
  'input_solution': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr',
  'output': 'output.zarr'
}
meta = {
  'name': 'true_labels'
}
## VIASH END

print('Reading input files', flush=True)
sdata_input = sd.read_zarr(par['input'])
sdata_solution = sd.read_zarr(par['input_solution'])

print('Copying ground truth cell_labels as prediction', flush=True)
output = sd.SpatialData(
  labels={
    'segmentation': sdata_solution['cell_labels']
  },
  tables={
    'table': ad.AnnData(
      uns={
        'dataset_id': sdata_solution.tables['table'].uns['dataset_id'],
        'method_id': meta['name']
      }
    )
  }
)

print('Writing output', flush=True)
output.write(par['output'], overwrite=True)
