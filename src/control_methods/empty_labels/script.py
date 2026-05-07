import anndata as ad
import numpy as np
import spatialdata as sd
from spatialdata.models import Labels2DModel

## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr',
  'input_solution': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr',
  'output': 'output.zarr'
}
meta = {
  'name': 'empty_labels'
}
## VIASH END

print('Reading input files', flush=True)
sdata_solution = sd.read_zarr(par['input_solution'])

print('Creating empty (all-zero) segmentation labels', flush=True)
gt_labels = sdata_solution['cell_labels']

# Resolve the image array (handles both DataTree and DataArray)
import xarray as xr
if isinstance(gt_labels, xr.DataTree):
    gt_array = gt_labels['scale0'].image.to_numpy()
else:
    gt_array = gt_labels.to_numpy()

empty_array = np.zeros_like(gt_array)

# Preserve the same coordinate system and transform as the GT labels
transform = sd.transformations.get_transformation(gt_labels, get_all=True)
segmentation = Labels2DModel.parse(
    empty_array,
    transformations=transform,
)

output = sd.SpatialData(
  labels={
    'segmentation': segmentation
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
