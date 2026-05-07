import anndata as ad
import numpy as np
import spatialdata as sd
import xarray as xr
from spatialdata.models import Labels2DModel

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
sdata_solution = sd.read_zarr(par['input_solution'])

print('Relabelling ground truth cell_labels as prediction', flush=True)
gt_labels = sdata_solution['cell_labels']

# Resolve the image array (handles both DataTree and DataArray)
if isinstance(gt_labels, xr.DataTree):
    gt_array = gt_labels['scale0'].image.to_numpy()
else:
    gt_array = gt_labels.to_numpy()

# Randomly permute cell IDs while keeping background (0) unchanged.
# A correct metric (e.g. ARI) must be invariant to label permutation
rng = np.random.default_rng(42)
cell_ids = np.unique(gt_array[gt_array != 0])
perm = rng.permutation(cell_ids)
lut = np.zeros(cell_ids.max() + 1, dtype=gt_array.dtype)
lut[cell_ids] = perm
relabelled = np.where(gt_array != 0, lut[gt_array], 0)

transform = sd.transformations.get_transformation(gt_labels, get_all=True)
segmentation = Labels2DModel.parse(
    relabelled,
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
