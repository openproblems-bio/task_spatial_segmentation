import anndata as ad
import numpy as np
import spatialdata as sd
import xarray as xr
from scipy.spatial import cKDTree
from spatialdata.models import Labels2DModel

## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr',
  'input_solution': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr',
  'output': 'output.zarr'
}
meta = {
  'name': 'random_voronoi'
}
## VIASH END

print('Reading input files', flush=True)
sdata_solution = sd.read_zarr(par['input_solution'])

print('Building random Voronoi segmentation', flush=True)
gt_labels = sdata_solution['cell_labels']

# Resolve the image array (handles both DataTree and DataArray)
if isinstance(gt_labels, xr.DataTree):
    gt_array = gt_labels['scale0'].image.to_numpy()
else:
    gt_array = gt_labels.to_numpy()

H, W = gt_array.shape
rng = np.random.default_rng(42)

# Use the same number of cells as in the ground truth
n_cells = len(np.unique(gt_array)) - 1  # subtract 1 for background (0)
n_cells = max(n_cells, 1)

# Place seed points uniformly at random
seeds_y = rng.integers(0, H, size=n_cells)
seeds_x = rng.integers(0, W, size=n_cells)
seeds = np.column_stack([seeds_y, seeds_x])

# Assign every pixel to the nearest seed via KD-tree
print(f'Assigning {H * W} pixels to {n_cells} random seeds', flush=True)
yx = np.mgrid[0:H, 0:W].reshape(2, -1).T  # (H*W, 2)
tree = cKDTree(seeds)
_, idx = tree.query(yx, workers=-1)
voronoi_array = (idx + 1).reshape(H, W).astype(np.int32)  # labels start at 1

# Preserve the same coordinate system and transform as the GT labels
transform = sd.transformations.get_transformation(gt_labels, get_all=True)
segmentation = Labels2DModel.parse(
    voronoi_array,
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
