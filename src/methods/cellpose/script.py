import anndata as ad
import numpy as np
import os
import shutil
import spatialdata as sd
import xarray as xr
from cellpose.models import CellposeModel
from spatialdata.models import Labels2DModel
import torch

# Check whether a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr',
  'output': 'prediction.zarr'
}
meta = {
  'name': 'cellpose'
}
## VIASH END

# TODO: move to helper file
def convert_to_lower_dtype(arr):
    max_val = arr.max()
    if max_val <= np.iinfo(np.uint8).max:
        new_dtype = np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        new_dtype = np.uint16
    elif max_val <= np.iinfo(np.uint32).max:
        new_dtype = np.uint32
    else:
        new_dtype = np.uint64

    return arr.astype(new_dtype)

print('Reading input', flush=True)
sdata = sd.read_zarr(par["input"])
image = sdata['morphology_mip']['scale0'].image.compute().to_numpy()
transformation = sdata['morphology_mip']['scale0'].image.transform.copy()

print('Initializing Cellpose model', flush=True)
model = CellposeModel(gpu=torch.cuda.is_available())

eval_params = {k: par[k] for k in ("diameter", "flow_threshold", "niter", "min_size", "resample") if par.get(k) is not None}
print('Running Cellpose segmentation with parameters:', eval_params, flush=True)
masks, _, _ = model.eval(image[0], progress=True, **eval_params)

print('Cellpose segmentation finished, post-processing results', flush=True)
masks = convert_to_lower_dtype(masks)

print('Creating output data structure', flush=True)
sd_output = sd.SpatialData(
  labels={
    'segmentation': Labels2DModel.parse(
      xr.DataArray(masks, name='segmentation', dims=('y', 'x')),
      transformations=transformation
    )
  },
  tables={
    'table': ad.AnnData(
      uns={
        'dataset_id': sdata.tables['table'].uns['dataset_id'],
        'method_id': meta['name']
      }
    )
  }
)

print('Saving output', flush=True)
if os.path.exists(par["output"]):
    shutil.rmtree(par["output"])
sd_output.write(par["output"])
