import dask.array as da
import numpy as np
import os
import shutil
import spatialdata as sd
import xarray as xr
from cellpose.models import CellposeModel
from spatialdata.models import Labels2DModel

## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/common_ist.zarr',
  'output': 'resources_test/task_spatial_segmentation/mouse_brain_combined/prediction.h5ad'
}
meta = {
  'name': 'cellpose'
}
## VIASH END

print('Reading input', flush=True)
sdata = sd.read_zarr(par["input"])
image = sdata['morphology_mip']['scale0'].image.compute().to_numpy()
transformation = sdata['morphology_mip']['scale0'].image.transform.copy()

print('Initializing Cellpose model', flush=True)
model = CellposeModel()

eval_params = {k: par[k] for k in ("diameter", "flow_threshold", "niter", "min_size", "resample") if par.get(k) is not None}
print(f"Running Cellpose segmentation with parameters: {eval_params}")
masks, _, _ = model.eval(image[0], progress=True, **eval_params)

print('Cellpose segmentation finished, post-processing results', flush=True)
# Convert to smallest sufficient unsigned int dtype
max_val = masks.max()
for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
    if max_val <= np.iinfo(dtype).max:
        masks = masks.astype(dtype)
        break

print('Segmentation done, preparing output', flush=True)
sd_output = sd.SpatialData()
# Wrap masks as a single-chunk dask array with flat chunk shape for zarr v3 compat
dask_masks = da.from_array(masks, chunks=masks.shape)
data_array = xr.DataArray(dask_masks, name='segmentation', dims=('y', 'x'))
parsed = Labels2DModel.parse(data_array, transformations=transformation)
sd_output.labels['segmentation'] = parsed

print('Saving output', flush=True)
if os.path.exists(par["output"]):
    shutil.rmtree(par["output"])
sd_output.write(par["output"])
