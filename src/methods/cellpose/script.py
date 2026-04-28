import anndata as ad
import numpy as np
import os
import pandas as pd
import shutil
import spatialdata as sd
import xarray as xr
from cellpose.models import CellposeModel
from spatialdata.models import Labels2DModel

## VIASH START
par = {
  'input': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_dataset.zarr',
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
model = CellposeModel()

eval_params = {k: par[k] for k in ("diameter", "flow_threshold", "niter", "min_size", "resample") if par.get(k) is not None}
print(f"Running Cellpose segmentation with parameters: {eval_params}")
masks, _, _ = model.eval(image[0], progress=True, **eval_params)

print('Cellpose segmentation finished, post-processing results', flush=True)
masks = convert_to_lower_dtype(masks)

print('Segmentation done, preparing output', flush=True)
sd_output = sd.SpatialData()
data_array = xr.DataArray(masks, name='segmentation', dims=('y', 'x'))
parsed = Labels2DModel.parse(data_array, transformations=transformation)
sd_output.labels['segmentation'] = parsed

cell_ids = np.unique(masks)[1:]  # exclude background (0)
table = ad.AnnData(
  obs=pd.DataFrame(
    {'cell_id': cell_ids.astype(str), 'region': 'segmentation'},
    index=cell_ids.astype(str),
  ),
  uns={
    'dataset_id': sdata.tables['table'].uns['dataset_id'],
    'method_id': meta['name']
  }
)
sd_output.tables['table'] = table

print('Saving output', flush=True)
if os.path.exists(par["output"]):
    shutil.rmtree(par["output"])
sd_output.write(par["output"])
