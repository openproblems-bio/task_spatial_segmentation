import txsim as tx
import numpy as np
import os
import yaml
import spatialdata as sd
import anndata as ad
import shutil
import numpy as np
from spatialdata.models import Labels2DModel
import xarray as xr

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

## VIASH START
par = {
  "input": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr",
  "output": "prediction.zarr"
}
meta = {
  'name': 'binning'
}
## VIASH END

hyperparameters = par.copy()

hyperparameters = {k:(v if v != "None" else None) for k,v in hyperparameters.items()}
del hyperparameters['input']
del hyperparameters['output']

sdata = sd.read_zarr(par["input"])

if "image" in sdata.images:
    input_image_name = "image"
elif "morphology_mip" in sdata.images:
    print("WARNING: 'morphology_mip' image found but expected 'image'. Using 'morphology_mip' as fallback.", flush=True)
    input_image_name = "morphology_mip"
else:
    raise ValueError("No suitable image found in spatial data. Expected 'image' or 'morphology_mip'.")

image = sdata[input_image_name]['scale0'].image.compute().to_numpy()
transformation = sdata[input_image_name]['scale0'].image.transform.copy()
img_arr = tx.preprocessing.segment_binning(image[0], hyperparameters['bin_size'])   ### TOdo find the optimal bin_size
image = convert_to_lower_dtype(img_arr)

sd_output = sd.SpatialData(
  labels={
    'segmentation': Labels2DModel.parse(
      xr.DataArray(image, name='segmentation', dims=('y', 'x')),
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


print(sd_output)

print("Writing output", flush=True)
if os.path.exists(par["output"]):
  shutil.rmtree(par["output"])
sd_output.write(par["output"])

