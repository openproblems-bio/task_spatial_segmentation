import numpy as np
import xarray as xr
import anndata as ad
import spatialdata as sd
from sklearn.metrics import adjusted_rand_score

## VIASH START
par = {
    # TODO: add path
    'input_prediction': 'resources_test/task_spatial_segmentation/XXX',
    # TODO: this solution should be a list of marker genes from each superset
    'input_solution': 'resources_test/task_spatial_segmentation/XXXX',
    'output': 'output.h5ad'
}
meta = {
    'name': 'specificity_celltype_marker'
}
## VIASH END

print(">> Reading input files", flush=True)
sdata_pred = sd.read_zarr(par["input_prediction"])
# TODO: this should be reading in the list, which will not be a Zarr file
sdata_sol = sd.read_zarr(par["input_solution"])

dataset_id = sdata_sol.tables["table"].uns["dataset_id"]
method_id = sdata_pred.tables["table"].uns["method_id"]

print(">> Get ground truth cell IDs from cell_labels", flush=True)
gt_cell_ids = sdata.Labels['groundtruth_cell_labels'] 

# TODO: calculate expression of marker superset for each cell

# TODO: calculate specificity metric

print(">> Writing output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": dataset_id,
        "normalization_id": "counts",
        "method_id": method_id,
        "metric_ids": ["specificity_celltype_marker"],
        "metric_values": [float(specificity_score)],
    }
)
output.write_h5ad(par["output"], compression="gzip")