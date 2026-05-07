import numpy as np
import xarray as xr
import anndata as ad
import spatialdata as sd
from sklearn.metrics import adjusted_rand_score

## VIASH START
par = {
    'input_prediction': 'resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr',
    'input_solution': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr',
    'output': 'output.h5ad'
}
meta = {
    'name': 'ari'
}
## VIASH END

print(">> Reading input files", flush=True)
sdata_pred = sd.read_zarr(par["input_prediction"])
sdata_sol = sd.read_zarr(par["input_solution"])

dataset_id = sdata_sol.tables["table"].uns["dataset_id"]
method_id = sdata_pred.tables["table"].uns["method_id"]

print(">> Loading transcripts in global coordinate system", flush=True)
transcripts = sd.transform(sdata_sol["transcripts"], to_coordinate_system="global")

def lookup_labels(label_element, transcripts_global):
    """Look up label image values at transcript coordinates."""
    trans = sd.transformations.get_transformation(
        label_element, get_all=True
    )["global"].inverse()
    transcripts_local = sd.transform(transcripts_global, trans, "global")
    y = transcripts_local.y.compute().to_numpy(dtype=np.int64)
    x = transcripts_local.x.compute().to_numpy(dtype=np.int64)
    if isinstance(label_element, xr.DataTree):
        img = label_element["scale0"].image.to_numpy()
    else:
        img = label_element.to_numpy()
    y = np.clip(y, 0, img.shape[0] - 1)
    x = np.clip(x, 0, img.shape[1] - 1)
    return img[y, x]

print(">> Looking up ground truth cell IDs from cell_labels", flush=True)
gt_cell_ids = lookup_labels(sdata_sol["cell_labels"], transcripts)

print(">> Looking up predicted cell IDs from segmentation", flush=True)
pred_cell_ids = lookup_labels(sdata_pred["segmentation"], transcripts)

print(">> Computing ARI", flush=True)
# Exclude transcripts that are background in both
mask = (gt_cell_ids != 0) | (pred_cell_ids != 0)
print(f"  Transcripts used: {mask.sum()} / {len(mask)}", flush=True)
print(f"  GT unique cells: {len(np.unique(gt_cell_ids[mask]))}", flush=True)
print(f"  Pred unique cells: {len(np.unique(pred_cell_ids[mask]))}", flush=True)

ari_score = adjusted_rand_score(gt_cell_ids[mask], pred_cell_ids[mask])
print(f"  ARI = {ari_score:.4f}", flush=True)

print(">> Writing output", flush=True)
output = ad.AnnData(
    uns={
        "dataset_id": dataset_id,
        "normalization_id": "counts",
        "method_id": method_id,
        "metric_ids": ["ari"],
        "metric_values": [float(ari_score)],
    }
)
output.write_h5ad(par["output"], compression="gzip")