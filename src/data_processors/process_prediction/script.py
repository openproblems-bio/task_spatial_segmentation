import numpy as np
import xarray as xr
import anndata as ad
import pandas as pd
import spatialdata as sd
import scanpy as sc

## VIASH START
par = {
    'input_prediction': 'resources_test/task_spatial_segmentation/mouse_brain_combined/prediction.zarr',
    'input_spatial_unlabelled': 'resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_unlabelled.zarr',
    'output': 'output.zarr'
}
## VIASH END

print(">> Reading input files", flush=True)
sdata_pred = sd.read_zarr(par["input_prediction"])
sdata_sp = sd.read_zarr(par["input_spatial_unlabelled"])

dataset_id = sdata_sp.tables["table"].uns["dataset_id"]
method_id = sdata_pred.tables["table"].uns["method_id"]

print(">> Transforming transcripts to global coordinate system", flush=True)
transcripts = sd.transform(sdata_sp["transcripts"], to_coordinate_system="global")

# Adjust for any translation applied to the segmentation
trans = sd.transformations.get_transformation(
    sdata_pred["segmentation"], get_all=True
)["global"].inverse()
transcripts = sd.transform(transcripts, trans, "global")

print(">> Assigning transcripts to cells via label image lookup", flush=True)
y_coords = transcripts.y.compute().to_numpy(dtype=np.int64)
x_coords = transcripts.x.compute().to_numpy(dtype=np.int64)

if isinstance(sdata_pred["segmentation"], xr.DataTree):
    label_image = sdata_pred["segmentation"]["scale0"].image.to_numpy()
else:
    label_image = sdata_pred["segmentation"].to_numpy()

# Clip coordinates to valid label image bounds
y_coords = np.clip(y_coords, 0, label_image.shape[0] - 1)
x_coords = np.clip(x_coords, 0, label_image.shape[1] - 1)

cell_ids = label_image[y_coords, x_coords]

# NOTE: Is it useful to build a cxg count matrix? Is this used downstream?
print(">> Building cell x gene count matrix", flush=True)
feature_names = transcripts["feature_name"].compute().to_numpy()

transcript_df = pd.DataFrame({"cell_id": cell_ids, "feature_name": feature_names})
# Remove background (cell_id == 0)
transcript_df = transcript_df[transcript_df["cell_id"] != 0]

count_matrix = (
    transcript_df.groupby(["cell_id", "feature_name"])
    .size()
    .unstack(fill_value=0)
)

obs = pd.DataFrame(
    {"cell_id": count_matrix.index.astype(str), "region": "segmentation"},
    index=count_matrix.index.astype(str),
)
var = pd.DataFrame(index=count_matrix.columns.astype(str))
var.index.name = "feature_name"

table = ad.AnnData(X=count_matrix.values.astype(np.float32), obs=obs, var=var)
table.layers["counts"] = table.X.copy()

print(">> Normalizing counts", flush=True)
sc.pp.normalize_total(table, target_sum=1e4)
table.layers["normalized"] = table.X.copy()

sc.pp.log1p(table)
table.layers["normalized_log"] = table.X.copy()

sc.pp.scale(table)
table.layers["normalized_log_scaled"] = table.X.copy()

table.uns["dataset_id"] = dataset_id
table.uns["method_id"] = method_id
table.uns["spatialdata_attrs"] = {
    "instance_key": "cell_id",
    "region": ["segmentation"],
    "region_key": "region",
}

print(">> Writing output", flush=True)
output = sd.SpatialData(
    labels={"segmentation": sdata_pred["segmentation"]},
    tables={"table": table},
)
output.write(par["output"], overwrite=True)
