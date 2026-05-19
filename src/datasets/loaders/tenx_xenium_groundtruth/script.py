import spatialdata as sd
import shutil
import os

## VIASH START
par = {
    "input": "resources/datasets/gt_annotated_data/Xenium_Prime_Cervical_Cancer_FFPE_Aligned.zarr",
    "segmentation_id": [
        "cell",
        "nucleus",
    ],
    "dataset_id": "value",
    "dataset_name": "value",
    "dataset_url": "value",
    "dataset_reference": "value",
    "dataset_summary": "value",
    "dataset_description": "value",
    "dataset_organism": "value",
    "output": "temp/datasets/10x_xenium/cervical_cancer/spatialData.zarr"
}
meta = {
    "cpus": 1,
}
## VIASH END

# read the data
sdata = sd.read_zarr(
    store=par["input"],
    selection=None
)

print("Raw data input: ", sdata, flush=True)

print("Add uns to table", flush=True)
new_uns = {
    "dataset_id": par["dataset_id"],
    "dataset_name": par["dataset_name"],
    "dataset_url": par["dataset_url"],
    "dataset_reference": par["dataset_reference"],
    "dataset_summary": par["dataset_summary"],
    "dataset_description": par["dataset_description"],
    "dataset_organism": par["dataset_organism"],
    "segmentation_id": par["segmentation_id"],
}
for key, value in new_uns.items():
    sdata.tables["table"].uns[key] = value

# add ground truth cell labels
## these annotations were derived by Caner Ercan
sdata.tables["table"].obs["groundtruth_celltype"] = sdata.tables["table"].obs.pop("histoplus_cell_class")

# rename Images
## rename raw images to accomodate format
sdata.images['image'] = sdata.images['morphology_focus']
## rm morphology_focus
_ = sdata.images.pop("morphology_focus")
## rename hne image 
sdata.images['he_image'] = sdata.images['hne_aligned']
## rm hne_aligned
_ = sdata.images.pop("hne_aligned")

# rename Labels
## add ground truth to cell labels
## these annotations were derived by Caner Ercan
sdata.Labels['groundtruth_cell_labels'] = sdata.tables['table'].obs.pop('histoplus_cell_class')

print(f"Output: {sdata}", flush=True)

print(f"Writing to '{par['output']}'", flush=True)
if os.path.exists(par["output"]):
    shutil.rmtree(par["output"])

print(f"Output: {sdata}")

sdata.write(par["output"])
