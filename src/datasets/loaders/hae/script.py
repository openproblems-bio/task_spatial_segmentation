import spatialdata as sd
import anndata as ad
from spatialdata_io import xenium
import shutil
import os
import zipfile
import tempfile
import tifffile as tiff
import json

## VIASH START
par = {
    "input": "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif",
    "dataset_id": "value",
    "dataset_name": "value",
    "dataset_url": "value",
    "dataset_reference": "value",
    "dataset_summary": "value",
    "dataset_description": "value",
    "dataset_organism": "value",
    "output": "temp/datasets/hae/breast/breast.tiff"
}
meta = {
    "cpus": 1,
}

## VIASH END

# Download the data if it's a download url, extract the data if it's a zip file
par_input = par["input"]
with tempfile.TemporaryDirectory() as tmpdirname:
    if par_input.startswith("http"):
        print(f"Downloading data to {tmpdirname}", flush=True)
        file_name = par_input.split("/")[-1]
        # download the data
        os.system(f"wget {par['input']} -O {tmpdirname}/{file_name}")
        par_input = tmpdirname + "/" + file_name

    if zipfile.is_zipfile(par_input):
        print(f"Extracting input zip to {tmpdirname}", flush=True)
        with zipfile.ZipFile(par_input, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)
            par_input = tmpdirname

    # read the data
    img = tiff.imread(par_input)

    metadata = {
        "dataset_id": par["dataset_id"],
        "dataset_name": par["dataset_name"],
        "dataset_url": par["dataset_url"],
        "dataset_reference": par["dataset_reference"],
        "dataset_summary": par["dataset_summary"],
        "dataset_description": par["dataset_description"],
        "dataset_organism": par["dataset_organism"],
        "segmentation_id": par["segmentation_id"],
    }

    print(f"Writing to '{par['output']}'", flush=True)
    if os.path.exists(par["output"]):
        shutil.rmtree(par["output"])

    tiff.imwrite(
        par["output"],
        img,
        description=json.dumps(metadata),
        metadata=metadata,
    )