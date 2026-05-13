#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

publish_dir="s3://openproblems-data/resources/datasets"

cat > /tmp/params.yaml << HERE
param_list:

  - id: "10x_xenium/10x_mouse_breast_cancer_xenium/rep1"
    input: https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif
    dataset_name: "Xenium FFPE Human Breast Cancer Replicate 1"
    dataset_url: "https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast"
    dataset_summary: "The Xenium data was registered with post-Xenium IF / H&E images (workflow is non-destructive to the tissue) and integrated with Chromium and Visium data."
    dataset_description: "Two formalin-fixed & paraffin-embedded (FFPE) breast cancer tissue blocks were obtained from Discovery Life Sciences. Sample #1 was annotated by a pathologist to be T2N1M0, Stage II-B, ER+/HER2+/PR−. Sample #2 was characterized as stage pT2 pN1a pMX, ER−/HER2+/PR−. Corresponding dissociated tumor cells for Sample #1, fresh frozen (FF) in liquid nitrogen, were also sampled from the same 2.5 cm biopsy. For the Chromium Flex workflow, two 25 μm curls were pooled as a single replicate. 5 μm sections from Sample #1 were taken from the FFPE tissue using a microtome. Two replicate 5 μm sections were taken each for Visium CytAssist and Xenium. A 5 μm section was also taken from Sample #2 for Xenium."
    dataset_organism: "homo_sapiens"

output_dataset: "\$id/dataset.zarr"
output_state: "\$id/state.yaml"
publish_dir: "$publish_dir"
HERE

tw launch https://github.com/openproblems-bio/task_spatial_segmentation.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/process_datasets/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels task_template,process_datasets
