#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

cat > /tmp/params.yaml << HERE
param_list:
  - id: tenx_xenium_groundtruth/cervical_cancer
    input: s3://hca-op-spatial/datasets/gt_annotated_data/Xenium_Prime_Cervical_Cancer_FFPE_Aligned.zarr
    dataset_name: 10X Xenium - Cervical Cancer
    dataset_url: https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-cervical-cancer
    dataset_summary: Gene expression library for 5K Xenium Prime panel + 100 custom genes on cervical cancer sample
    dataset_description: Xenium Prime 5K In Situ Gene Expression with Cell Segmentation data for human cervical cancer (FFPE) using the Xenium Prime 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes.
    dataset_organism: homo_sapiens

publish_dir: temp
output_dataset: '\$id/dataset.zarr'
output_state: '\$id/state.yaml'
HERE

# convert to zarr
nextflow run . \
  -main-script target/nextflow/datasets/loaders/tenx_xenium_groundtruth/main.nf \
  -profile docker \
  -resume \
  -params-file /tmp/params.yaml

# sync to s3
# aws s3 sync --profile op \
#   "resources_test/datasets/2023_10x_mouse_brain_xenium_rep1" \
#   "s3://openproblems-data/resources_test/common/2023_10x_mouse_brain_xenium_rep1" \
#   --delete --dryrun
