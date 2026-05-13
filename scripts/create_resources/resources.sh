#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

# remove this when you have implemented the script
echo "TODO: once the 'process_datasets' workflow is implemented, update this script to use it."
echo "  Step 1: replace 'task_spatial_segmentation' with the name of the task in the following command."
echo "  Step 2: replace the rename keys parameters to fit your process_dataset inputs"
echo "  Step 3: replace the settings parameter to fit your process_dataset outputs"
echo "  Step 4: remove this message"
exit 1

cat > /tmp/params.yaml << 'HERE'
input_states: s3://openproblems-data/resources/datasets/**/state.yaml
rename_keys: 'input_spatial_unlabelled:output_spatial_unlabelled,input_spatial_solution:output_spatial_solution,input_scrnaseq_reference:output_scrnaseq_reference'
output_state: '$id/state.yaml'
settings: '{"output_spatial_unlabelled": "$id/output_spatial_unlabelled.zarr", "output_spatial_solution": "$id/output_spatial_solution.zarr", "output_scrnaseq": "$id/output_scrnaseq.h5ad"}'
publish_dir: s3://openproblems-data/resources/task_spatial_segmentation/datasets/
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
  --labels task_spatial_segmentation,process_datasets
