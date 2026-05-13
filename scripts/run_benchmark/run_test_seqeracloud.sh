#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

resources_test_s3=s3://openproblems-data/resources_test/task_spatial_segmentation
publish_dir_s3="s3://hca-op-spatial/temp/results/$(date +%Y-%m-%d_%H-%M-%S)"

# write the parameters to file
cat > /tmp/params.yaml << HERE
id: mouse_brain_combined
input_spatial_unlabelled: $resources_test_s3/mouse_brain_combined/spatial_unlabelled.zarr
input_spatial_solution: $resources_test_s3/mouse_brain_combined/spatial_solution.zarr
input_scrnaseq_reference: $resources_test_s3/mouse_brain_combined/scrnaseq_reference.h5ad
output_state: "state.yaml"
publish_dir: $publish_dir_s3
HERE

tw launch https://github.com/openproblems-bio/task_spatial_segmentation.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 8386213183400 \
  --compute-env 7Odt43ln9XureGja6Frdm7 \
  --params-file /tmp/params.yaml \
  --config src/base/labels_tw.config \
  --labels task_spatial_segmentation,test
