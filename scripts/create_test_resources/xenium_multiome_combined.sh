#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

# # remove this when you have implemented the script
# echo "TODO: replace the commands in this script with the sequence of components that you need to run to generate test_resources."
# echo "  Inside this script, you will need to place commands to generate example files for each of the 'src/api/file_*.yaml' files."
# exit 1

set -e

DATASET_ID=Xenium_V1_Human_Kidney_FFPE

RAW_DATA=resources_test/common
DATASET_DIR=resources_test/task_spatial_segmentation/$DATASET_ID

if [ -d "$DATASET_DIR" ]; then
  rm -rf "$DATASET_DIR"
fi
mkdir -p "$DATASET_DIR"

# process dataset
viash run src/data_processors/process_dataset_multimodal/config.vsh.yaml -- \
  --input_sp $RAW_DATA/Xenium_V1_Human_Kidney_FFPE/Xenium_V1_Human_Kidney_FFPE_crop.zarr \
  --output_spatial_unlabelled $DATASET_DIR/spatial_unlabelled.zarr \
  --output_spatial_solution $DATASET_DIR/spatial_solution.zarr \
  --output_scrnaseq_reference $DATASET_DIR/scrnaseq_reference.h5ad \
  --dataset_id $DATASET_ID \
  --dataset_name "Test the multimodal approach from 10X" \
  --dataset_url "https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma" \
  --dataset_reference "10.1038/s41586-023-06812-z" \
  --dataset_summary "Demonstration of gene expression and proteomce profiling for fresh frozen mouse brain on the Xenium platform" \
  --dataset_description "Demonstration of gene expression profiling for fresh frozen mouse brain" \
  --dataset_organism "homo_sapiens"

# run one method
viash run src/control_methods/random_voronoi/config.vsh.yaml -- \
    --input $DATASET_DIR/spatial_unlabelled.zarr \
    --input_solution $DATASET_DIR/spatial_solution.zarr \
    --output $DATASET_DIR/prediction.zarr

# run prediction processor
viash run src/data_processors/process_prediction/config.vsh.yaml -- \
    --input_prediction $DATASET_DIR/prediction.zarr \
    --input_spatial_unlabelled $DATASET_DIR/spatial_unlabelled.zarr \
    --output $DATASET_DIR/processed_prediction.zarr

# run one metric
viash run src/metrics/ari/config.vsh.yaml -- \
    --input_prediction $DATASET_DIR/processed_prediction.zarr \
    --input_solution $DATASET_DIR/spatial_solution.zarr \
    --output $DATASET_DIR/score.h5ad

# write manual state.yaml. this is not actually necessary but you never know it might be useful
cat > $DATASET_DIR/state.yaml << HERE
id: $DATASET_ID
spatial_unlabelled: spatial_unlabelled.zarr
spatial_solution: spatial_solution.zarr
scrnaseq_reference: scrnaseq_reference.h5ad
prediction: prediction.zarr
processed_prediction: processed_prediction.zarr
score: score.h5ad
HERE

# only run this if you have access to the openproblems-data bucket
aws s3 sync --profile op \
  resources_test/task_spatial_segmentation/mouse_brain_combined/ \
  s3://openproblems-data/resources_test/task_spatial_segmentation/mouse_brain_combined/ \
  --delete --dryrun
