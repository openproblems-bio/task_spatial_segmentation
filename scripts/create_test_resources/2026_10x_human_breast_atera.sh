#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

if [ ! -d temp/datasets/10x_xenium/2026_10x_human_breast_atera ]; then
  mkdir -p temp/datasets/10x_xenium/2026_10x_human_breast_atera
fi
if [ ! -f temp/datasets/10x_xenium/2026_10x_human_breast_atera/WTA_Preview_FFPE_Breast_Cancer_xe_outs.zip ]; then
  wget -O temp/datasets/10x_xenium/2026_10x_human_breast_atera/WTA_Preview_FFPE_Breast_Cancer_xe_outs.zip \
    https://s3-us-west-2.amazonaws.com/10x.files/samples/atera/dev/WTA_Preview_FFPE_Breast_Cancer/WTA_Preview_FFPE_Breast_Cancer_xe_outs.zip
fi

cat > /tmp/params.yaml << HERE
param_list:
  - id: 2026_10x_human_breast_atera
    input: temp/datasets/10x_xenium/2026_10x_human_breast_atera/WTA_Preview_FFPE_Breast_Cancer_xe_outs.zip
    segmentation_id:
      - cell
      - nucleus
    dataset_name: "Atera FFPE Human Breast Cancer"
    dataset_url: "https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer"
    dataset_summary: "Preview dataset showcasing the pre-commercial Atera Whole Transcriptome Assay (WTA) applied to FFPE human breast cancer tissue, profiling 18,028 genes and detecting 170,057 cells."
    dataset_description: "This human FFPE breast cancer data showcases results using the pre-commercial version of the Atera Whole Transcriptome Assay (WTA), which is currently under development. The assay is designed to closely match the Chromium Flex Apex assay in terms of content and sensitivity, and includes 18,028 genes. A single 5µm FFPE section of breast cancer tissue (DCIS Grade 3, T1c N0 M0) was analyzed, yielding 170,057 detected cells with a median of 2,116 transcripts per cell and 624,095,990 total high-quality decoded transcripts across 58.9 million µm² of tissue area. Output files are formatted to closely resemble Xenium Onboard Analysis v4 file formats."
    dataset_organism: homo_sapiens
    crop_region_min_x: 5000
    crop_region_max_x: 6000
    crop_region_min_y: 5000
    crop_region_max_y: 6000

publish_dir: resources_test/common
output_dataset: '\$id/dataset.zarr'
output_state: '\$id/state.yaml'
HERE

# convert to zarr
nextflow run . \
  -main-script target/nextflow/datasets/workflows/process_tenx_xenium/main.nf \
  -profile docker \
  -resume \
  -params-file /tmp/params.yaml

# sync to s3
aws s3 sync --profile op \
  "resources_test/common/2026_10x_human_breast_atera" \
  "s3://openproblems-data/resources_test/common/2026_10x_human_breast_atera" \
  --delete --dryrun