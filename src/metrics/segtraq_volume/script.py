from pathlib import Path
import sys

REF_CELL_TYPE_KEY = "cell_type"
REF_GENE_KEY = "feature_name"
REF_RAW_COUNTS_LAYER = "counts"

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "input_scrnaseq_reference": "resources_test/task_spatial_segmentation/mouse_brain_combined/scrnaseq_reference.h5ad",
    "output": "output.h5ad",
}
meta = {
    "name": "segtraq_volume",
}
## VIASH END

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segtraq_common"))
if "resources_dir" in meta:
    sys.path.insert(0, meta["resources_dir"])

from adapter import (
    all_nan_metrics,
    initialize_segtraq,
    load_prepared_sdata,
    prepare_reference_adata,
    median_obs_metrics,
    write_metric_output,
)


METRIC_IDS = [
    "segtraq_median_similarity_top_bottom",
    "segtraq_median_vertical_signal_integrity",
    "segtraq_median_heterotypic_overlap_area",
    "segtraq_median_heterotypic_overlap_fraction",
]

print(">> Reading and preparing input files", flush=True)
sdata_solution, sdata_prediction, sdata_segtraq = load_prepared_sdata(
    par["input_prediction"],
    par["input_solution"],
)

print(">> Initializing SegTraQ and filtering transcripts", flush=True)
st = initialize_segtraq(sdata_segtraq)
metrics = all_nan_metrics(METRIC_IDS)

print(">> Running SegTraQ volume metrics", flush=True)
# Disabled: ovrlpy.Ovrlp currently triggers a segmentation fault.
# The code works on the same data in notebooks but crashes in this component.

# st.run_volume(
#     adata_ref=prepare_reference_adata(par["input_scrnaseq_reference"]),
#     ref_cell_type=REF_CELL_TYPE_KEY,
#     ref_gene_key=REF_GENE_KEY,
#     ref_raw_counts_layer=REF_RAW_COUNTS_LAYER
# )

metrics = median_obs_metrics(
    table,
    {
        "similarity_top_bottom": "segtraq_median_similarity_top_bottom",
        "vertical_signal_integrity": "segtraq_median_vertical_signal_integrity",
        "heterotypic_overlap_area": "segtraq_median_heterotypic_overlap_area",
        "heterotypic_overlap_fraction": "segtraq_median_heterotypic_overlap_fraction",
    },
)

print(">> Writing scalar metric output", flush=True)
print(metrics)
write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
