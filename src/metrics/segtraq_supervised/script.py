from pathlib import Path
import sys

import numpy as np

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
    "name": "segtraq_supervised",
}
## VIASH END

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segtraq_common"))
if "resources_dir" in meta:
    sys.path.insert(0, meta["resources_dir"])

from adapter import (
    finite_matrix_mean,
    initialize_segtraq,
    load_prepared_sdata,
    prepare_reference_adata,
    median_obs_metrics,
    nanmedian,
    write_metric_output,
)


print(">> Reading and preparing input files", flush=True)
sdata_solution, sdata_prediction, sdata_segtraq = load_prepared_sdata(
    par["input_prediction"],
    par["input_solution"],
)

print(">> Initializing SegTraQ and filtering transcripts", flush=True)
st = initialize_segtraq(sdata_segtraq)

print(">> Running SegTraQ supervised metrics", flush=True)
st.run_supervised(
    adata_ref=prepare_reference_adata(par["input_scrnaseq_reference"]),
    ref_cell_type=REF_CELL_TYPE_KEY,
    ref_gene_key=REF_GENE_KEY,
    ref_raw_counts_layer=REF_RAW_COUNTS_LAYER
)

table = st.sdata.tables["table"]
metrics = median_obs_metrics(
    table,
    {
        "positive_marker_recall": "segtraq_median_positive_marker_recall",
        "negative_marker_avoidance": "segtraq_median_negative_marker_avoidance",
        "marker_balanced_accuracy": "segtraq_median_marker_balanced_accuracy",
        "contamination_counts": "segtraq_median_contamination_counts",
        "contamination_fraction": "segtraq_median_contamination_fraction",
    },
)

if "contamination_fraction_matrix" in table.uns:
    metrics["segtraq_mean_contamination_fraction_matrix"] = finite_matrix_mean(
        table.uns["contamination_fraction_matrix"]
    )
else:
    metrics["segtraq_mean_contamination_fraction_matrix"] = float("nan")

if "contamination_counts_matrix" in table.uns:
    metrics["segtraq_mean_contamination_counts_matrix"] = finite_matrix_mean(
        table.uns["contamination_counts_matrix"]
    )
else:
    metrics["segtraq_mean_contamination_counts_matrix"] = float("nan")

if "mutually_exclusive_coexpression_rate" in table.uns:
    mecr = table.uns["mutually_exclusive_coexpression_rate"]
    significant_mecr = mecr[
        mecr["odds_ratio"].notna()
        & np.isfinite(mecr["odds_ratio"])
        & mecr["pvalue"].notna()
        & np.isfinite(mecr["pvalue"])
        & (mecr["pvalue"] < 0.05)
    ]
    metrics["segtraq_median_significant_mecr_odds_ratio"] = nanmedian(significant_mecr["odds_ratio"])
else:
    metrics["segtraq_median_significant_mecr_odds_ratio"] = float("nan")

print(">> Writing scalar metric output", flush=True)

write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
