from pathlib import Path
import sys

import numpy as np

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
    median_obs_metrics,
    nanmedian,
    run_label_transfer_and_markers,
    write_metric_output,
)


print(">> Reading and preparing input files", flush=True)
sdata_solution, sdata_prediction, sdata_segtraq = load_prepared_sdata(
    par["input_prediction"],
    par["input_solution"],
)

print(">> Initializing SegTraQ and filtering transcripts", flush=True)
st = initialize_segtraq(sdata_segtraq)

print(">> Running label transfer and marker detection", flush=True)
cell_type_key = "transferred_cell_type"
markers = run_label_transfer_and_markers(
    st,
    par["input_scrnaseq_reference"],
    ref_cell_type="cell_type",
)

print(">> Running SegTraQ supervised metrics", flush=True)
st.run_supervised(
    markers=markers,
)

table = st.sdata.tables["table"]
metrics = median_obs_metrics(
    table,
    {
        "positive_marker_recall": "segtraq_median_positive_marker_recall",
        "negative_marker_avoidance": "segtraq_median_negative_marker_avoidance",
        "marker_balanced_accuracy": "segtraq_median_marker_balanced_accuracy",
        "negative_marker_contamination_counts": "segtraq_median_negative_marker_contamination_counts",
        "negative_marker_contamination_fraction": "segtraq_median_negative_marker_contamination_fraction",
    },
)

if "negative_marker_contamination" in table.uns:
    metrics["segtraq_mean_negative_marker_contamination"] = finite_matrix_mean(
        table.uns["negative_marker_contamination"]
    )
else:
    metrics["segtraq_mean_negative_marker_contamination"] = float("nan")

if "negative_marker_contamination_binary" in table.uns:
    metrics["segtraq_mean_negative_marker_contamination_binary"] = finite_matrix_mean(
        table.uns["negative_marker_contamination_binary"]
    )
else:
    metrics["segtraq_mean_negative_marker_contamination_binary"] = float("nan")

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
st.sdata.write("sdata.zarr", overwrite=True)
write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
