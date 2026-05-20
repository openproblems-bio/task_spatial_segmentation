from pathlib import Path
import sys

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output": "output.h5ad",
}
meta = {
    "name": "segtraq_point_statistics",
}
## VIASH END

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segtraq_common"))
if "resources_dir" in meta:
    sys.path.insert(0, meta["resources_dir"])

from adapter import initialize_segtraq, load_prepared_sdata, median_obs_metrics, write_metric_output


print(">> Reading and preparing input files", flush=True)
sdata_solution, sdata_prediction, sdata_segtraq = load_prepared_sdata(
    par["input_prediction"],
    par["input_solution"],
)

print(">> Initializing SegTraQ and filtering transcripts", flush=True)
st = initialize_segtraq(sdata_segtraq)

print(">> Running SegTraQ point statistics", flush=True)
st.run_point_statistics(inplace=True)

table = st.sdata.tables["table"]
metrics = median_obs_metrics(
    table,
    {
        "perc_outside_cell_all_genes": "segtraq_median_perc_outside_cell",
        "perc_nucleus_all_genes": "segtraq_median_perc_nucleus",
        "perc_cytoplasm_all_genes": "segtraq_median_perc_cytoplasm",
        "distance_to_cell_centroid_norm_all_genes": "segtraq_median_distance_to_cell_centroid_norm",
        "distance_to_cell_membrane_norm_all_genes": "segtraq_median_distance_to_cell_membrane_norm",
        "skew_dist_to_cell_membrane_all_genes": "segtraq_median_membrane_distance_skewness",
    },
)

print(">> Writing scalar metric output", flush=True)

write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
