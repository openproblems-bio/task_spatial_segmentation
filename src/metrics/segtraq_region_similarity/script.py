from pathlib import Path
import sys

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output": "output.h5ad",
}
meta = {
    "name": "segtraq_region_similarity",
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

print(">> Running SegTraQ region similarity", flush=True)
st.run_region_similarity(n_jobs=1, parallel_backend="threading")

table = st.sdata.tables["table"]
metrics = median_obs_metrics(
    table,
    {
        "iou": "segtraq_median_nucleus_cell_iou",
        "nucleus_fraction": "segtraq_median_nucleus_fraction",
        "similarity_nucleus_cell": "segtraq_median_similarity_nucleus_cell",
        "similarity_nucleus_cytoplasm": "segtraq_median_similarity_nucleus_cytoplasm",
        "similarity_center_border": "segtraq_median_similarity_center_border",
        "similarity_border_neighborhood": "segtraq_median_similarity_border_neighborhood",
        "border_admixture_score": "segtraq_median_border_admixture_score",
    },
)

print(">> Writing scalar metric output", flush=True)

write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
