from pathlib import Path
import sys

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output": "output.h5ad",
}
meta = {
    "name": "segtraq_clustering_stability",
}
## VIASH END

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segtraq_common"))
if "resources_dir" in meta:
    sys.path.insert(0, meta["resources_dir"])

from adapter import initialize_segtraq, load_prepared_sdata, prepare_clustering_table, safe_float, write_metric_output


print(">> Reading and preparing input files", flush=True)
sdata_solution, sdata_prediction, sdata_segtraq = load_prepared_sdata(
    par["input_prediction"],
    par["input_solution"],
)

print(">> Initializing SegTraQ and filtering transcripts", flush=True)
st = initialize_segtraq(sdata_segtraq)

print(">> Preparing PCA/neighbors for clustering stability", flush=True)
st.sdata.tables["table"] = prepare_clustering_table(st.sdata.tables["table"])

print(">> Running SegTraQ clustering stability", flush=True)
st.run_clustering_stability(
    inplace=True
)

table = st.sdata.tables["table"]
metrics = {
    "segtraq_cluster_connectedness": safe_float(table.uns.get("cluster_connectedness")),
    "segtraq_silhouette_score": safe_float(table.uns.get("silhouette_score")),
    "segtraq_mean_purity": safe_float(table.uns.get("mean_purity")),
    "segtraq_mean_ari": safe_float(table.uns.get("mean_ari")),
}

print(">> Writing scalar metric output", flush=True)

write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
