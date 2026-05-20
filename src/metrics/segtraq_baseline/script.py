from pathlib import Path
import sys

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output": "output.h5ad",
}
meta = {
    "name": "segtraq_baseline",
}
## VIASH END

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "segtraq_common"))
if "resources_dir" in meta:
    sys.path.insert(0, meta["resources_dir"])

from adapter import initialize_segtraq, load_prepared_sdata, median_obs_metrics, safe_float, write_metric_output


def get_scalar_metrics(sdata_segtraq):
    table = sdata_segtraq.tables["table"]

    metrics = {
        "segtraq_num_cells": safe_float(table.uns.get("num_cells")),
        "segtraq_num_transcripts": safe_float(table.uns.get("num_transcripts")),
        "segtraq_num_genes": safe_float(table.uns.get("num_genes")),
        "segtraq_perc_unassigned_transcripts": safe_float(table.uns.get("perc_unassigned_transcripts")),
    }

    metrics.update(
        median_obs_metrics(
            table,
            {
                "transcript_count": "segtraq_median_transcripts_per_cell",
                "gene_count": "segtraq_median_genes_per_cell",
                "mean_transcripts_per_gene": "segtraq_median_mean_transcripts_per_gene",
                "cell_area": "segtraq_median_cell_area",
                "transcript_density": "segtraq_median_transcript_density",
                "perimeter": "segtraq_median_perimeter",
                "circularity": "segtraq_median_circularity",
                "bbox_width": "segtraq_median_bbox_width",
                "bbox_height": "segtraq_median_bbox_height",
                "extent": "segtraq_median_extent",
                "solidity": "segtraq_median_solidity",
                "convexity": "segtraq_median_convexity",
                "elongation": "segtraq_median_elongation",
                "eccentricity": "segtraq_median_eccentricity",
                "compactness": "segtraq_median_compactness",
                "num_polygons": "segtraq_median_num_polygons",
            },
        )
    )
    return metrics


print(">> Reading and preparing input files", flush=True)
sdata_solution, sdata_prediction, sdata_segtraq = load_prepared_sdata(
    par["input_prediction"],
    par["input_solution"],
)

print(">> Initializing SegTraQ and filtering transcripts", flush=True)
st = initialize_segtraq(sdata_segtraq)

print(">> Running SegTraQ baseline", flush=True)
st.run_baseline(inplace=True)

print(">> Writing scalar metric output", flush=True)
metrics = get_scalar_metrics(st.sdata)
write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
