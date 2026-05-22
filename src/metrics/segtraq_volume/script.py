from pathlib import Path
import sys

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
    median_obs_metrics,
    n_components_from_cell_types,
    run_ovrlpy,
    write_metric_output,
    run_label_transfer_and_markers
)


METRIC_IDS = [
    "segtraq_median_similarity_top_bottom_z",
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


print(">> Running label transfer and marker detection", flush=True)
cell_type_key = "transferred_cell_type"
markers = run_label_transfer_and_markers(
    st,
    par["input_scrnaseq_reference"],
    ref_cell_type="cell_type",
)
table = st.sdata.tables["table"]

print(">> Running SegTraQ volume metrics", flush=True)
st.vl.similarity_top_bottom(inplace=True)

n_comp = n_components_from_cell_types(table) #requires cell type labels, do later

# ovrlpy triggers a segmentation fault
# if n_comp is not None:
#     vsi_map = run_ovrlpy(st.sdata, n_comp=n_comp)
#     print(vsi_map)
#     st.vl.vertical_signal_integrity_per_cell(vsi_map=vsi_map, inplace=True)
#     print("Test")
# else:
#     print(">> Skipping vertical signal integrity because no cell-type labels are available", flush=True)

z_shape_keys = sorted(key for key in st.sdata.shapes if key.startswith("cell_boundaries_z"))
if len(z_shape_keys) > 1:
    cell_type_key = next(
        (key for key in ("transferred_cell_type", "cell_type") if key in table.obs),
        None,
    )
    if cell_type_key is not None:
        st.vl.fraction_heterotypic_overlap(
            cell_type_key=cell_type_key,
            shapes_key_list=z_shape_keys,
            inplace=True,
        )
    else:
        print(">> Skipping heterotypic overlap because no cell-type labels are available", flush=True)
else:
    print(">> Skipping heterotypic overlap because fewer than two z shape layers are available", flush=True)

metrics = median_obs_metrics(
    table,
    {
        "cosine_sim_top_bottom_z": "segtraq_median_similarity_top_bottom_z",
        "vertical_signal_integrity": "segtraq_median_vertical_signal_integrity",
        "heterotypic_overlap_area": "segtraq_median_heterotypic_overlap_area",
        "heterotypic_overlap_fraction": "segtraq_median_heterotypic_overlap_fraction",
    },
)

print(">> Writing scalar metric output", flush=True)

write_metric_output(par["output"], sdata_solution, sdata_prediction, metrics)
