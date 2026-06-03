import numpy as np
import pandas as pd
import anndata as ad
import spatialdata as sd

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution":   "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output":           "output.h5ad",
}
meta = {
    "name": "mitotic_specificity"
}
## VIASH END

MARKERS = {
    "Mitosis_specific": [
        "AURKB", "PLK1", "CENPF", "BUB1", "MAD2L1", "PRC1", "KIF23",
        "AURKA", "MKLP1",
    ],
    "Non_mitosis_specific": [
        "CCND1", "CDKN1A", "CDKN1B", "GAS1", "NEAT1", "MALAT1", "RB1",
    ],
}

CLASS_TO_MAP = {
    "Cancer cell":     "non_mitotic",
    "Mitotic Figures": "mitotic",
}

METRIC_IDS = ["mitotic_sensitivity", "mitotic_specificity", "nonmitotic_purity"]


def _write_nan(par, dataset_id, method_id, reason):
    print(f"WARNING: {reason}  Writing NaN.", flush=True)
    ad.AnnData(uns={
        "dataset_id":       dataset_id,
        "normalization_id": "counts",
        "method_id":        method_id,
        "metric_ids":       METRIC_IDS,
        "metric_values":    [float("nan")] * len(METRIC_IDS),
    }).write_h5ad(par["output"], compression="gzip")


print(">> Reading inputs", flush=True)
sdata_pred = sd.read_zarr(par["input_prediction"])
sdata_sol  = sd.read_zarr(par["input_solution"])

dataset_id = sdata_pred.tables["table"].uns["dataset_id"]
method_id  = sdata_pred.tables["table"].uns["method_id"]

sol_obs = sdata_sol.tables["table"].obs
if "groundtruth_cell_type" not in sol_obs.columns:
    _write_nan(par, dataset_id, method_id,
               "groundtruth_cell_type not found in spatial_solution.")
    raise SystemExit(0)

# Map GT cells to mitotic / non_mitotic; exclude unmapped classes
mapped_obs = sol_obs[sol_obs["groundtruth_cell_type"].isin(CLASS_TO_MAP)].copy()
mapped_obs["group"] = mapped_obs["groundtruth_cell_type"].map(CLASS_TO_MAP)
id_to_group = mapped_obs.set_index("cell_id")["group"].to_dict()

if not id_to_group:
    _write_nan(par, dataset_id, method_id,
               "No GT cells mapped to mitotic/non_mitotic classes.")
    raise SystemExit(0)

# Build per-GT-cell count matrix from solution transcripts
print(">> Building GT cell expression from solution transcripts", flush=True)
tx = sdata_sol["transcripts"][["cell_id", "feature_name"]].compute()
tx = tx[(tx["cell_id"] != 0) & (tx["cell_id"].isin(id_to_group))]

if tx.empty:
    _write_nan(par, dataset_id, method_id,
               "No transcripts found for mapped GT cells.")
    raise SystemExit(0)

counts_df  = tx.groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0)
panel      = set(counts_df.columns)
col_idx    = {g: i for i, g in enumerate(counts_df.columns)}
counts_arr = counts_df.values

# Filter each marker group to genes present in the panel
marker_panel = {}
for grp, genes in MARKERS.items():
    in_panel = [g for g in genes if g in panel]
    marker_panel[grp] = in_panel
    n_miss = len(genes) - len(in_panel)
    if n_miss:
        print(f"   {grp}: {n_miss}/{len(genes)} genes absent from panel", flush=True)

# Boolean arrays per marker group (one value per GT cell row)
def _any_expressed(genes):
    if not genes:
        return np.zeros(len(counts_df), dtype=bool)
    idxs = [col_idx[g] for g in genes]
    return (counts_arr[:, idxs] > 0).any(axis=1)

print(">> Evaluating marker groups", flush=True)
mitotic_expr     = _any_expressed(marker_panel["Mitosis_specific"])
nonmitotic_expr  = _any_expressed(marker_panel["Non_mitosis_specific"])

# Cell group membership arrays aligned to counts_df index
groups = np.array([id_to_group[cid] for cid in counts_df.index])
is_mitotic     = groups == "mitotic"
is_nonmitotic  = groups == "non_mitotic"

n_mitotic    = is_mitotic.sum()
n_nonmitotic = is_nonmitotic.sum()

print(f"   Mitotic GT cells:     {n_mitotic:,}", flush=True)
print(f"   Non-mitotic GT cells: {n_nonmitotic:,}", flush=True)

def _safe_frac(num, denom):
    return float(num / denom) if denom > 0 else float("nan")

# mitotic_sensitivity: fraction of mitotic cells expressing ≥1 mitotic gene
#   normalised by total mitotic cell count
mitotic_sensitivity = _safe_frac(
    (is_mitotic & mitotic_expr).sum(), n_mitotic
)

# mitotic_specificity: fraction of non-mitotic cells expressing zero mitotic genes
#   mitotic signal should not leak into cancer cells
mitotic_specificity = _safe_frac(
    (is_nonmitotic & ~mitotic_expr).sum(), n_nonmitotic
)

# nonmitotic_purity: fraction of mitotic cells expressing zero non-mitotic genes
#   non-mitotic signal in mitotic cells indicates mis-assignment contamination
nonmitotic_purity = _safe_frac(
    (is_mitotic & ~nonmitotic_expr).sum(), n_mitotic
)

print(
    f">> mitotic_sensitivity={mitotic_sensitivity:.3f}  "
    f"mitotic_specificity={mitotic_specificity:.3f}  "
    f"nonmitotic_purity={nonmitotic_purity:.3f}",
    flush=True,
)

print(">> Writing output", flush=True)
ad.AnnData(uns={
    "dataset_id":       dataset_id,
    "normalization_id": "counts",
    "method_id":        method_id,
    "metric_ids":       METRIC_IDS,
    "metric_values":    [mitotic_sensitivity, mitotic_specificity, nonmitotic_purity],
}).write_h5ad(par["output"], compression="gzip")
