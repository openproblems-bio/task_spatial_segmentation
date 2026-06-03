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
    "name": "marker_specificity"
}
## VIASH END

MARKERS = {
    "Tumour_Epithelial": [
        "EPCAM", "CDH1", "TACSTD2", "CD24", "SFN", "PERP", "PKP1", "JUP", "DSP",
        "KRT5", "KRT6A", "KRT7", "KRT8", "KRT10", "KRT13", "KRT14", "KRT16",
        "KRT17", "KRT18", "KRT19", "TP63", "DSG3", "EGFR",
    ],
    "Immune": [
        "PTPRC", "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "TRAC",
        "CD79A", "CD79B", "MS4A1", "MZB1", "JCHAIN", "IGHA1", "IGHG1", "IGHM",
        "NKG7", "GNLY", "KLRD1", "KLRB1",
        "CD68", "CD163", "C1QA", "C1QB", "C1QC", "CSF1R", "MRC1", "ITGAM", "ITGAX",
        "CSF3R", "FCGR3B", "CEACAM8", "CXCR2",
        "CLC", "SIGLEC8", "RNASE2", "EPX",
        "TPSAB1", "CPA3", "CD1C", "CLEC9A",
    ],
    "Fibroblast": [
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
        "DCN", "LUM", "PDGFRA", "PDGFRB", "FAP", "POSTN", "BGN", "CCDC80",
        "FBLN1", "FBLN2", "VCAN", "LOXL1", "ACTA2", "RGS5", "MCAM",
    ],
    "Endothelial": [
        "PECAM1", "VWF", "CLDN5", "CDH5", "ERG", "FLT1", "KDR", "TIE1", "TEK",
        "ENG", "NOS3", "MMRN1", "PLVAP", "ROBO4", "ESM1", "ACVRL1", "SOX17", "ACKR1",
    ],
}

CLASS_TO_SUPERCLASS = {
    "Cancer cell":        "Tumour_Epithelial",
    "Mitotic Figures":    "Tumour_Epithelial",
    "Epithelial":         "Tumour_Epithelial",
    "Lymphocytes":        "Immune",
    "Plasmocytes":        "Immune",
    "Eosinophils":        "Immune",
    "Neutrophils":        "Immune",
    "Macrophages":        "Immune",
    "Fibroblasts":        "Fibroblast",
    "Minor Stromal Cell": "Fibroblast",
    "Endothelial Cell":   "Endothelial",
}


def _write_nan(par, dataset_id, method_id, reason):
    print(f"WARNING: {reason}  Writing NaN.", flush=True)
    ad.AnnData(uns={
        "dataset_id":       dataset_id,
        "normalization_id": "counts",
        "method_id":        method_id,
        "metric_ids":       ["marker_specificity"],
        "metric_values":    [float("nan")],
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

# Cell IDs with a mapped GT class
mapped_obs = sol_obs[sol_obs["groundtruth_cell_type"].isin(CLASS_TO_SUPERCLASS)]
valid_ids   = set(mapped_obs["cell_id"].values)

if not valid_ids:
    _write_nan(par, dataset_id, method_id, "No GT cells with a mapped class.")
    raise SystemExit(0)

# Build per-GT-cell count matrix from solution transcripts
print(">> Building GT cell expression from solution transcripts", flush=True)
tx = sdata_sol["transcripts"][["cell_id", "feature_name"]].compute()
tx = tx[(tx["cell_id"] != 0) & (tx["cell_id"].isin(valid_ids))]

if tx.empty:
    _write_nan(par, dataset_id, method_id, "No transcripts for mapped GT cells.")
    raise SystemExit(0)

counts_df = tx.groupby(["cell_id", "feature_name"]).size().unstack(fill_value=0)
panel     = set(counts_df.columns)
col_idx   = {g: i for i, g in enumerate(counts_df.columns)}
counts_arr = counts_df.values

# Filter marker genes to those present in the panel
marker_panel = {}
for sc, genes in MARKERS.items():
    in_panel = [g for g in genes if g in panel]
    marker_panel[sc] = in_panel
    n_miss = len(genes) - len(in_panel)
    if n_miss:
        print(f"   {sc}: {n_miss}/{len(genes)} marker genes absent from panel", flush=True)

# Count how many groups have ≥1 expressed gene per cell
print(">> Evaluating marker groups", flush=True)
n_groups = np.zeros(len(counts_df), dtype=np.int8)
for sc, genes in marker_panel.items():
    if genes:
        idxs = [col_idx[g] for g in genes]
        n_groups += (counts_arr[:, idxs] > 0).any(axis=1).astype(np.int8)

n         = len(counts_df)
specific  = float((n_groups == 1).sum() / n)
ambiguous = float((n_groups >  1).sum() / n)
negative  = float((n_groups == 0).sum() / n)

print(
    f">> n={n:,}  Specific={specific:.3f}  "
    f"Ambiguous={ambiguous:.3f}  Negative={negative:.3f}",
    flush=True,
)

print(">> Writing output", flush=True)
ad.AnnData(uns={
    "dataset_id":       dataset_id,
    "normalization_id": "counts",
    "method_id":        method_id,
    "metric_ids":       ["marker_specificity"],
    "metric_values":    [specific],
}).write_h5ad(par["output"], compression="gzip")
