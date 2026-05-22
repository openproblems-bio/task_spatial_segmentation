from __future__ import annotations

import math

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from rasterio.features import shapes
from shapely.affinity import scale, translate
from shapely.geometry import shape

IMAGES_KEY = None

TABLES_KEY = "table"
TABLES_CELL_ID_KEY = "cell_id"
TABLES_AREA_KEY = None
TABLES_CENTROID_X_KEY = None
TABLES_CENTROID_Y_KEY = None

POINTS_KEY = "transcripts"
POINTS_CELL_ID_KEY = "cell_id"
POINTS_BACKGROUND_ID = 0
POINTS_X_KEY = "x"
POINTS_Y_KEY = "y"
POINTS_Z_KEY = "z"
POINTS_GENE_KEY = "feature_name"

SHAPES_KEY = "cell_boundaries"
SHAPES_CELL_ID_KEY = "cell_id"

NUCLEUS_SHAPES_KEY = "nucleus_boundaries"
NUCLEUS_SHAPES_CELL_ID_KEY = "nucleus_ID"

REF_CELL_TYPE_KEY = "cell_type"


# -------------------------------------------------------------------------
# TEMPORARY OPEN PROBLEMS ADAPTER
#
# The functions in this block convert the current Open Problems
# prediction/solution files into the SpatialData layout used by the SegTraQ
# metric runners. This is intentionally local to the benchmark for now.
#
# Longer term, this should move into a separate Viash module that prepares a
# metrics-ready SpatialData object from segmentation outputs. The SegTraQ metric
# components should then only read that prepared object and run the metrics.
# -------------------------------------------------------------------------
def label_to_array(label_element):
    if isinstance(label_element, xr.DataTree):
        return label_element["scale0"].image.to_numpy()
    return label_element.to_numpy()


def labels_to_shapes(
    label_img: np.ndarray,
    cell_id_key: str = "cell_id",
    simplify_tolerance: float | None = 0.5,
    translation_yx: tuple[float, float] | None = None,
    scale_xy: tuple[float, float] | None = None,
) -> gpd.GeoDataFrame:
    if label_img.ndim != 2:
        raise ValueError("Input label_img must be 2D.")

    lab = np.asarray(label_img).astype(np.int32, copy=False)
    mask = lab != 0

    geoms = []
    ids = []
    for geom_mapping, value in shapes(lab, mask=mask, connectivity=8):
        ids.append(int(value))
        geoms.append(shape(geom_mapping))

    if len(ids) == 0:
        gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")
        gdf.index.name = cell_id_key
        return gdf

    gdf = gpd.GeoDataFrame(
        {cell_id_key: ids, "geometry": geoms},
        geometry="geometry",
    ).set_index(cell_id_key)

    gdf = gdf.dissolve(by=cell_id_key, as_index=True)
    gdf.index.name = cell_id_key

    x_scale, y_scale = (1.0, 1.0) if scale_xy is None else scale_xy

    # Convert label-image pixel coordinates into the local coordinate system
    # expected by the existing cell_boundaries transform.
    if scale_xy is not None:
        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: scale(
                geom,
                xfact=1 / x_scale,
                yfact=1 / y_scale,
                origin=(0, 0),
            )
        )

    if translation_yx is not None:
        y_translation, x_translation = translation_yx
        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: translate(
                geom,
                xoff=x_translation / x_scale,
                yoff=y_translation / y_scale,
            )
        )

    if simplify_tolerance and simplify_tolerance > 0 and not gdf.empty:
        gdf["geometry"] = gdf.geometry.simplify(
            simplify_tolerance,
            preserve_topology=True,
        )

    return gdf


def _extract_translation_yx(sdata_solution: sd.SpatialData, labels_key: str) -> tuple[float, float]:
    label_transform = sdata_solution[labels_key]["scale0"].image.attrs["transform"]["global"]
    if hasattr(label_transform, "transformations"):
        for transform in label_transform.transformations:
            if hasattr(transform, "translation"):
                return tuple(float(value) for value in transform.translation)
    if hasattr(label_transform, "translation"):
        return tuple(float(value) for value in label_transform.translation)
    raise ValueError(f"Could not extract a y/x translation from labels key {labels_key!r}.")


def _extract_scale_xy(sdata_solution: sd.SpatialData, shapes_key: str) -> tuple[float, float]:
    shape_transform = sdata_solution.shapes[shapes_key].attrs["transform"]["global"]
    if hasattr(shape_transform, "scale"):
        return tuple(float(value) for value in shape_transform.scale)
    if hasattr(shape_transform, "transformations"):
        for transform in shape_transform.transformations:
            if hasattr(transform, "scale"):
                return tuple(float(value) for value in transform.scale)
    raise ValueError(f"Could not extract an x/y scale from shapes key {shapes_key!r}.")


def prepare_sdata_for_segtraq(
    sdata_solution: sd.SpatialData,
    sdata_prediction: sd.SpatialData,
    tables_key: str = TABLES_KEY,
    tables_cell_id_key: str = TABLES_CELL_ID_KEY,
    points_key: str = POINTS_KEY,
    points_cell_id_key: str = POINTS_CELL_ID_KEY,
    points_x_key: str = POINTS_X_KEY,
    points_y_key: str = POINTS_Y_KEY,
    points_z_key: str = POINTS_Z_KEY,
    shapes_key: str = SHAPES_KEY,
    labels_key: str = "cell_labels",
    nucleus_labels_key: str = "nucleus_labels",
    shapes_cell_id_key: str = SHAPES_CELL_ID_KEY,
    nucleus_shapes_key: str = NUCLEUS_SHAPES_KEY,
    nucleus_shapes_cell_id_key: str = NUCLEUS_SHAPES_CELL_ID_KEY,
    segmentation_key: str = "segmentation",
) -> sd.SpatialData:
    # Create predicted cell boundary shapes from the predicted segmentation label image.
    translation_yx = _extract_translation_yx(sdata_solution, labels_key)
    scale_xy = _extract_scale_xy(sdata_solution, shapes_key)

    cell_labels = label_to_array(sdata_prediction[segmentation_key])
    cell_shapes_gdf = labels_to_shapes(
        cell_labels,
        translation_yx=translation_yx,
        scale_xy=scale_xy,
        simplify_tolerance=0.5,
    )

    sdata_prediction.shapes[shapes_key] = sd.models.ShapesModel.parse(cell_shapes_gdf)
    sdata_prediction.shapes[shapes_key].attrs["transform"] = sdata_solution.shapes[shapes_key].attrs["transform"]

    # Copy nucleus labels from the solution, as nuclei are not predicted, and convert them to shapes.
    nucleus_labels = label_to_array(sdata_solution[nucleus_labels_key])
    nucleus_shapes_gdf = labels_to_shapes(
        nucleus_labels,
        translation_yx=translation_yx,
        scale_xy=scale_xy,
        simplify_tolerance=0.5,
    )
    nucleus_shapes_gdf.index.name = nucleus_shapes_cell_id_key
    sdata_prediction.shapes[nucleus_shapes_key] = sd.models.ShapesModel.parse(nucleus_shapes_gdf)
    sdata_prediction.shapes[nucleus_shapes_key].attrs["transform"] = (
        sdata_solution.shapes[nucleus_shapes_key].attrs["transform"]
    )

    # Copy transcripts from the solution and replace ground-truth cell IDs with predicted cell IDs.
    transcripts = sdata_solution[points_key].compute().copy()
    transcripts[f"{points_cell_id_key}_source"] = transcripts[points_cell_id_key]
    transcripts = transcripts.drop(columns=[points_cell_id_key])

    tx_gdf = gpd.GeoDataFrame(
        transcripts,
        geometry=gpd.points_from_xy(transcripts[points_x_key], transcripts[points_y_key]),
    )

    cells_gdf = sdata_prediction.shapes[shapes_key].reset_index()[[shapes_cell_id_key, "geometry"]].copy()

    joined = gpd.sjoin(
        tx_gdf,
        cells_gdf,
        how="left",
        predicate="within",
    )

    transcripts_out = joined.drop(columns=["geometry", "index_right"])

    # Keep unassigned transcripts as 0 so SegTraQ can compute unassigned transcript rates.
    transcripts_out[points_cell_id_key] = transcripts_out[shapes_cell_id_key].fillna(0).astype(int)

    transcripts_ddf = dd.from_pandas(
        pd.DataFrame(transcripts_out),
        npartitions=1,
    )

    coordinates = {
        "x": points_x_key,
        "y": points_y_key,
    }
    if points_z_key in transcripts_out.columns:
        coordinates["z"] = points_z_key

    sdata_prediction[points_key] = sd.models.PointsModel.parse(
        transcripts_ddf,
        coordinates=coordinates,
    )
    sdata_prediction[points_key].attrs["transform"] = sdata_solution[points_key].attrs["transform"]

    sdata_prediction.tables[tables_key].obs["region"] = shapes_key

    # Cell IDs across layers have to use the same data type.
    sdata_prediction.tables[tables_key].obs[tables_cell_id_key] = (
        sdata_prediction.tables[tables_key].obs[tables_cell_id_key].astype(int)
    )
    sdata_prediction.tables[tables_key].obs.index.name = None

    sdata_prediction.tables[tables_key].obs["region"] = (
        sdata_prediction.tables[tables_key].obs["region"].astype("category")
    )
    sdata_prediction.set_table_annotates_spatialelement(tables_key, region=shapes_key)

    return sdata_prediction


def load_prepared_sdata(input_prediction: str, input_solution: str):
    sdata_prediction = sd.read_zarr(input_prediction)
    sdata_solution = sd.read_zarr(input_solution)
    sdata_segtraq = prepare_sdata_for_segtraq(
        sdata_solution=sdata_solution,
        sdata_prediction=sdata_prediction,
    )
    return sdata_solution, sdata_prediction, sdata_segtraq


# -------------------------------------------------------------------------
# END TEMPORARY OPEN PROBLEMS ADAPTER
# -------------------------------------------------------------------------


def initialize_segtraq(sdata_segtraq: sd.SpatialData):
    import segtraq

    st = segtraq.SegTraQ(
        sdata_segtraq,
        images_key=IMAGES_KEY,
        tables_key=TABLES_KEY,
        tables_cell_id_key=TABLES_CELL_ID_KEY,
        tables_area_key=TABLES_AREA_KEY,
        tables_centroid_x_key=TABLES_CENTROID_X_KEY,
        tables_centroid_y_key=TABLES_CENTROID_Y_KEY,
        points_key=POINTS_KEY,
        points_cell_id_key=POINTS_CELL_ID_KEY,
        points_background_id=POINTS_BACKGROUND_ID,
        points_x_key=POINTS_X_KEY,
        points_y_key=POINTS_Y_KEY,
        points_z_key=POINTS_Z_KEY,
        points_gene_key=POINTS_GENE_KEY,
        shapes_key=SHAPES_KEY,
        shapes_cell_id_key=SHAPES_CELL_ID_KEY,
        nucleus_shapes_key=NUCLEUS_SHAPES_KEY,
        nucleus_shapes_cell_id_key=NUCLEUS_SHAPES_CELL_ID_KEY,
    )
    #st.filter_control_and_low_quality_transcripts() #qv not in transcripts, so delete for now
    return st


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def nanmedian(values) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    return safe_float(np.nanmedian(numeric))


def nanmean(values) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    return safe_float(np.nanmean(numeric))


def finite_matrix_mean(df: pd.DataFrame) -> float:
    values = df.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return safe_float(np.mean(values))


def median_obs_metrics(table: ad.AnnData, columns_to_metric_ids: dict[str, str]) -> dict[str, float]:
    metrics = {}
    for column, metric_id in columns_to_metric_ids.items():
        metrics[metric_id] = nanmedian(table.obs[column]) if column in table.obs else float("nan")
    return metrics


def write_metric_output(
    output_path: str,
    sdata_solution: sd.SpatialData,
    sdata_prediction: sd.SpatialData,
    metrics: dict[str, float],
    normalization_id: str = "counts",
) -> None:
    output = ad.AnnData(
        uns={
            "dataset_id": sdata_solution.tables[TABLES_KEY].uns["dataset_id"],
            "normalization_id": normalization_id,
            "method_id": sdata_prediction.tables[TABLES_KEY].uns["method_id"],
            "metric_ids": list(metrics.keys()),
            "metric_values": [safe_float(value) for value in metrics.values()],
        }
    )
    output.write_h5ad(output_path, compression="gzip")


def add_script_dir_to_path(script_file: str, globals_dict: dict) -> None:
    """Allow metric scripts to import this resource both from source and from Viash staging."""
    import sys
    from pathlib import Path

    source_common_dir = Path(script_file).resolve().parents[1] / "segtraq_common"
    resources_dir = Path(globals_dict.get("meta", {}).get("resources_dir", ""))

    for path in (source_common_dir, resources_dir):
        if path.exists():
            sys.path.insert(0, str(path))


def prepare_clustering_table(adata: ad.AnnData) -> ad.AnnData:
    import scanpy as sc

    if "normalized_log" in adata.layers:
        adata.X = adata.layers["normalized_log"].copy()
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

    elif "counts" in adata.layers:
        sc.pp.normalize_total(adata, layer="counts")
        sc.pp.log1p(adata, layer="counts")

        adata.X = adata.layers["counts"].copy()

        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

    else:
        raise ValueError(
            "Neither 'normalized_log' nor 'counts' found in adata.layers."
        )

    return adata


def prepare_reference_adata(
    input_scrnaseq_reference: str
) -> ad.AnnData:

    adata_ref = ad.read_h5ad(input_scrnaseq_reference).copy()

    if "feature_name" in adata_ref.var.columns:
        adata_ref.var_names = adata_ref.var["feature_name"].astype(str).values
        adata_ref = adata_ref[:, ~adata_ref.var_names.duplicated()].copy()

    if "normalized_log" in adata_ref.layers:
        adata_ref.X = adata_ref.layers["normalized_log"].copy()

    adata_ref.var_names_make_unique()
    return adata_ref


def run_label_transfer_and_markers(
    st,
    input_scrnaseq_reference: str,
    ref_cell_type: str = "cell_type",
) -> dict[str, dict[str, list[str]]]:
    import segtraq

    adata_ref = prepare_reference_adata(input_scrnaseq_reference)

    segtraq.run_label_transfer(
        sdata=st.sdata,
        adata_ref=adata_ref,
        ref_cell_type=REF_CELL_TYPE_KEY,
        tables_key=TABLES_KEY,
        tables_cell_id_key=TABLES_CELL_ID_KEY,
        points_key=POINTS_KEY,
        points_cell_id_key=POINTS_CELL_ID_KEY,
        points_gene_key=POINTS_GENE_KEY,
        ref_ensemble_key=None,
        query_ensemble_key=None,
        use_hvg=False,
        inplace=True,
    )

    return segtraq.markers_from_reference(
        adata_ref,
        cell_type_key=REF_CELL_TYPE_KEY,
        n_jobs=1,
    )


def n_components_from_cell_types(table: ad.AnnData) -> int | None:
    for column in ("transferred_cell_type", REF_CELL_TYPE_KEY):
        if column in table.obs:
            n_celltypes = table.obs[column].nunique(dropna=True)
            if n_celltypes > 0:
                return int(n_celltypes)
    return None


def run_ovrlpy(
    sdata: sd.SpatialData, 
    n_comp: int, 
    points_cell_id_key: str = POINTS_CELL_ID_KEY,
    points_gene_key: str = POINTS_GENE_KEY, 
    points_x_key: str = POINTS_X_KEY,
    points_y_key: str = POINTS_Y_KEY,
    points_z_key: str = POINTS_Z_KEY,
    n_workers: int = 1
):
    import ovrlpy
    print(f">> ovrlpy version: {ovrlpy.__version__}", flush=True)

    coordinate_df = sdata.points[POINTS_KEY].rename(columns={points_gene_key: "gene"})
    coordinate_df = coordinate_df.loc[:, ["gene", points_x_key, points_y_key, points_z_key, points_cell_id_key]].compute()
    coordinate_df[points_z_key] = coordinate_df[points_z_key] - coordinate_df[points_z_key].min()

    ovrlpy_sdata = ovrlpy.Ovrlp( #this part triggers a segmentation fault
        coordinate_df,
        n_components=n_comp,
        n_workers=n_workers,
    )

    ovrlpy_sdata.analyse()
    return ovrlpy_sdata.integrity_map

def all_nan_metrics(metric_ids: list[str]) -> dict[str, float]:
    return {metric_id: float("nan") for metric_id in metric_ids}


def is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)
