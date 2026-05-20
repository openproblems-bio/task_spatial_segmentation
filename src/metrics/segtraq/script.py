import anndata as ad
import segtraq
import shapely
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.affinity import translate, scale

## VIASH START
par = {
    "input_prediction": "resources_test/task_spatial_segmentation/mouse_brain_combined/processed_prediction.zarr",
    "input_solution": "resources_test/task_spatial_segmentation/mouse_brain_combined/spatial_solution.zarr",
    "output": "output.h5ad",
}
meta = {
    "name": "segtraq",
}
## VIASH END

# ------------------------------------------------------------------------
# Temporary adapter - start
#
# This converts the current Open Problems prediction/solution objects into
# the SpatialData layout expected by SegTraQ metrics.
# Longer term, this should live in a separate Viash data processor that
# produces metrics-ready SpatialData objects from segmentation outputs.
# ------------------------------------------------------------------------
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

    # Convert label-image pixel coordinates into the local coordinate system
    # expected by the existing cell_boundaries transform.
    if scale_xy is not None:
        x_scale, y_scale = scale_xy
        y_translation, x_translation = translation_yx
        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: scale(
                geom,
                xfact=1/scale_xy[0],
                yfact=1/scale_xy[1],
                origin=(0, 0)
            )
        )

    if translation_yx is not None:
        gdf["geometry"] = gdf.geometry.apply(
            lambda geom: translate(
                geom,
                xoff=x_translation/ x_scale,
                yoff=y_translation/ y_scale,
            )
        )

    if simplify_tolerance and simplify_tolerance > 0 and not gdf.empty:
        gdf["geometry"] = shapely.simplify(
            gdf.geometry.values,
            simplify_tolerance,
            preserve_topology=True,
        )

    return gdf

def prepare_sdata_for_segtraq(
    sdata_solution: sd.SpatialData,
    sdata_prediction: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str = "z",
    shapes_key: str = "cell_boundaries",
    labels_key: str = "cell_labels",
    nucleus_labels_key: str = "nucleus_labels",
    shapes_cell_id_key: str = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    nucleus_shapes_cell_id_key: str = "nucleus_ID",
    segmentation_key: str = "segmentation",
):
    # Create predicted cell boundary shapes from the predicted segmentation label image.
    label_transform = sdata_solution[labels_key]["scale0"].image.attrs["transform"]["global"]
    shape_transform = sdata_solution.shapes[shapes_key].attrs["transform"]["global"]

    translation_yx = tuple(label_transform.transformations[0].translation)
    scale_xy = tuple(shape_transform.scale)

    cell_labels = label_to_array(sdata_prediction[segmentation_key])
    cell_shapes_gdf = labels_to_shapes(cell_labels, translation_yx=translation_yx, scale_xy=scale_xy, simplify_tolerance=0.5)

    sdata_prediction.shapes[shapes_key] = sd.models.ShapesModel.parse(cell_shapes_gdf)

    # Reuse the same transform as the solution cell boundaries.
    # This assumes the predicted and solution boundary coordinates use the same native frame.
    sdata_prediction.shapes[shapes_key].attrs["transform"] = (
        sdata_solution.shapes[shapes_key].attrs["transform"]
    )

    # Copy nucleus labels from solution, as these are not predicted, and convert to shapes
    nucleus_labels = label_to_array(sdata_solution[nucleus_labels_key])
    nucleus_shapes_gdf = labels_to_shapes(nucleus_labels, translation_yx=translation_yx, scale_xy=scale_xy, simplify_tolerance=0.5)
    nucleus_shapes_gdf.index.name = nucleus_shapes_cell_id_key
    sdata_prediction.shapes[nucleus_shapes_key] = sd.models.ShapesModel.parse(nucleus_shapes_gdf)
    sdata_prediction.shapes[nucleus_shapes_key].attrs["transform"] = (
        sdata_solution.shapes[nucleus_shapes_key].attrs["transform"]
    )

    # Copy transcripts from solution and replace ground-truth cell IDs with predicted cell IDs.
    transcripts = sdata_solution[points_key].compute().copy()
    transcripts[f"{points_cell_id_key}_source"] = transcripts[points_cell_id_key]
    transcripts = transcripts.drop(columns=[points_cell_id_key])

    tx_gdf = gpd.GeoDataFrame(
        transcripts,
        geometry=gpd.points_from_xy(transcripts[points_x_key], transcripts[points_y_key]),
    )

    # prepare shapes for geopandas spatial join
    cells_gdf = sdata_prediction.shapes[shapes_key].reset_index()[
        [shapes_cell_id_key, "geometry"]
    ].copy()

    joined = gpd.sjoin(
        tx_gdf,
        cells_gdf,
        how="left",
        predicate="within",
    )

    transcripts_out = joined.drop(columns=["geometry", "index_right"])

    # Keep unassigned transcripts as 0 so SegTraQ can compute % unassigned transcripts.
    transcripts_out[points_cell_id_key] = (
        transcripts_out[shapes_cell_id_key]
        .fillna(0)
        .astype(int)
    )

    transcripts_ddf = dd.from_pandas(
        pd.DataFrame(transcripts_out),
        npartitions=1,
    )

    coordinates = {
        "x": points_x_key,
        "y": points_y_key,
        "z": points_z_key
    }

    sdata_prediction[points_key] = sd.models.PointsModel.parse(
        transcripts_ddf,
        coordinates=coordinates,
    )

    sdata_prediction[points_key].attrs["transform"] = (
        sdata_solution[points_key].attrs["transform"]
    )

    sdata_prediction.tables[tables_key].obs["region"] = shapes_key

    # cell ids across layers have to be of the same data type
    sdata_prediction.tables["table"].obs[tables_cell_id_key] = sdata_prediction.tables["table"].obs[tables_cell_id_key].astype(int)
    sdata_prediction.tables["table"].obs.index.name = None

    sdata_prediction.tables["table"].obs["region"] = sdata_prediction.tables["table"].obs["region"].astype("category")
    sdata_prediction.set_table_annotates_spatialelement("table", region=shapes_key)

    return sdata_prediction

# ------------------------------------------------------------------------
# Temporary adapter - end
# ------------------------------------------------------------------------

def get_scalar_metrics(sdata_segtraq):
    table = sdata_segtraq.tables["table"]

    metrics = {
        "segtraq_num_cells": table.uns["num_cells"],
        "segtraq_num_transcripts": table.uns["num_transcripts"],
        "segtraq_num_genes": table.uns["num_genes"],
        "segtraq_perc_unassigned_transcripts": table.uns["perc_unassigned_transcripts"],
    }

    optional_obs_metrics = {
        "transcript_count": "segtraq_median_transcripts_per_cell",
        "gene_count": "segtraq_median_genes_per_cell",
        "cell_area": "segtraq_median_cell_area",
        "transcript_density": "segtraq_median_transcript_density",
    }

    for column, metric_id in optional_obs_metrics.items():
        if column in table.obs:
            metrics[metric_id] = table.obs[column].median()

    return {key: float(value) for key, value in metrics.items()}

print(">> Reading input files", flush=True)
sdata_prediction = sd.read_zarr(par["input_prediction"])
sdata_solution = sd.read_zarr(par["input_solution"])

print(">> Preparing SpatialData object for SegTraQ", flush=True)
sdata_segtraq = prepare_sdata_for_segtraq(
    sdata_solution=sdata_solution,
    sdata_prediction=sdata_prediction,
)

print(">> Running SegTraQ baseline", flush=True)
st = segtraq.SegTraQ(
    sdata_segtraq,
    images_key=None,
    tables_centroid_x_key=None,
    tables_centroid_y_key=None,
    tables_area_key=None,
    nucleus_shapes_cell_id_key="nucleus_ID",
    points_background_id=0,
)

st.run_baseline(inplace=True)

print(">> Collecting scalar metric output", flush=True)
metrics = get_scalar_metrics(st.sdata)

dataset_id = sdata_solution.tables["table"].uns["dataset_id"]
method_id = sdata_prediction.tables["table"].uns["method_id"]

output = ad.AnnData(
    uns={
        "dataset_id": dataset_id,
        "normalization_id": "counts",
        "method_id": method_id,
        "metric_ids": list(metrics.keys()),
        "metric_values": list(metrics.values()),
    }
)

print(">> Writing output", flush=True)
output.write_h5ad(par["output"], compression="gzip")