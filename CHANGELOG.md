# task_spatial_segmentation x.y.z

## BREAKING CHANGES

<!-- * Restructured `src` directory (PR #3). -->

## NEW FUNCTIONALITY

* Add three new benchmarking metrics: **ASR** (Assignment Specificity Ratio) for liver zonation, **Marker Specificity** for cell-type transcript assignment, and **Mitotic Specificity** for mitotic cell marker evaluation. All three are wired into the `run_benchmark` workflow.
* Extend `file_common_ist.yaml` and `file_spatial_solution.yaml` schemas with optional `pa_boundaries`/`cv_boundaries` for liver zonation landmarks.
* Propagate `groundtruth_cell_type` from `tenx_xenium_groundtruth` loader through `process_dataset` to `spatial_solution`.
* Preserve the vendor `cell_id` on transcripts in `spatial_unlabelled` as an optional segmentation prior, instead of stripping it together with the held-out ground-truth columns. Declared as an optional integer column in `src/api/file_spatial_unlabelled.yaml`.

## MAJOR CHANGES

* ...

## MINOR CHANGES

* ...

## BUGFIXES

* ...
