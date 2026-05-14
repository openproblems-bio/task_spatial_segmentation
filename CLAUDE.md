# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Framework: Viash + Nextflow

This is a [Viash](https://viash.io)-based benchmarking task for spatial transcriptomics cell segmentation, part of the [OpenProblems](https://openproblems.bio) initiative. Every executable is a **Viash component** — a self-contained unit pairing a `config.vsh.yaml` (metadata, arguments, Docker setup) with a `script.py` or `script.R`. Nextflow workflows orchestrate multi-component pipelines.

## Common Commands

```bash
# Sync test resources from S3 (required before testing)
scripts/sync_resources.sh

# Build all components (Docker-cached)
viash ns build --parallel --setup cachedbuild

# Build all Docker containers (run before local benchmark)
scripts/project/build_all_docker_containers.sh

# Test a single component
viash test src/methods/cellpose/config.vsh.yaml

# Test all components in parallel
viash ns test --parallel

# Run benchmark locally (test scale)
scripts/run_benchmark/run_test_local.sh

# Run full local benchmark
scripts/run_benchmark/run_full_local.sh

# Create a new method or metric from template
common/scripts/create_component

# Inspect the Docker image ID and Dockerfile for a built component
target/executable/<name>/<name> ---docker_image_id
target/executable/<name>/<name> ---dockerfile

# Run a single built component directly
target/executable/<name>/<name> --input <input> --output <output>

# Run a built component as a Nextflow workflow
nextflow run target/nextflow/<name> -profile docker --id <id> --input <input> --publish_dir out/
```

## Architecture

The task follows a fixed pipeline defined by the API specs in `src/api/`:

```
Raw spatial data
    → [data_processor] process_dataset    → unlabelled + solution files
    → [method / control_method]           → prediction file
    → [data_processor] process_prediction → formatted prediction
    → [metric] ari                        → score file
```

**Component types** (each defined in `src/api/comp_*.yaml`):
- `control_method` — baseline segmentations (random Voronoi, true labels, empty labels)
- `method` — segmentation algorithms under evaluation (currently: Cellpose)
- `data_processor` — preprocessing and postprocessing steps
- `metric` — evaluation metrics (currently: ARI)
- `workflow` — Nextflow pipelines in `src/workflows/`

**File format contracts** are defined in `src/api/file_*.yaml`. All component I/O must conform to these schemas. The `common/component_tests/run_and_check_output.py` helper enforces them during tests.

## Component Structure

Each component under `src/` follows this pattern:
```
src/<type>/<name>/
├── config.vsh.yaml   # arguments, Docker image, test resources, test defaults
└── script.py         # or script.R
```

Test resources and default parameters for `viash test` are declared inside `config.vsh.yaml` under `info.test_resources` and `info.test_default`.

## Data Formats

- **Spatial data**: Zarr-based `SpatialData` objects (`.zarr/`)
- **Single-cell data**: AnnData HDF5 files (`.h5ad`)
- Additional sources of ground truth will include spatial proteomics or annotated histology
- Test datasets live in `resources_test/` (downloaded via `scripts/sync_resources.sh` from S3)

## Important Notes

- The `common/` directory is a **git submodule** — clone with `git clone --recursive` or run `git submodule update --init` after cloning.
- `README.md` is **auto-generated** from the YAML API specs; do not edit it directly.
- All components run inside Docker containers by default. Use `--platform docker` / `--engine docker` flags with Viash when needed.
- `_viash.yaml` is the project-level Viash config (project name, organization, package registry, test resource S3 paths).
- Don't commit to main, always create a new branch
- Fill in the summary for a src/methods/<component>/config.vsh.yaml. Write a one sentence summary and a one paragraph summary of how this method works based on documentation and references.