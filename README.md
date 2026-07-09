# IrrMapper

Annual, 30 m maps of irrigated agriculture across the western United States,
1985–present, produced with Random Forest classification of Landsat imagery in
Google Earth Engine.

Coverage: Arizona, California, Colorado, Idaho, Montana, Nevada, New Mexico,
Oregon, Utah, Washington, Wyoming.

## Data access

The published product is an Earth Engine image collection, one image per
state-year (e.g. `MT_2024`):

```python
import ee
ee.Initialize()

# boolean irrigated-only product
irr = ee.Image('projects/ee-dgketchum/assets/IrrMapper/version1_2/MT_2024')
```

Two product levels:

| Collection | Classes |
|---|---|
| `projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp` | 0 irrigated, 1 rainfed, 2 uncultivated, 3 wetland |
| `projects/ee-dgketchum/assets/IrrMapper/version1_2` | irrigated only (class 0, masked) |

Training data (labeled polygons: irrigated with year annotation, dryland,
fallow, uncultivated, wetland) are published at
[Zenodo, DOI 10.5281/zenodo.17980068](https://zenodo.org/records/17980068).

## How it works

1. Points are sampled from labeled polygons (`map/distribute_points.py`).
2. A multi-season Landsat feature stack — surface reflectance composites,
   vegetation indices, GRIDMET climate, terrain, and ancillary layers — is
   sampled at the points in Earth Engine (`map/call_ee.py`).
3. Per-state training tables are assembled and the most important ~50 features
   selected (`map/tables.py`, `map/models.py`); the feature lists used by every
   production run are archived in `provenance/variable_importance/`.
4. A Random Forest (150 trees) is trained per state and applied per year
   (`map/call_ee.py::export_classification`).
5. State-specific rules clean the raw classifications (frequency, NDVI, slope,
   center-pivot evidence) before export to the published collections
   (`map/call_ee.py::export_special`, `map/assets.py`).

Metadata manifests for every published asset are in `provenance/`.

## Repository layout

- `map/` — production pipeline (Earth Engine + scikit-learn)
- `training_redevelopment/` — experimental ML (MLP/U-Net, embedding-based
  classification) toward IrrMapper v2
- `provenance/` — asset metadata manifests and archived per-run feature lists

## Environment

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run python map/statewise.py
```

## Status

IrrMapper is under active development with NASA ROSES support (OSIRIS and
FM4Irr projects, 2026–2029): the pipeline is being refactored into a
config-driven, pip-installable `irrmapper` package with multi-sensor inputs
(Sentinel-1/2, VIIRS) and expanded coverage. The `v1.2-2025` tag preserves the
as-run December 2025 production state.

## Citation

Ketchum, D., Jencso, K., Maneta, M.P., Melton, F., Jones, M.O., Huntington, J.
(2020). IrrMapper: A Machine Learning Approach for High Resolution Mapping of
Irrigated Agriculture Across the Western U.S. *Remote Sensing* 12(14), 2328.
https://doi.org/10.3390/rs12142328

Ketchum, D., Hoylman, Z.H., Huntington, J., Brinkerhoff, D., Jencso, K.G.
(2023). Irrigation intensification impacts sustainability of streamflow in the
Western United States. *Communications Earth & Environment*.
https://doi.org/10.1038/s43247-023-01152-2

## License

Apache-2.0
