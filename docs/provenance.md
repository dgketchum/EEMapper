# IrrMapper Product Provenance

Authored July 2026 from code archaeology: git history, the uncommitted working tree,
EE asset paths, and `postproc_2025.txt`. Covers the product as published through the
2025 vintage. Companion doc: `reproducibility.md`.

---

## 1. Product identity

- **IrrMapper**: annual, 30 m irrigation classification, 11 western states
  (AZ, CA, CO, ID, MT, NM, NV, OR, UT, WA, WY), 1985–2025 (CA from 1986;
  confirmed by the Jul 2026 asset harvest).
- **Methods paper**: Ketchum et al. 2020, *Remote Sensing* 12(14):2328
  (https://www.mdpi.com/2072-4292/12/14/2328).
- **Application paper**: Ketchum et al. 2023, *Communications Earth & Environment*
  (https://www.nature.com/articles/s43247-023-01152-2).
- **Public product**: EE public data catalog `UMT/Climate/IrrMapper_RF/v1_2`
  (irrigated = 1, others masked; CC-BY-4.0; provider University of Montana /
  Montana Climate Office). Fed from the private working collection
  `projects/ee-dgketchum/assets/IrrMapper/version1_2` (boolean, irrigated-only).
- **Version lineage** (from `../earthengine-catalog` git history and a
  2026-07-09 metadata harvest of the public collections):
  - **v1.0** = the actual first release: the 2020 *Remote Sensing* paper
    product, built on Landsat Collection 1. Catalog entry
    (`UMT/Climate/IrrMapper_RF/v1_0`) hidden 2023-07-31 — commit message:
    "hide ... while the source assets are missing." The v1.0 assets are gone;
    reproducibility genuinely lost, accepted (it was the least accurate).
  - **v1.1** = 2023-08-11 catalog rename of the v1_0 entry: the original
    training data re-run on Landsat Collection 2. Public collection
    `UMT/Climate/IrrMapper_RF/v1_1` still serves 363 images (11 states ×
    1986–2018) but is marked deprecated. Its assets carry only `system:*`
    properties — no provenance stamps to recover.
  - **v1.2** = current (catalog entry added Jan 2024): greatly expanded
    training data, per-state RF models, validation/uncertainty analysis in
    the 2023 Comms Earth & Environ supplement. Public collection holds 450
    images (1985–2025), in sync with the private `version1_2` collection
    through the Dec 2025 run. Everything in this document describes v1.2.
  - **v2.0** = planned under OSIRIS (multi-sensor, new models, 17 states).
- **Training labels**: Zenodo DOI 10.5281/zenodo.17980068, v3 (updated 17 Dec 2025).
  Polygon shapefiles in 5 classes (irrigated with year annotation, dryland, fallow,
  uncultivated, wetland), EPSG:102008, ~910 MB.

## 2. Asset lineage

Three-stage production flow, plus side products:

| Stage | Collection | Producer | Content |
|---|---|---|---|
| 1. Raw RF output | `users/dgketchum/IrrMapper/IrrMapper_sw` | `call_ee.export_classification` | per state-year, 4-class (0 irr, 1 rainfed, 2 uncultivated, 3 wetland) |
| 2. Post-processed | `projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp` | `call_ee.export_special` | per-state rule cleanup; AZ copied unmodified |
| 3. Catalog boolean | `projects/ee-dgketchum/assets/IrrMapper/version1_2` | `assets.convert_to_boolean_and_export` | mask to class==0 (irrigated) |

Side products: `IrrMapper_RF2` (min-years ≥ 3 binary mask via `assets.mask_move`;
**no longer present on EE** per the Jul 2026 harvest); per-state GeoTIFF and
irrigation-frequency rasters exported to the GCS `wudr` bucket via
`call_ee.export_raster` (EPSG:5071, `min_years=3`).

## 3. Training pipeline (as run)

Chain: Zenodo polygons → `distribute_points.PointsRunspec` sampling → point
shapefiles with `POINT_TYPE` (0 irrigated, 1 dryland, 2 uncultivated, 3 wetland,
4 fallow) and `YEAR` → uploaded to EE → `call_ee.request_band_extract` samples the
feature stack at points per year → CSVs to GCS `wudr` →
`tables.concatenate_band_extract` → training table → EE table asset
`users/dgketchum/bands/state/{ST}_{vintage}`.

Production vintages:
- **Training tables**: `TRAINING_DATA` dict (`statewise.py:32`) — all dated
  Nov 2021 (e.g. `MT_15NOV2021`). Note: tables exist for **17 states**, including
  KS, ND, NE, OK, SD, TX (`{ST}_7NOV2021`) — a head start on the OSIRIS 17-state
  expansion.
- **Band-extract glob**: `_glob = '09MAY2023'` for the 2025 run (`statewise.py:145`,
  uncommitted; was `05MAY2023` for the 2024 run).
- **Feature selection**: `models.find_rf_variable_importance` writes
  `variables_{ST}_{glob}.json` (top ~50 of ~150 candidate features per state).
  These JSONs live only on local disk — not in the repo, not archived (gap).

## 4. Feature stack (`call_ee.stack_bands`, call_ee.py:756)

- **Imagery**: Landsat 4/5/7/8/9 Collection 2 Level-2 SR
  (`LANDSAT/L*/C02/T1_L2`), QA_PIXEL bit-mask cloud/shadow filtering
  (adapted from cgmorton/OpenET; `ee_utils.py:33`).
- **Seasonal windows**: winter, spring, late spring, summer, fall, plus prior-year
  and two-years-prior growing seasons; alternate "southern" window set for AZ/CA.
- **Per window**: mean B2–B7, B10; max/mean NDVI, NDWI, EVI, GI; plus summed
  "integrated" indices over mid-season periods.
- **Climate**: GRIDMET temperature, precip + ETo sums, climatic water deficit over
  spring and water-year windows; WorldClim precip/temp anomalies (`ee_utils.py:14`).
- **Terrain**: USGS/NED elevation, slope, aspect; TPI at 150/250/1250 m.
- **Position/ancillary**: pixel lon/lat; NLCD 2011; CDL cultivated masks
  (`cdl.py`); JRC Global Surface Water occurrence.
- Band naming uses a positional `_1`/`_2` de-duplication loop (call_ee.py:877) —
  the origin of the odd names in `map/__init__.py` `FEATURE_NAMES`.

## 5. Classifier

`ee.Classifier.smileRandomForest(numberOfTrees=150, minLeafPopulation=1,
bagFraction=0.5)`, CLASSIFICATION mode (call_ee.py:505–508). Trained per state on
`POINT_TYPE` using that state's top-N importance features. Classified per
state-year, exported at 30 m with mode pyramiding.

Metadata stamped on every raw asset (call_ee.py:529–537): `training_data` (table
path), `bag_fraction`, `class_key`, `date_ingested`, `image_name`,
`system:time_start/end`.

## 6. Post-processing rules (`call_ee.export_special`, call_ee.py:271–485)

Inputs per state-year: the classification; an all-years irrigation-frequency `SUM`
(remap [0,1,2,3]→[1,0,0,0], summed over `IrrMapper_sw` **at run time**);
growing-season max NDVI; NED slope; CDL cropland; center-pivot polygons
(`users/dgketchum/openet/western_17_pivots`; MT uses vintage-specific
`mt_pivot_{2009,2011,2013,2015,2019}`).

Core expression pattern:
`(IRR==1 && NDVI>0.75 && SUM>t) → irrigated; (IRR==0 && SUM<t) → rainfed;
(IRR==0 && SLOPE>s) → uncultivated`, plus a pivot rescue rule
`(IRR!=0 && NDVI>0.68 && PIVOT==1 [&& SUM>t]) → irrigated` in later years.

Verified per-state parameters:

| State(s) | SUM threshold t | Slope s | Pivot rule | Notes |
|---|---|---|---|---|
| MT | 6 | 10 | year ≥ 2008, no SUM term | MT-specific pivot vintages |
| ID | 5 (dynamic calc at :347 dead-coded, overwritten at :350) | 6 | year > 2010 | extra de-irrigation rule: IRR==0 && NDVI<0.68 && SUM>t → rainfed |
| WA, OR, NM, NV | 5 if year<2016 else max(2025−year−1, 0) | 10 | year > 2011 | run-year constant **2025** |
| CO, WY, UT | 5 if year<2016 else max(2025−year−1, 0) | 4 | none | |
| CA | 5 if year<2016 else max(2021−year−1, 0) | 10 | year > 2011 | run-year constant **2021** |
| AZ | — | — | — | no rule; copied unmodified |

The run-year constants (2021/2023/2025) are fingerprints of when each state's rule
was last hand-edited. For year==2025 under the 2025 formula, t clamps to 0.

**Intended-but-broken provenance feature**: the code intends to stamp the exact
expression into each `IrrMapperComp` asset as a `post_process` property
(call_ee.py:471), but the original line `target.set(props)` discarded the return
value (EE images are immutable), so **no asset ever received it** — confirmed by
the July 2026 metadata harvest (`provenance/*.csv`), which found `post_process` on
zero of 1,354 assets. Fixed July 2026 (`target = target.set(props)`); effective
from the next post-processing run. What assets do carry: the
`export_classification` properties (`training_data`, `bag_fraction`, `class_key`,
`date_ingested`, `image_name`), and on `version1_2` a `postprocessing_date`
(observed values: 2023-12-28/29, 2024-11-05, 2025-12-19).

**Caveat**: `SUM` depends on the contents of `IrrMapper_sw` at post-processing
time, so post-processing is state-of-collection dependent, not purely a function
of (code, year).

## 7. Production timeline (from git + logs)

| Era | Evidence |
|---|---|
| v1.0, Landsat C1 (2020) | `93c4387` "version 1 create binary classification"; RS 2020 paper |
| version 2 era (2021) | `ab16627`/`b36b6bb` "running version 2 on western states"; `ca2f1ca` eastern-state training data; Nov 2021 training tables |
| PNAS/CommsEnv freeze (2022–23) | `1f37e54` "freeze for pnas submission"; `pnas_submission` branch |
| C1→C2 migration (2022 vintage) | `f316925` "run IrrMapper 2022... on collection 2... sent to IrrMapper_sw, then post-processed and sent to IrrMapperComp" |
| 2023 run on C2 | `6a1fe63`, `803a7ea`; catalog export `1268fd6` (Dec 2023) |
| 2024 run | `fa2d1b7` (Nov 2024); boolean raster `ef589b6` |
| UT 2022 re-run | `1ef0cd5` (Jul 2025, current HEAD) |
| **2025 run (Dec 2025)** | **uncommitted**: working-tree edits (`_glob='09MAY2023'`, classify `range(2025, 2026)`, parameterized `export_raster`, conda-path fixes) + `postproc_2025.txt` showing `IrrMapperComp/{ST}_2025` for 10 states, AZ copied |
| AlphaEarth/ML experiments | `alphaearth` branch, last commit 18 Dec 2025 (same day as the 2025 post-proc): `training_redevelopment/` package, MLP/U-Net, MGRS extracts, "reached 94%" |

Note: `export_special` in the tree still reads `range(2022, 2023)` (call_ee.py:276)
— the year edit for the Dec 2025 post-proc run was made and then reverted or lost,
and the intended `post_process` metadata stamp never worked (see §6). The 2025
expressions are therefore **inferred, not recorded**: assuming only the year range
was edited for the run, the committed rules imply, for year 2025, t=0 for
WA/OR/NM/NV/CO/WY/UT (2025−2025−1 clamped) and CA (2021−2025−1 clamped), t=6 for
MT, t=5 for ID, AZ copied. The 2025 run is confirmed by harvest to have used the
`{ST}_09MAY2023` training tables uniformly across all 11 states and all three
collections (`date_ingested` 2025-12-18, `postprocessing_date` 2025-12-19).

## 8. Provenance gaps and risks

1. ~~The 2025 production state is uncommitted~~ **Resolved Jul 2026**: committed
   as-run (`a74140c`, `d562bb6`, `eb69010`) and tagged `v1.2-2025`.
2. ~~No git tags~~ **Resolved Jul 2026**: `v1.2-2025` pushed; earlier vintages
   still untagged (optional retro-tags).
3. **Hand-edited run-year constants** baked into post-processing rules.
4. **`variables_{ST}_{glob}.json` and derived training tables are unarchived** —
   the actual model inputs exist only on local disk and in private EE assets.
5. **Zenodo → EE training-table derivation is not scripted end-to-end**
   (see `reproducibility.md`).
6. **Deprecated upstream EE assets**: `USGS/NED` (deprecation warning captured in
   `postproc_2025.txt`), `USGS/NLCD/NLCD2011`, `JRC/GSW1_0`. Note: the
   `alphaearth` branch had already swapped `USGS/NED`→`USGS/3DEP/10m` and
   `JRC/GSW1_0`→`JRC/GSW1_4`; those hunks were deliberately reverted in the
   Jul 2026 merge (`df91f37`) to preserve as-run production behavior. Adopt them
   as a config-gated change with a Gate 2 pixel-equivalence check (expect real
   differences — these are input data changes, not refactors).
7. **No code SHA or post-processing record in asset metadata** — the intended
   `post_process` stamp never landed due to the discarded-`set()` bug (§6, fixed
   Jul 2026); only `export_classification` properties and `postprocessing_date`
   exist on assets.
8. **Frequency `SUM` layer is unversioned** (§6 caveat).
9. **`IrrMapper_sw` is missing CO 2005 and CO 2007** (39 years vs 41 in
   `IrrMapperComp`/`version1_2`) — the raw inputs behind those two composite
   years are unaccounted for.
10. **`IrrMapper_RF2` no longer exists** on EE despite `assets.mask_move`
    targeting it.

**Metadata harvest completed Jul 2026** (`provenance/harvest_asset_metadata.py`):
`IrrMapper_sw` 454 assets (11 states 1985–2025, CA from 1986, plus single-year
2017 KS/ND/NE/OK/SD/TX extensions on `7NOV2021` tables); `IrrMapperComp` 450;
`version1_2` 450. Manifests: `provenance/{collection}_metadata.csv`.
