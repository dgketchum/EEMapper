# Reproducibility Plan: Zenodo → IrrMapper

Goal: a third party (or David on a new machine in 2030) can go from the published
training polygons (Zenodo 10.5281/zenodo.17980068) to a classified state-year map
and get the published answer. Companion doc: `provenance.md` (what was actually
done). Phases 2/4 of the internal refactor plan build the machinery referenced here.

---

## 1. The chain today, stage by stage

| # | Stage | Code | Status |
|---|---|---|---|
| 1 | Labels | Zenodo record v3 (public) | ✅ published, but no manifest maps which shapefiles fed which points run |
| 2 | Point sampling | `distribute_points.PointsRunspec` | ⚠️ scripted; sampling parameters per vintage not recorded (pandas-2.x compat fixed 2026-07-09) |
| 3 | Point upload to EE | `statewise.push_points_to_asset` | ⚠️ private assets; hand-run |
| 4 | Band extract at points | `call_ee.request_band_extract` | ⚠️ hand-edited years/glob; CSVs in private GCS `wudr` |
| 5 | Training-table build | `tables.concatenate_band_extract` | ⚠️ hardcoded local paths; pandas pinned `>=2,<3` (2026-07-09) because the fallow(4)→irrigated(1) remap at tables.py:166/219/240 is a chained assignment that pandas 3.0 CoW silently no-ops — rewrite with `.loc` in Phase 2, then lift the pin |
| 6 | Table upload | → `users/dgketchum/bands/state/{ST}_{vintage}` | ❌ private; the NOV2021 tables ARE the training data and exist nowhere else |
| 7 | Feature selection | `models.find_rf_variable_importance` → `variables_{ST}_{glob}.json` | ❌ local-disk only, unarchived |
| 8 | Classification | `call_ee.export_classification` | ⚠️ hand-edited years/glob per run |
| 9 | Post-processing | `call_ee.export_special` | ⚠️ hand-edited year; `SUM` depends on collection state at run time |
| 10 | Environment | `pyproject.toml` + `uv.lock` | ⚠️ exist but untracked (commit in Phase 0) |

Legend: ✅ reproducible now · ⚠️ scripted but requires undocumented hand-edits ·
❌ artifact not archived anywhere durable.

## 2. Actions

### A. Archive derived artifacts as-is (short-term, no refactor needed)

1. Export the production EE training tables (`{ST}_*NOV2021`, `{ST}_09MAY2023`
   extracts) to GCS and archive them — either a new versioned Zenodo record or a
   `data/` release attached to a GitHub release.
2. Commit the `variables_{ST}_{glob}.json` feature lists to the repo.
3. ~~Harvest asset-metadata manifest~~ **Done Jul 2026**
   (`provenance/harvest_asset_metadata.py` + per-collection CSVs). Result: the
   `post_process` property was never stamped (discarded-`set()` bug, since fixed),
   so the Dec 2025 post-processing parameters are inferred from committed code,
   not recovered; the harvest did confirm the 2025 run's training tables
   (`{ST}_09MAY2023`) and full collection coverage.
4. Record the `PointsRunspec` parameters used for the 2021 vintage, to the extent
   recoverable from git history and memory, in `provenance.md`.

### B. Script the chain (medium-term = refactor Phases 2 & 4)

Config-driven pipeline where one TOML + one command reproduces each stage, and
every run emits `provenance.json`: code SHA, resolved config hash, training-table
asset + vintage, feature list, upstream EE dataset IDs (NED/NLCD/GRIDMET versions).
Stamp the same into exported asset metadata.

### C. Public access

- Make `IrrMapperComp` and `version1_2` world-readable if they aren't already
  (`assets.change_permissions` exists; currently used to set private).
- README documents: catalog asset IDs, GCS raster layout, Zenodo DOI, and the
  class key.

### D. Verification harness

A small fixed ROI (one HUC8 or county) + one year, reclassified end-to-end and
diffed against the published raster. This is simultaneously:
- the "did the refactor change outputs?" regression gate (refactor Constraint #2),
- the reproducibility demonstration for OSIRIS reporting.

### E. Benchmark-era data versioning (forward-looking)

FM4Irr D1.1 introduces semver'd benchmark data (major = schema/label change,
minor = new grids/states, patch = QC fixes) with git-tagged changelogs, and models
must record the benchmark version they trained on. Adopt that scheme here from the
first post-refactor run; retro-label the 2021/2023 vintages as best-effort
(e.g. `legacy-2021.11`, `legacy-2023.05`).

## 3. Zenodo record notes

The current record is the CommsEnv paper dataset (v3, Dec 2025), tied to that
publication. Plan: once benchmark work starts (Aug 2026), create a dedicated
"IrrMapper training data" record with its own DOI lineage and semver-aligned
versions, and reference the CommsEnv record as the ancestor. Keep polygons
(labels), derived training tables (band extracts), and feature lists in the same
versioned release so stages 1–7 of the chain are pinned by one DOI per vintage.
