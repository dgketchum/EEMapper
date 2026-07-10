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
| 5 | Training-table build | `tables.concatenate_band_extract` | ⚠️ hardcoded local paths; the fallow(4)→irrigated(1) chained-assignment remaps were rewritten with `.loc` and the pandas pin lifted (pandas 3 supported, 2026-07-09) |
| 6 | Table upload | → `users/dgketchum/bands/state/{ST}_{vintage}` | ⚠️ the production `{ST}_09MAY2023` tables (231,382 points) are archived with SHA-256 checksums (`provenance/archive_training_tables.py` → `/nas` + `provenance/training_table_archive_09MAY2023.json`, 2026-07-09); Zenodo staging pending; earlier NOV2021 vintages still EE-only |
| 7 | Feature selection | `models.find_rf_variable_importance` → `variables_{ST}_{glob}.json` | ✅ archived in `provenance/variable_importance/`; `map/runner.py` reads them from there |
| 8 | Classification | `call_ee.export_classification` via `map/runner.py classify` | ✅ config-driven from `configs/irrmapper_v1_2.toml`; dry-run default; provenance-stamped (2026-07-09) |
| 9 | Post-processing | `map/postproc.py::export_special` via `map/runner.py postprocess` | ✅ config-driven; graph parity with legacy verified live (see §3); `SUM` still depends on collection state at run time |
| 10 | Environment | `pyproject.toml` + `uv.lock` | ⚠️ exist but untracked (commit in Phase 0) |

Legend: ✅ reproducible now · ⚠️ scripted but requires undocumented hand-edits ·
❌ artifact not archived anywhere durable.

## 2. Actions

### A. Archive derived artifacts as-is (short-term, no refactor needed)

1. Export the production EE training tables to GCS and archive them — either a
   new versioned Zenodo record or a `data/` release attached to a GitHub
   release. **09MAY2023 done Jul 2026** (`provenance/archive_training_tables.py`:
   gs://wudr/training_table_archive/ → `/nas/irrmapper/training_table_archive/`,
   manifest with SHA-256 + row counts committed). The `{ST}_*NOV2021` vintages
   and the Zenodo record itself remain.
2. ~~Commit the `variables_{ST}_{glob}.json` feature lists to the repo~~
   **Done** (`provenance/variable_importance/`).
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

**Done 2026-07-09** for the classification/post-processing stages:
`map/runner.py` drives extract, classify, postprocess, and rasters from the
TOML (dry-run by default; `--execute` starts tasks). Every executed run writes
a fully-resolved manifest to `provenance/runs/` and stamps
`product_version`, `run_config_sha256`, `code_git_sha`,
`earthengine_api_version`, and `run_created` onto each exported asset
(`map/config.py::asset_properties`).

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

See §3 for the pieces in place.

### E. Benchmark-era data versioning (forward-looking)

FM4Irr D1.1 introduces semver'd benchmark data (major = schema/label change,
minor = new grids/states, patch = QC fixes) with git-tagged changelogs, and models
must record the benchmark version they trained on. Adopt that scheme here from the
first post-refactor run; retro-label the 2021/2023 vintages as best-effort
(e.g. `legacy-2021.11`, `legacy-2023.05`).

## 3. Equivalence gates (refactor regression protection)

Three gates guard "the refactor didn't change the product":

1. **Gate 1 — golden fixtures (CI, no EE)**: serialized `stack_bands` graph
   hashes and the `export_special` expression strings for all 11 states are
   committed under `tests/fixtures/`; `pytest -m "not ee"` re-derives both
   from current code and diffs against the fixtures on every push.
2. **Gate 2 — live graph parity + determinism baseline**:
   - `tests/test_gate2_graph_parity.py` (marked `ee`) captures the image each
     implementation hands to `Export.image.toAsset` and compares full
     serialized-graph SHA-256, legacy `call_ee.export_special` vs config-driven
     `map/postproc.py`, per state at 2022. **Passed for all 11 states
     2026-07-09** — identical graphs guarantee identical pixels, no export
     needed. AZ's copy-only path verified identical.
   - `tests/fixtures/gate2_determinism.py` submits the production MT
     classification twice over a small Greenfields Bench ROI (2020), exports
     both to GCS, and pixel-diffs the rasters. **Result (2026-07-09): 161 of
     1,237,400 pixels differ (0.013%)** — scattered single pixels flipping
     between all class pairs, i.e. decision-boundary pixels under EE's
     distributed execution (row-order/accumulation jitter in RF training and
     compositing), not tile artifacts. Consequence: bit-identity is NOT
     achievable for any comparison that re-trains the classifier; pixel-level
     parity checks must use a tolerance at or above this ~1.3e-4 noise floor,
     and serialized-graph identity (the parity test above) is the exact
     equivalence tool because it is execution-independent. Artifacts in
     `/nas/irrmapper/gate2_determinism/`.
3. **Gate 3 — full-state parity** before the Dec 2026 production run: one
   complete state-year via the runner diffed against the legacy path.

## 4. Zenodo record notes

The current record is the CommsEnv paper dataset (v3, Dec 2025), tied to that
publication. Plan: once benchmark work starts (Aug 2026), create a dedicated
"IrrMapper training data" record with its own DOI lineage and semver-aligned
versions, and reference the CommsEnv record as the ancestor. Keep polygons
(labels), derived training tables (band extracts), and feature lists in the same
versioned release so stages 1–7 of the chain are pinned by one DOI per vintage.
