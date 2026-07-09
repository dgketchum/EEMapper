"""Equivalence gate for the config-driven post-processing engine.

map.postproc.export_special driven by configs/irrmapper_v1_2.toml must build,
for every state at year 2022, exactly the expression strings the legacy
map.call_ee.export_special hardcodes — asserted against the same golden
fixture (tests/fixtures/export_special_expressions.json). Fully mocked, no
EE credentials needed.

Also covers the pure config machinery: TOML loading/validation, threshold
formulas across years, pivot-asset schedules, and the run manifest.
"""

import json
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest

import map.postproc
from map.config import ThresholdRule, load_config, resolved_manifest

YEAR = 2022
STATES = ["MT", "ID", "WA", "OR", "NM", "NV", "CO", "WY", "UT", "CA", "AZ"]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "configs", "irrmapper_v1_2.toml")
FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "export_special_expressions.json",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


def _capture_state(cfg, state, year=YEAR):
    """Run the config-driven export_special with everything mocked.

    Mirrors the capture in test_export_special_expressions.py so the two
    functions are compared on identical terms.
    """
    mock_ee = MagicMock(name="ee")
    mock_ee.Image.return_value.getInfo.return_value = {"properties": {}}

    mock_landsat = MagicMock(name="landsat_composites")
    mock_get_cdl = MagicMock(name="get_cdl")
    mock_get_cdl.return_value = (MagicMock(name="cdl0"), MagicMock(name="cdl1"))
    mock_copy = MagicMock(name="copy_asset")

    with mock.patch.multiple(
        "map.postproc",
        ee=mock_ee,
        landsat_composites=mock_landsat,
        get_cdl=mock_get_cdl,
        copy_asset=mock_copy,
    ):
        map.postproc.export_special(
            cfg, "in_coll", "out_coll", "roi", state, years=[year], start_tasks=False
        )

    expressions = []
    for call in mock_ee.mock_calls:
        if call[0].split(".")[-1] == "expression":
            expressions.append(call.args[0])

    return expressions, mock_copy.call_count


# --------------------------------------------------------------------------- #
# equivalence with the legacy hardcoded function
# --------------------------------------------------------------------------- #
def test_config_engine_matches_legacy_golden_fixture(cfg):
    with open(FIXTURE_PATH) as f:
        expected = json.load(f)

    captured = {}
    for state in STATES:
        expressions, copy_calls = _capture_state(cfg, state)
        captured["{}_{}".format(state, YEAR)] = expressions
        if state == "AZ":
            captured["AZ_copy_asset_called"] = copy_calls == 1

    assert captured == expected


def test_min_year_gating_matches_legacy_branches(cfg):
    # MT before 2008: pivot step skipped -> one expression only.
    exprs, _ = _capture_state(cfg, "MT", year=2007)
    assert len(exprs) == 1
    # ID at 2010: year > 2010 pivot step skipped -> two expressions.
    exprs, _ = _capture_state(cfg, "ID", year=2010)
    assert len(exprs) == 2
    # WA at 2011: year > 2011 pivot step skipped -> one expression.
    exprs, _ = _capture_state(cfg, "WA", year=2011)
    assert len(exprs) == 1
    assert "(SUM > 5)" in exprs[0]


# --------------------------------------------------------------------------- #
# threshold formulas
# --------------------------------------------------------------------------- #
def test_threshold_ramp_matches_source_formula():
    # threshold = 5 if year < 2016 else max(2025 - year - 1, 0)  (WA group)
    rule = ThresholdRule(type="ramp", before=5, switch_year=2016, horizon=2025)
    assert [rule.for_year(y) for y in (2015, 2016, 2020, 2024, 2030)] == [5, 8, 4, 0, 0]
    # CA anchors to 2021: max(2021 - year - 1, 0) after 2016
    ca = ThresholdRule(type="ramp", before=5, switch_year=2016, horizon=2021)
    assert [ca.for_year(y) for y in (2015, 2016, 2020, 2022)] == [5, 4, 0, 0]


def test_threshold_fixed(cfg):
    assert cfg.postproc["MT"].threshold.for_year(1990) == 6
    assert cfg.postproc["ID"].threshold.for_year(2022) == 5


# --------------------------------------------------------------------------- #
# pivot schedules
# --------------------------------------------------------------------------- #
def test_mt_pivot_schedule_matches_source(cfg):
    mt = cfg.postproc["MT"]
    root = "users/dgketchum/openet/pivots/mt_pivot_{}"
    assert mt.pivot_asset_for_year(2005) == root.format(2009)
    assert mt.pivot_asset_for_year(2010) == root.format(2011)
    assert mt.pivot_asset_for_year(2012) == root.format(2013)
    assert mt.pivot_asset_for_year(2014) == root.format(2015)
    assert mt.pivot_asset_for_year(2019) == root.format(2019)
    assert mt.pivot_asset_for_year(2025) == root.format(2019)


# --------------------------------------------------------------------------- #
# config loading / validation
# --------------------------------------------------------------------------- #
def test_canonical_config_loads(cfg):
    assert cfg.product.version == "1.2"
    assert cfg.product.public_collection == "UMT/Climate/IrrMapper_RF/v1_2"
    assert cfg.model.number_of_trees == 150
    assert cfg.model.bag_fraction == 0.5
    assert cfg.run.vintage_glob == "09MAY2023"
    assert set(cfg.run.states) <= set(cfg.postproc)
    assert cfg.postproc["AZ"].copy_only


def test_unknown_key_rejected(tmp_path, cfg):
    with open(CONFIG_PATH) as f:
        text = f.read()
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("scale = 30", "scale = 30\nbogus_key = 1"))
    with pytest.raises(ValueError, match="bogus_key"):
        load_config(str(bad))


def test_missing_section_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('[project]\nee_project = "x"\nuser_root = "y"\ngcs_bucket = "z"\n')
    with pytest.raises(ValueError, match="missing section"):
        load_config(str(bad))


def test_state_without_rules_rejected(tmp_path):
    with open(CONFIG_PATH) as f:
        text = f.read()
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("[postproc.AZ]\ncopy_only = true\n", ""))
    with pytest.raises(ValueError, match="AZ"):
        load_config(str(bad))


# --------------------------------------------------------------------------- #
# run manifest
# --------------------------------------------------------------------------- #
def test_resolved_manifest_content(cfg):
    m = resolved_manifest(cfg, CONFIG_PATH)
    assert m["config"]["model"]["number_of_trees"] == 150
    assert m["config_file"] == os.path.abspath(CONFIG_PATH)
    assert m["git_sha"] is None or len(m["git_sha"]) == 40
    assert m["created"].endswith("+00:00") or "T" in m["created"]
    # round-trips to JSON (what gets written next to run outputs)
    json.dumps(m)
