"""Unit tests for the config-driven runner (map/runner.py).

The runner replaces the hand-edited orchestration in map/statewise.py: every
per-state parameter now comes from configs/irrmapper_v1_2.toml, the default is
a dry run that touches no Earth Engine surface, and every executed export is
stamped with resolved-run provenance. These tests lock the plan conventions to
what statewise.py produced and prove the dry-run/execute contract, using the
real canonical TOML and no Earth Engine credentials.
"""

import json
import os
from unittest.mock import MagicMock

import pytest

from map import runner
from map.config import load_config
from map.runner import (
    feature_list,
    plan_classify,
    plan_extract,
    plan_postprocess,
    plan_rasters,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "configs", "irrmapper_v1_2.toml")


@pytest.fixture
def cfg():
    """A fresh load of the canonical production config for each test."""
    return load_config(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# classify plan parity with statewise.py
# --------------------------------------------------------------------------- #
def test_classify_plan_matches_statewise_conventions(cfg):
    """Locks plan_classify to the export_classification() call statewise.py made:
    same out_name/table/asset_root/region/years/bag_fraction for MT."""
    mt = plan_classify(cfg, states=["MT"])[0]
    assert mt["out_name"] == "MT"
    assert mt["table"] == "users/dgketchum/bands/state/MT_09MAY2023"
    assert mt["asset_root"] == "users/dgketchum/IrrMapper/IrrMapper_sw"
    assert mt["region"] == "users/dgketchum/boundaries/MT"
    assert mt["years"] == [2025]
    assert mt["bag_fraction"] == 0.5
    assert mt["southern"] is False


def test_classify_southern_windows_match_production(cfg):
    """statewise.py ran southern composite windows for AZ and CA only; every
    other state (including NM) classified with southern=False."""
    southern = {
        s: plan_classify(cfg, states=[s])[0]["southern"]
        for s in ("AZ", "CA", "NM", "MT")
    }
    assert southern == {"AZ": True, "CA": True, "NM": False, "MT": False}


def test_classify_input_props_from_variable_importance(cfg):
    """input_props is the archived 50-feature list ([f[0] for f in ...]) that the
    state's RF was trained on, resolved against the repo root."""
    jpath = os.path.join(
        REPO, "provenance", "variable_importance", "variables_MT_09MAY2023.json"
    )
    with open(jpath) as fp:
        expected = [f[0] for f in json.load(fp)["MT"]]
    assert len(expected) == 50
    mt = plan_classify(cfg, states=["MT"])[0]
    assert mt["input_props"] == expected
    # feature_list() reaches the same archive by resolving the relative path
    assert feature_list(cfg, "MT") == expected


# --------------------------------------------------------------------------- #
# postprocess plan
# --------------------------------------------------------------------------- #
def test_postprocess_plan_uses_raw_and_comp_collections(cfg):
    """postproc reads the raw RF collection and writes the composited one."""
    p = plan_postprocess(cfg, states=["MT"])[0]
    assert p["input_coll"] == cfg.paths.raw_collection
    assert p["out_coll"] == cfg.paths.comp_collection
    assert p["roi"] == "users/dgketchum/boundaries/MT"
    assert p["state"] == "MT"
    assert p["years"] == [2025]


def test_postprocess_years_override(cfg):
    """An explicit years list overrides run.years."""
    p = plan_postprocess(cfg, states=["MT"], years=[2001, 2002])[0]
    assert p["years"] == [2001, 2002]


# --------------------------------------------------------------------------- #
# extract and rasters plans
# --------------------------------------------------------------------------- #
def test_extract_plan_fields(cfg):
    """extract point-samples the feature stack: file/points naming from the vintage
    globs, a 1e5 buffer, and a bounds filter, per request_band_extract()."""
    p = plan_extract(
        cfg, years=[2020], points_glob="7NOV2021", out_glob="20DEC2023", states=["MT"]
    )[0]
    assert p["file_prefix"] == "bands_MT_20DEC2023"
    assert p["points_layer"] == "users/dgketchum/points/state/points_MT_7NOV2021"
    assert p["region"] == "users/dgketchum/boundaries/MT"
    assert p["years"] == [2020]
    assert p["buffer"] == 1e5
    assert p["filter_bounds"] is True
    assert p["southern"] is False


def test_rasters_plan_fields(cfg):
    """rasters export from the composited collection with the default 3-year
    minimum-irrigated mask."""
    p = plan_rasters(cfg, years=[2025], states=["MT"])[0]
    assert p["irr_coll"] == cfg.paths.comp_collection
    assert p["roi"] == "users/dgketchum/boundaries/MT"
    assert p["years"] == [2025]
    assert p["min_years"] == 3
    assert p["state"] == "MT"
    assert p["export_freq"] is True


# --------------------------------------------------------------------------- #
# dry run touches no Earth Engine surface
# --------------------------------------------------------------------------- #
def test_dry_run_makes_no_ee_calls(cfg, monkeypatch):
    """The default dry run only builds and prints plans; it must never reach into
    call_ee or postproc (no attribute call at all)."""
    mock_call_ee = MagicMock(name="call_ee")
    mock_postproc = MagicMock(name="postproc")
    monkeypatch.setattr("map.runner.call_ee", mock_call_ee)
    monkeypatch.setattr("map.runner.postproc", mock_postproc)

    runner.run_classify(cfg, states=["MT"], dry_run=True)
    runner.run_postprocess(cfg, states=["MT"], dry_run=True)
    runner.run_extract(
        cfg,
        years=[2020],
        points_glob="7NOV2021",
        out_glob="20DEC2023",
        states=["MT"],
        dry_run=True,
    )
    runner.run_rasters(cfg, years=[2025], states=["MT"], dry_run=True)

    assert mock_call_ee.mock_calls == []
    assert mock_postproc.mock_calls == []


# --------------------------------------------------------------------------- #
# execute path stamps provenance
# --------------------------------------------------------------------------- #
def test_run_classify_execute_stamps_provenance(cfg, monkeypatch, tmp_path):
    """Executing classify authorizes, writes a manifest, then calls
    export_classification once with the plan kwargs plus a provenance
    extra_props dict."""
    mock_call_ee = MagicMock(name="call_ee")
    mock_call_ee.is_authorized.return_value = True
    monkeypatch.setattr("map.runner.call_ee", mock_call_ee)
    monkeypatch.setattr(
        "map.runner.write_manifest",
        lambda manifest, out_dir: str(tmp_path / "manifest.json"),
    )

    runner.run_classify(cfg, states=["MT"], dry_run=False, config_path=CONFIG_PATH)

    assert mock_call_ee.export_classification.call_count == 1
    kwargs = dict(mock_call_ee.export_classification.call_args.kwargs)
    extra = kwargs.pop("extra_props")
    expected = plan_classify(cfg, states=["MT"])[0]
    expected.pop("stage")
    assert kwargs == expected
    for key in ("product_name", "product_version", "run_config_sha256", "run_created"):
        assert key in extra


def test_run_postprocess_execute_stamps_provenance(cfg, monkeypatch, tmp_path):
    """Executing postprocess passes cfg positionally to export_special and stamps
    a run_config_sha256 into extra_props."""
    mock_call_ee = MagicMock(name="call_ee")
    mock_call_ee.is_authorized.return_value = True
    monkeypatch.setattr("map.runner.call_ee", mock_call_ee)
    mock_postproc = MagicMock(name="postproc")
    monkeypatch.setattr("map.runner.postproc", mock_postproc)
    monkeypatch.setattr(
        "map.runner.write_manifest",
        lambda manifest, out_dir: str(tmp_path / "manifest.json"),
    )

    runner.run_postprocess(cfg, states=["MT"], dry_run=False, config_path=CONFIG_PATH)

    assert mock_postproc.export_special.call_count == 1
    args, kwargs = mock_postproc.export_special.call_args
    assert args[0] is cfg
    assert "run_config_sha256" in kwargs["extra_props"]


# --------------------------------------------------------------------------- #
# validation and CLI
# --------------------------------------------------------------------------- #
def test_unknown_state_rejected(cfg):
    """A state outside run.states is a config error, not a silent skip."""
    with pytest.raises(ValueError, match="TX"):
        plan_classify(cfg, states=["TX"])


def test_cli_classify_dry_run(monkeypatch, capsys):
    """main() defaults to a dry run and reports it on stdout."""
    monkeypatch.chdir(REPO)
    runner.main(["configs/irrmapper_v1_2.toml", "classify", "--states", "MT"])
    out = capsys.readouterr().out
    assert "dry run" in out


def test_cli_extract_requires_globs(monkeypatch):
    """extract without --years/--points-glob/--out-glob exits via parser.error."""
    monkeypatch.chdir(REPO)
    with pytest.raises(SystemExit):
        runner.main(["configs/irrmapper_v1_2.toml", "extract"])
