"""Unit tests for run-provenance (manifest stamping) helpers.

Covers the pure provenance machinery in irrmapper.config (manifest_digest,
asset_properties, write_manifest, resolve_path / repo_root, resolved_manifest)
and the extra_props threading added to irrmapper.postproc.exports.export_special
and irrmapper.models.rf_ee.export_classification. Everything is fully mocked; no Earth Engine
credentials or network access are required (CI runs pytest -m "not ee").
"""

import json
import os
import re
from unittest import mock
from unittest.mock import MagicMock

import pytest

from irrmapper.models import rf_ee
from irrmapper.postproc import exports
from irrmapper.config import (
    asset_properties,
    load_config,
    manifest_digest,
    repo_root,
    resolve_path,
    resolved_manifest,
    write_manifest,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "configs", "irrmapper_v1_2.toml")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


def _full_manifest(cfg):
    """A manifest whose provenance fields are all populated non-None strings.

    resolved_manifest leaves git_sha/earthengine_api_version None when git or
    ee are unavailable (as in CI); this fixture forces a complete manifest so
    the "full" expectations are deterministic regardless of environment.
    """
    m = resolved_manifest(cfg, CONFIG_PATH)
    m["git_sha"] = "0" * 40
    m["earthengine_api_version"] = "1.4.3"
    m["created"] = "2026-07-09T00:00:00+00:00"
    return m


# --------------------------------------------------------------------------- #
# manifest_digest
# --------------------------------------------------------------------------- #
def test_manifest_digest_ignores_timestamp_and_code(cfg):
    """Two manifests for the same config hash equal even when created/git_sha
    differ: the digest is over config values only."""
    m1 = resolved_manifest(cfg, CONFIG_PATH)
    m2 = resolved_manifest(cfg, CONFIG_PATH)
    m2["created"] = "1999-01-01T00:00:00+00:00"
    m2["git_sha"] = "deadbeef"
    assert manifest_digest(m1) == manifest_digest(m2)


def test_manifest_digest_changes_with_config(cfg):
    """Mutating a config value (run.years) changes the digest."""
    m1 = resolved_manifest(cfg, CONFIG_PATH)
    m2 = resolved_manifest(cfg, CONFIG_PATH)
    m2["config"]["run"]["years"] = [1999]
    assert manifest_digest(m1) != manifest_digest(m2)


# --------------------------------------------------------------------------- #
# asset_properties
# --------------------------------------------------------------------------- #
def test_asset_properties_full_manifest(cfg):
    """A fully-populated manifest yields exactly the six provenance keys, all
    string-valued, with product_version and run_config_sha256 as expected."""
    m = _full_manifest(cfg)
    props = asset_properties(m)
    assert set(props) == {
        "product_name",
        "product_version",
        "run_config_sha256",
        "code_git_sha",
        "earthengine_api_version",
        "run_created",
    }
    assert all(isinstance(v, str) for v in props.values())
    assert props["product_version"] == "1.2"
    assert props["run_config_sha256"] == manifest_digest(m)


def test_asset_properties_drops_none(cfg):
    """None-valued provenance fields (e.g. git_sha) are omitted, not stamped."""
    m = _full_manifest(cfg)
    m["git_sha"] = None
    props = asset_properties(m)
    assert "code_git_sha" not in props
    assert "product_name" in props


# --------------------------------------------------------------------------- #
# write_manifest
# --------------------------------------------------------------------------- #
def test_write_manifest_creates_dir_and_roundtrips(cfg, tmp_path):
    """write_manifest makes missing dirs, names the file
    manifest_<name>_<version>_<stamp>.json, and JSON round-trips the dict."""
    m = _full_manifest(cfg)
    out_dir = tmp_path / "nested" / "runs"
    path = write_manifest(m, str(out_dir))

    assert os.path.isdir(str(out_dir))
    fname = os.path.basename(path)
    assert re.fullmatch(r"manifest_IrrMapper_1\.2_\d{8}T\d{6}\.json", fname)

    with open(path) as fp:
        loaded = json.load(fp)
    assert loaded == m


# --------------------------------------------------------------------------- #
# resolve_path / repo_root
# --------------------------------------------------------------------------- #
def test_resolve_path_absolute_unchanged():
    """An absolute path is returned verbatim."""
    abs_in = "/nas/irrmapper/thing"
    assert resolve_path(abs_in) == abs_in


def test_resolve_path_relative_anchored_to_repo_root():
    """A relative path is joined onto the repo root."""
    rel = "provenance/runs"
    assert resolve_path(rel) == os.path.join(repo_root(), rel)


def test_repo_root_locates_repo():
    """repo_root points at the repository (has irrmapper/ and the canonical config)."""
    root = repo_root()
    assert os.path.isdir(os.path.join(root, "irrmapper"))
    assert os.path.isfile(os.path.join(root, "configs", "irrmapper_v1_2.toml"))


# --------------------------------------------------------------------------- #
# extra_props threading: irrmapper.postproc.exports.export_special
# --------------------------------------------------------------------------- #
def _run_postproc_export(cfg, extra_props):
    """Run export_special (MT, 2022) fully mocked; return the props dict.

    Mirrors the _capture_state mocking in test_postproc_config_equivalence.py.
    Because ee.Image(...).getInfo()['properties'] returns the SAME dict object
    every call, the returned dict is the one the function mutates in place
    before target.set(props); asserting on it verifies what gets stamped.
    """
    props = {}
    mock_ee = MagicMock(name="ee")
    mock_ee.Image.return_value.getInfo.return_value = {"properties": props}

    mock_landsat = MagicMock(name="landsat_composites")
    mock_get_cdl = MagicMock(name="get_cdl")
    mock_get_cdl.return_value = (MagicMock(name="cdl0"), MagicMock(name="cdl1"))
    mock_copy = MagicMock(name="copy_asset")

    with mock.patch.multiple(
        "irrmapper.postproc.exports",
        ee=mock_ee,
        landsat_composites=mock_landsat,
        get_cdl=mock_get_cdl,
        copy_asset=mock_copy,
    ):
        exports.export_special(
            cfg,
            "in_coll",
            "out_coll",
            "roi",
            "MT",
            years=[2022],
            start_tasks=False,
            extra_props=extra_props,
        )
    return props


def test_export_special_threads_extra_props(cfg):
    """extra_props are merged into the stamped properties alongside
    post_process."""
    props = _run_postproc_export(
        cfg, {"run_config_sha256": "abc", "product_version": "1.2"}
    )
    assert "post_process" in props
    assert props["run_config_sha256"] == "abc"
    assert props["product_version"] == "1.2"


def test_export_special_omits_stamp_without_extra_props(cfg):
    """Without extra_props the properties carry post_process but no stamp."""
    props = _run_postproc_export(cfg, None)
    assert "post_process" in props
    assert "run_config_sha256" not in props
    assert "product_version" not in props


# --------------------------------------------------------------------------- #
# extra_props threading: irrmapper.models.rf_ee.export_classification
# --------------------------------------------------------------------------- #
def _set_call_args(*mocks):
    """Collect the positional args of every ``.set(...)`` call recorded across
    the given mocks."""
    found = []
    for m in mocks:
        for call in m.mock_calls:
            if call[0].split(".")[-1] == "set":
                found.append(call.args)
    return found


def _run_export_classification(extra_props):
    """Run export_classification (MT, 2020) with ee and stack_bands mocked.

    input_props=['a'] takes the ee.List branch, so ee.List(...).getInfo() must
    be iterable (->[]) and bandNames().getInfo() must be iterable (->[]) so the
    band-check comprehension is empty and no re-training path runs.
    """
    mock_ee = MagicMock(name="ee")
    mock_ee.List.return_value.getInfo.return_value = []
    mock_stack = MagicMock(name="stack_bands")
    mock_stack.return_value.bandNames.return_value.getInfo.return_value = []

    kwargs = dict(
        out_name="MT",
        table="t",
        asset_root="root",
        region="r",
        years=[2020],
        input_props=["a"],
    )
    if extra_props is not None:
        kwargs["extra_props"] = extra_props

    with mock.patch.multiple("irrmapper.models.rf_ee", ee=mock_ee, stack_bands=mock_stack):
        rf_ee.export_classification(**kwargs)
    return mock_ee, mock_stack


def test_export_classification_threads_extra_props():
    """extra_props reaches the exported image as classified_img.set(extra_props)."""
    mock_ee, mock_stack = _run_export_classification({"run_config_sha256": "abc"})
    assert ({"run_config_sha256": "abc"},) in _set_call_args(mock_ee, mock_stack)


def test_export_classification_omits_extra_props():
    """Without extra_props no such .set(extra_props) call is made."""
    mock_ee, mock_stack = _run_export_classification(None)
    assert ({"run_config_sha256": "abc"},) not in _set_call_args(mock_ee, mock_stack)
