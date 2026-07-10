"""Gate 2 graph parity: legacy vs config-driven post-processing, live EE.

The Gate 1 golden fixtures compare expression strings; these tests compare
the ENTIRE serialized expression graph of the image each implementation
hands to ee.batch.Export.image.toAsset, for every state at 2022 (the year
hardcoded in the legacy function). Identical serialization means Earth
Engine is guaranteed to compute identical pixels, so no export or pixel
diff is needed to establish old-vs-new parity for the post-processing
stage.

Requires EE credentials (marked ee; auto-skipped otherwise) and reads live
asset metadata from users/dgketchum/IrrMapper/IrrMapper_sw, but starts no
tasks: the export constructor is patched out.
"""

import hashlib
import os
from unittest import mock

import ee
import pytest

from map import call_ee
from map import postproc
from map.config import load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "irrmapper_v1_2.toml",
)

YEAR = 2022  # the year hardcoded in the legacy function's loop

EXPORT_STATES = [
    "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY",
]  # AZ is copy-only and produces no export task


def _sha(serialized):
    return hashlib.sha256(serialized.encode()).hexdigest()


def _exported_image(mock_to_asset):
    assert mock_to_asset.call_count == 1
    call = mock_to_asset.call_args
    image = call.args[0] if call.args else call.kwargs["image"]
    kwargs = {k: v for k, v in call.kwargs.items() if k != "image"}
    return image, kwargs


def _capture_legacy(cfg, state):
    roi = os.path.join(cfg.paths.boundaries, state)
    with mock.patch.object(ee.batch.Export.image, "toAsset") as to_asset:
        call_ee.export_special(
            cfg.paths.raw_collection, cfg.paths.comp_collection, roi, state
        )
    return _exported_image(to_asset)


def _capture_config(cfg, state):
    roi = os.path.join(cfg.paths.boundaries, state)
    with mock.patch.object(ee.batch.Export.image, "toAsset") as to_asset:
        postproc.export_special(
            cfg,
            cfg.paths.raw_collection,
            cfg.paths.comp_collection,
            roi,
            state,
            [YEAR],
            start_tasks=False,
        )
    return _exported_image(to_asset)


@pytest.mark.ee
@pytest.mark.parametrize("state", EXPORT_STATES)
def test_postproc_graph_parity(ee_initialized, state):
    """The config engine builds a byte-identical EE graph to the legacy code."""
    cfg = load_config(CONFIG_PATH)
    legacy_img, legacy_kwargs = _capture_legacy(cfg, state)
    config_img, config_kwargs = _capture_config(cfg, state)

    assert legacy_kwargs == config_kwargs
    assert _sha(legacy_img.serialize()) == _sha(config_img.serialize())


@pytest.mark.ee
def test_postproc_copy_only_parity(ee_initialized):
    """AZ: both implementations copy the same source asset to the same id."""
    cfg = load_config(CONFIG_PATH)
    roi = os.path.join(cfg.paths.boundaries, "AZ")

    with mock.patch("map.call_ee.copy_asset") as legacy_copy:
        call_ee.export_special(
            cfg.paths.raw_collection, cfg.paths.comp_collection, roi, "AZ"
        )
    with mock.patch("map.postproc.copy_asset") as config_copy:
        postproc.export_special(
            cfg,
            cfg.paths.raw_collection,
            cfg.paths.comp_collection,
            roi,
            "AZ",
            [YEAR],
            start_tasks=False,
        )

    assert legacy_copy.call_args == config_copy.call_args
