"""Golden-graph regression test for irrmapper.features.stack.stack_bands.

Rebuilds the feature-stack expression graph for each fixture case and
compares band names and the SHA-256 of the serialized graph against the
fixtures captured by tests/fixtures/capture_stack_bands.py. Any refactor
of stack_bands (or its helpers in irrmapper.ingest) that alters the
computation graph fails here.

Requires authenticated Earth Engine credentials (two metadata getInfo
calls per case); skipped automatically when none are present. A hash
mismatch with unchanged code can also mean the earthengine-api version
changed serialization — check earthengine_api_version in the fixture,
and if band names still match, re-capture fixtures on the new version.
On failure the fresh graph is written next to the fixture as
stack_bands_{case}.serialized.fresh.json for diffing.
"""

import glob
import hashlib
import json
import os

import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

CASES = sorted(
    os.path.basename(f)[len("stack_bands_") : -len(".json")]
    for f in glob.glob(os.path.join(FIXTURE_DIR, "stack_bands_*.json"))
    if not f.endswith(".serialized.json")
)


@pytest.mark.ee
@pytest.mark.parametrize("case", CASES)
def test_stack_bands_graph_matches_fixture(ee_initialized, case):
    import ee

    from irrmapper.features.stack import stack_bands

    with open(os.path.join(FIXTURE_DIR, "stack_bands_{}.json".format(case))) as fp:
        fixture = json.load(fp)

    roi = ee.FeatureCollection(fixture["roi"])
    stack = stack_bands(fixture["year"], roi, southern=fixture["southern"])

    band_names = stack.bandNames().getInfo()
    assert band_names == fixture["band_names"], (
        "stack_bands({year}, {state}, southern={southern}) band names changed".format(
            **fixture
        )
    )

    serialized = stack.serialize()
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if digest != fixture["serialized_sha256"]:
        fresh = os.path.join(
            FIXTURE_DIR, "stack_bands_{}.serialized.fresh.json".format(case)
        )
        with open(fresh, "w") as fp:
            fp.write(serialized)
        old = os.path.join(
            FIXTURE_DIR, "stack_bands_{}.serialized.json.gz".format(case)
        )
        pytest.fail(
            "stack_bands graph changed for {} (fixture ee {} vs current ee {}). "
            "Diff: zcat {} | diff - {}".format(
                case, fixture["earthengine_api_version"], ee.__version__, old, fresh
            )
        )


def test_fixture_cases_present():
    """The four golden cases must exist; guards against fixture deletion."""
    assert set(CASES) == {"MT_1996", "MT_2024", "AZ_1996", "AZ_2024"}
