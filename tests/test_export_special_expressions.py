"""Golden-fixture regression test for ``irrmapper.postproc.legacy.export_special``.

This test locks down the per-state Earth Engine ``.expression()`` strings that
``export_special`` builds for the hardcoded ``range(2022, 2023)`` loop. It runs
with the entire ``ee`` stack mocked, so it needs no Earth Engine credentials and
performs no network access. It exists as a regression gate: if a refactor changes
any expression string (or how many are applied per state), the captured mapping
will diverge from ``tests/fixtures/export_special_expressions.json`` and this test
will fail.

Regenerate the fixture (only when an intentional change is made) with::

    uv run python tests/test_export_special_expressions.py
"""

import json
import os
from unittest import mock
from unittest.mock import MagicMock

from irrmapper.postproc import legacy

# The single year exercised by export_special's hardcoded range(2022, 2023).
YEAR = 2022

# States with dedicated rule branches, plus AZ which falls through to the
# copy_asset "else" branch (no expressions).
STATES = ["MT", "ID", "WA", "OR", "NM", "NV", "CO", "WY", "UT", "CA", "AZ"]

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "export_special_expressions.json",
)


def _capture_state(state):
    """Run export_special for one state with everything mocked.

    Returns a ``(expressions, copy_asset_call_count)`` tuple where ``expressions``
    is the ordered list of first-positional-argument strings passed to every
    ``.expression(...)`` call the function makes.
    """
    mock_ee = MagicMock(name="ee")
    # props = target.getInfo()['properties'] must yield a real, mutable dict.
    mock_ee.Image.return_value.getInfo.return_value = {"properties": {}}

    mock_landsat = MagicMock(name="landsat_composites")

    # get_cdl(year)[1].select('cropland') -> return value must be subscriptable.
    mock_get_cdl = MagicMock(name="get_cdl")
    mock_get_cdl.return_value = (MagicMock(name="cdl0"), MagicMock(name="cdl1"))

    mock_copy = MagicMock(name="copy_asset")

    with mock.patch.multiple(
        "irrmapper.postproc.legacy",
        ee=mock_ee,
        landsat_composites=mock_landsat,
        get_cdl=mock_get_cdl,
        copy_asset=mock_copy,
    ):
        legacy.export_special("in_coll", "out_coll", "roi", state)

    # target starts as ee.Image(...), a descendant of mock_ee, so every chained
    # .expression() call is recorded in mock_ee.mock_calls in invocation order.
    expressions = []
    for call in mock_ee.mock_calls:
        name = call[0]
        if name.split(".")[-1] == "expression":
            expressions.append(call.args[0])

    return expressions, mock_copy.call_count


def build_mapping():
    """Capture the {``STATE_YEAR``: [expression strings]} golden mapping."""
    mapping = {}
    for state in STATES:
        expressions, copy_calls = _capture_state(state)
        mapping["{}_{}".format(state, YEAR)] = expressions
        if state == "AZ":
            # AZ takes the else branch: no expressions, exactly one copy_asset.
            mapping["AZ_copy_asset_called"] = copy_calls == 1
    return mapping


def test_export_special_expressions_match_fixture():
    with open(FIXTURE_PATH) as f:
        expected = json.load(f)

    captured = build_mapping()

    assert captured == expected


if __name__ == "__main__":
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    generated = build_mapping()
    with open(FIXTURE_PATH, "w") as f:
        json.dump(generated, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote {}".format(FIXTURE_PATH))
