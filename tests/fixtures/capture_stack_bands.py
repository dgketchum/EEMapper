"""Capture golden fixtures for irrmapper.features.stack.stack_bands.

Builds the feature-stack expression graph for a small matrix of
(state, southern-flag, year) cases and records, per case:

  - stack_bands_{ST}_{year}.json          band names, graph SHA-256, ee version
  - stack_bands_{ST}_{year}.serialized.json.gz   full serialized graph, for diffing

Building the graph makes two metadata-level server calls inside stack_bands
(projection().getInfo() and bandNames().getInfo()) so this script needs
authenticated Earth Engine credentials. It starts no exports and runs no
compute. Re-run only when an intentional change to the feature stack is
made; tests/test_stack_bands_graph.py compares fresh graphs against these
fixtures.

Usage:
    uv run python tests/fixtures/capture_stack_bands.py
"""

import gzip
import hashlib
import json
import os

import ee

from irrmapper.auth import is_authorized
from irrmapper.features.stack import stack_bands

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

BOUNDARIES = "users/dgketchum/boundaries"

# (state, southern) x years: MT exercises the northern seasonal windows,
# AZ the southern; 1996 is TM-era, 2024 is OLI-era.
CASES = [("MT", False), ("AZ", True)]
YEARS = [1996, 2024]


def capture_case(state, southern, year):
    roi = ee.FeatureCollection("{}/{}".format(BOUNDARIES, state))
    stack = stack_bands(year, roi, southern=southern)

    serialized = stack.serialize()
    band_names = stack.bandNames().getInfo()
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    tag = "{}_{}".format(state, year)

    summary = {
        "state": state,
        "year": year,
        "southern": southern,
        "roi": "{}/{}".format(BOUNDARIES, state),
        "earthengine_api_version": ee.__version__,
        "n_bands": len(band_names),
        "band_names": band_names,
        "serialized_sha256": digest,
    }
    summary_file = os.path.join(FIXTURE_DIR, "stack_bands_{}.json".format(tag))
    with open(summary_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    graph_file = os.path.join(
        FIXTURE_DIR, "stack_bands_{}.serialized.json.gz".format(tag)
    )
    with gzip.open(graph_file, "wt") as fp:
        fp.write(serialized)

    print("{}: {} bands, sha256 {}...".format(tag, len(band_names), digest[:12]))
    return summary


if __name__ == "__main__":
    is_authorized()
    for state_, southern_ in CASES:
        for year_ in YEARS:
            capture_case(state_, southern_, year_)
