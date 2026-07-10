"""State/county code and acreage lookups.

Data-backed loader. The lookup tables live in CSV files alongside this module
(``states.csv``, ``state_county_code.csv``, ``county_acres.csv``) and are read
lazily with the standard-library ``csv`` module. The public API is preserved
exactly: each function returns the same dict structure as before.
"""

import csv
import os

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

_CACHE = {}


def _read_csv(name):
    with open(os.path.join(_DATA_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def state_name_abbreviation():
    if "names" not in _CACHE:
        _CACHE["names"] = {r["abbreviation"]: r["name"] for r in _read_csv("states.csv")}
    return dict(_CACHE["names"])


def state_fips_code():
    if "fips" not in _CACHE:
        _CACHE["fips"] = {r["abbreviation"]: r["fips"] for r in _read_csv("states.csv")}
    return dict(_CACHE["fips"])


def state_county_code():
    if "counties" not in _CACHE:
        out = {}
        for r in _read_csv("state_county_code.csv"):
            out.setdefault(r["state"], {})[r["county_code"]] = {
                "GEOID": r["geoid"],
                "NAME": r["name"],
            }
        _CACHE["counties"] = out
    return {s: {c: dict(v) for c, v in d.items()} for s, d in _CACHE["counties"].items()}


def county_acres():
    if "acres" not in _CACHE:
        _CACHE["acres"] = {
            r["geoid"]: {"land": float(r["land"]), "water": float(r["water"])}
            for r in _read_csv("county_acres.csv")
        }
    return {k: dict(v) for k, v in _CACHE["acres"].items()}
