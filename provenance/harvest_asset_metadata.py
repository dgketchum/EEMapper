"""Harvest production provenance from Earth Engine asset metadata.

Read-only: lists child assets of the IrrMapper image collections, fetches each
asset's metadata via ee.data.getAsset, and writes one CSV per collection with
the union of all `properties` keys plus id/state/year/sizeBytes/updateTime.

No exports are started and no assets are modified.
"""

import os
import re
import sys

import ee
import pandas as pd

COLLECTIONS = [
    "users/dgketchum/IrrMapper/IrrMapper_sw",
    "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp",
    "projects/ee-dgketchum/assets/IrrMapper/version1_2",
    "users/dgketchum/IrrMapper/IrrMapper_RF2",
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Match a trailing STATE_YEAR token, e.g. 'MT_2025'.
NAME_RE = re.compile(r"([A-Z]{2})_(\d{4})$")


def parse_state_year(asset_id):
    """Return (state, year) parsed from the asset's final name token, else (None, None)."""
    base = asset_id.rstrip("/").split("/")[-1]
    m = NAME_RE.search(base)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def list_children(parent):
    """List all child assets of a parent collection, following pagination.

    Returns a list of asset dicts (as returned by ee.data.listAssets), or None
    if the collection does not exist / cannot be listed.
    """
    assets = []
    page_token = None
    try:
        while True:
            params = {"parent": parent}
            if page_token:
                params["pageToken"] = page_token
            resp = ee.data.listAssets(params)
            assets.extend(resp.get("assets", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except ee.EEException as e:
        print("  Could not list {}: {}".format(parent, e))
        return None
    except Exception as e:  # noqa: BLE001 - report and continue to next collection
        print("  Unexpected error listing {}: {}".format(parent, e))
        return None
    return assets


def harvest_collection(parent):
    """Fetch metadata for every child asset of `parent`. Returns a DataFrame or None."""
    print("\n=== {} ===".format(parent))
    children = list_children(parent)
    if children is None:
        print("  SKIPPED (collection not found or not listable)")
        return None

    print("  Found {} child assets".format(len(children)))
    if not children:
        return pd.DataFrame()

    rows = []
    prop_keys = set()
    for i, child in enumerate(children, start=1):
        asset_id = child.get("id") or child.get("name")
        try:
            meta = ee.data.getAsset(asset_id)
        except Exception as e:  # noqa: BLE001 - log the bad asset and keep going
            print("  getAsset FAILED for {}: {}".format(asset_id, e))
            state, year = parse_state_year(asset_id)
            rows.append(
                {
                    "asset_id": asset_id,
                    "name": child.get("name"),
                    "STATE": state,
                    "YEAR": year,
                    "sizeBytes": None,
                    "updateTime": None,
                    "getAsset_error": str(e),
                }
            )
            continue

        name = meta.get("name", asset_id)
        state, year = parse_state_year(name)
        props = meta.get("properties", {}) or {}
        prop_keys.update(props.keys())

        row = {
            "asset_id": meta.get("id", asset_id),
            "name": name,
            "STATE": state,
            "YEAR": year,
            "sizeBytes": meta.get("sizeBytes"),
            "updateTime": meta.get("updateTime"),
        }
        row.update(props)
        rows.append(row)

        if i % 50 == 0:
            print("  ...processed {}/{} assets".format(i, len(children)))

    fixed_cols = ["asset_id", "name", "STATE", "YEAR", "sizeBytes", "updateTime"]
    prop_cols = sorted(prop_keys)
    extra = [c for c in ("getAsset_error",) if any(c in r for r in rows)]
    columns = fixed_cols + prop_cols + extra

    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = None
    df = df[columns]
    df = df.sort_values(by=["STATE", "YEAR"], na_position="last").reset_index(drop=True)
    return df


def summarize(parent, df):
    """Print a per-collection summary: counts, per-state year range, missing props."""
    coll_name = parent.rstrip("/").split("/")[-1]
    print("\n--- summary: {} ---".format(coll_name))
    if df is None:
        print("  (collection unavailable)")
        return
    print("  asset count: {}".format(len(df)))
    if df.empty:
        return

    print("  year range per state:")
    for state, grp in df.groupby("STATE", dropna=False):
        years = sorted(y for y in grp["YEAR"].dropna().unique())
        if years:
            rng = "{}-{} ({} yrs)".format(int(min(years)), int(max(years)), len(years))
        else:
            rng = "no parsable year"
        print("    {}: {}".format(state, rng))

    for prop in ("post_process", "training_data"):
        if prop not in df.columns:
            print('  ALL assets MISSING property "{}"'.format(prop))
            continue
        missing = df[df[prop].isna() | (df[prop].astype(str).str.strip() == "")]
        if len(missing):
            ids = ", ".join(
                missing["name"].fillna(missing["asset_id"]).astype(str).tolist()
            )
            print('  {} asset(s) MISSING "{}": {}'.format(len(missing), prop, ids))
        else:
            print('  all assets carry "{}"'.format(prop))


def main():
    try:
        ee.Initialize(project="ee-dgketchum")
    except Exception as e:  # noqa: BLE001 - cannot proceed without EE
        print("ee.Initialize failed: {}".format(e))
        print("Not attempting interactive auth. Aborting.")
        sys.exit(1)
    print("Earth Engine initialized (project=ee-dgketchum)")

    written = []
    for parent in COLLECTIONS:
        df = harvest_collection(parent)
        summarize(parent, df)
        if df is not None:
            coll_name = parent.rstrip("/").split("/")[-1]
            out_path = os.path.join(OUT_DIR, "{}_metadata.csv".format(coll_name))
            df.to_csv(out_path, index=False)
            written.append(out_path)
            print("  wrote {}".format(out_path))

    print("\nWrote {} CSV file(s):".format(len(written)))
    for p in written:
        print("  {}".format(p))


if __name__ == "__main__":
    main()
