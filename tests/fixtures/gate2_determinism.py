"""Gate 2 determinism baseline: classify a small ROI twice, diff the pixels.

Submits the production classification (call_ee.export_classification, MT
model, 2020) twice in one session over a ~30 km box on the Greenfields
Bench near Fairfield, MT — a dense center-pivot district. The two requests
differ only by a provenance property (so they cannot be served as one
cached request), export as GeoTIFF to gs://wudr, and are then downloaded
and compared pixel-for-pixel.

Identical outputs establish the baseline Gate 2 needs: a same-day re-run
of the pipeline reproduces the product exactly (RF seed, compositing, and
batch execution are deterministic). It does not guard against drift in the
source collections over time — that is why parity runs are same-day.

Run (long-lived: submits, polls, downloads, diffs):
    uv run python tests/fixtures/gate2_determinism.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

import ee

from map.call_ee import export_classification, is_authorized
from map.config import resolve_path

STATE, YEAR, GLOB = "MT", 2020, "09MAY2023"
TABLE = "users/dgketchum/bands/state/{}_{}".format(STATE, GLOB)
# Greenfields Bench, Fairfield MT: heavily irrigated, mixed pivot/flood
ROI_RECT = [-112.10, 47.45, -111.70, 47.70]
RUNS = ("a", "b")
OUT_DIR = "/nas/irrmapper/gate2_determinism"
BUCKET = "gs://wudr"
POLL_S, TIMEOUT_S = 45, 3 * 3600


def _log(msg):
    print("{} {}".format(datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def _features():
    path = os.path.join(
        resolve_path("provenance/variable_importance"),
        "variables_{}_{}.json".format(STATE, GLOB),
    )
    with open(path) as fp:
        return [f[0] for f in json.load(fp)[STATE]]


def _out_name(run):
    return "gate2_det_{}".format(run)


def submit():
    roi = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(ROI_RECT))])
    features = _features()
    for run in RUNS:
        export_classification(
            out_name=_out_name(run),
            table=TABLE,
            asset_root="gate2",
            region=roi,
            years=[YEAR],
            export="cloud",
            bag_fraction=0.5,
            input_props=features,
            southern=False,
            extra_props={"gate2_run": run},
        )
        _log("submitted {}".format(_out_name(run)))


def wait():
    descs = {"{}_{}".format(_out_name(run), YEAR) for run in RUNS}
    tasks = {}
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        for t in ee.batch.Task.list()[:50]:
            d = t.status().get("description")
            if d in descs and d not in tasks:
                tasks[d] = t
        states = {d: t.status().get("state") for d, t in tasks.items()}
        _log("task states: {}".format(states))
        if len(states) == len(descs) and all(
            s in ("COMPLETED", "FAILED", "CANCELLED") for s in states.values()
        ):
            for d, t in tasks.items():
                status = t.status()
                if status["state"] != "COMPLETED":
                    _log("{} {}: {}".format(d, status["state"],
                                            status.get("error_message")))
                    return False
            return True
        time.sleep(POLL_S)
    _log("timed out waiting for tasks")
    return False


def fetch():
    os.makedirs(OUT_DIR, exist_ok=True)
    for run in RUNS:
        pattern = "{}/{}_{}*.tif".format(BUCKET, YEAR, _out_name(run))
        subprocess.run(["gsutil", "cp", pattern, OUT_DIR], check=True)
        _log("fetched {}".format(pattern))


def diff():
    import numpy as np
    import rasterio

    result = {"state": STATE, "year": YEAR, "roi": ROI_RECT,
              "compared": [], "identical": True,
              "created": datetime.now().isoformat()}
    pairs = []
    for run in RUNS:
        files = sorted(glob(
            os.path.join(OUT_DIR, "{}_{}*.tif".format(YEAR, _out_name(run)))))
        pairs.append(files)
    if len(pairs[0]) != len(pairs[1]) or not pairs[0]:
        raise RuntimeError("shard mismatch: {} vs {}".format(*pairs))

    for fa, fb in zip(*pairs):
        with rasterio.open(fa) as a, rasterio.open(fb) as b:
            arr_a, arr_b = a.read(1), b.read(1)
            grid_same = (a.crs == b.crs and a.transform == b.transform
                         and arr_a.shape == arr_b.shape)
            n_diff = int((arr_a != arr_b).sum()) if grid_same else -1
            result["compared"].append({
                "a": fa, "b": fb, "grid_same": grid_same,
                "n_pixels": int(arr_a.size), "n_diff": n_diff,
                "class_counts_a": {int(k): int(v) for k, v in
                                   zip(*np.unique(arr_a, return_counts=True))},
            })
            if not grid_same or n_diff:
                result["identical"] = False
        _log("{} vs {}: grid_same={} n_diff={}".format(
            os.path.basename(fa), os.path.basename(fb), grid_same, n_diff))

    out_json = os.path.join(OUT_DIR, "gate2_determinism_result.json")
    with open(out_json, "w") as fp:
        json.dump(result, fp, indent=2)
    _log("result written to {}".format(out_json))
    return result["identical"]


if __name__ == "__main__":
    if not is_authorized(project="ee-dgketchum"):
        sys.exit(2)
    submit()
    if not wait():
        sys.exit(2)
    fetch()
    identical = diff()
    _log("DETERMINISM BASELINE: {}".format(
        "IDENTICAL — pass" if identical else "DIFFERS — investigate"))
    sys.exit(0 if identical else 1)
