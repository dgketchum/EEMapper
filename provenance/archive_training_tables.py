"""Archive the production training tables out of Earth Engine.

The per-state band-extract tables (users/dgketchum/bands/state/{ST}_{vintage})
ARE the v1.2 training data and, until archived, existed only as private EE
assets — the last unarchived stage of the reproduction chain
(docs/reproducibility.md stage 6). This job exports each table referenced by
the canonical run config to GCS as CSV, downloads them to
/nas/irrmapper/training_table_archive/{vintage}/, checksums them, and writes
a manifest (also committed at provenance/training_table_archive_{vintage}.json)
so the files can be staged for the versioned Zenodo training-data record.

Run (long-lived: submits 11 table exports, polls, downloads, checksums):
    uv run python provenance/archive_training_tables.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob

import ee

from irrmapper.auth import is_authorized
from irrmapper.config import _git_sha, load_config, repo_root

CONFIG = os.path.join(repo_root(), "configs", "irrmapper_v1_2.toml")
GCS_PREFIX = "training_table_archive"
OUT_ROOT = "/nas/irrmapper/training_table_archive"
POLL_S, TIMEOUT_S = 45, 2 * 3600


def _log(msg):
    print("{} {}".format(datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def _desc(state, vintage):
    return "archive_{}_{}".format(state, vintage)


def submit(cfg):
    for state in cfg.run.states:
        name = "{}_{}".format(state, cfg.run.vintage_glob)
        fc = ee.FeatureCollection(os.path.join(cfg.paths.training_tables, name))
        task = ee.batch.Export.table.toCloudStorage(
            fc,
            description=_desc(state, cfg.run.vintage_glob),
            bucket=cfg.project.gcs_bucket,
            fileNamePrefix="{}/{}".format(GCS_PREFIX, name),
            fileFormat="CSV",
        )
        task.start()
        _log("submitted {}".format(name))


def wait(cfg):
    descs = {_desc(s, cfg.run.vintage_glob) for s in cfg.run.states}
    tasks = {}
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        for t in ee.batch.Task.list()[:80]:
            d = t.status().get("description")
            if d in descs and d not in tasks:
                tasks[d] = t
        states = {d: t.status().get("state") for d, t in tasks.items()}
        n_done = sum(1 for s in states.values() if s == "COMPLETED")
        _log("{}/{} completed".format(n_done, len(descs)))
        if len(states) == len(descs) and all(
            s in ("COMPLETED", "FAILED", "CANCELLED") for s in states.values()
        ):
            failed = {
                d: t.status()
                for d, t in tasks.items()
                if t.status()["state"] != "COMPLETED"
            }
            for d, status in failed.items():
                _log(
                    "{} {}: {}".format(d, status["state"], status.get("error_message"))
                )
            return not failed
        time.sleep(POLL_S)
    _log("timed out waiting for exports")
    return False


def fetch(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pattern = "gs://{}/{}/*.csv".format(cfg.project.gcs_bucket, GCS_PREFIX)
    subprocess.run(["gsutil", "-m", "cp", pattern, out_dir], check=True)
    _log("fetched {} -> {}".format(pattern, out_dir))


def checksum(cfg, out_dir):
    manifest = {
        "vintage": cfg.run.vintage_glob,
        "source_root": cfg.paths.training_tables,
        "archive_dir": out_dir,
        "git_sha": _git_sha(),
        "earthengine_api_version": ee.__version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "tables": {},
    }
    for state in cfg.run.states:
        name = "{}_{}".format(state, cfg.run.vintage_glob)
        files = sorted(glob(os.path.join(out_dir, "{}*.csv".format(name))))
        if not files:
            raise RuntimeError("no archived file for {}".format(name))
        entries = []
        for f in files:
            with open(f, "rb") as fp:
                digest = hashlib.sha256(fp.read()).hexdigest()
            with open(f) as fp:
                n_rows = sum(1 for _ in fp) - 1
            entries.append(
                {
                    "file": os.path.basename(f),
                    "sha256": digest,
                    "size_bytes": os.path.getsize(f),
                    "n_rows": n_rows,
                }
            )
            _log(
                "{}: {} rows, sha256 {}".format(
                    os.path.basename(f), n_rows, digest[:12]
                )
            )
        manifest["tables"][name] = {
            "asset_id": os.path.join(cfg.paths.training_tables, name),
            "gcs_uri": "gs://{}/{}/{}.csv".format(
                cfg.project.gcs_bucket, GCS_PREFIX, name
            ),
            "files": entries,
        }

    for dst in (
        os.path.join(out_dir, "manifest.json"),
        os.path.join(
            repo_root(),
            "provenance",
            "training_table_archive_{}.json".format(cfg.run.vintage_glob),
        ),
    ):
        with open(dst, "w") as fp:
            json.dump(manifest, fp, indent=2, sort_keys=True)
        _log("manifest written to {}".format(dst))


if __name__ == "__main__":
    cfg = load_config(CONFIG)
    if not is_authorized(project=cfg.project.ee_project):
        sys.exit(2)
    out_dir = os.path.join(OUT_ROOT, cfg.run.vintage_glob)
    submit(cfg)
    if not wait(cfg):
        sys.exit(2)
    fetch(cfg, out_dir)
    checksum(cfg, out_dir)
    _log("ARCHIVE COMPLETE: {}".format(out_dir))
