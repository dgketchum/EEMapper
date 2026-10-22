"""Per-state-year GeoTIFF export of the v1.2 boolean collection for Zenodo.

Exports each ``version1_2/{ST}_{YEAR}`` asset to GCS on its *native* grid
(crs + crsTransform read from the asset), as a single byte band:
1 = irrigated, 0 = not irrigated, 255 = nodata outside the state boundary.
The private assets store value 0 where irrigated and mask everything else
(``legacy/assets_cli.py::convert_to_boolean_and_export``), so the export
remaps with ``eq(0)`` and fills the mask with 0 before clipping to the state.
No min-years mask. The native export is a faithful copy of the collection,
which is *not* on one grid (see ``native_grid``); ``warp`` then re-grids the
fetched files locally onto one padded 30 m EPSG:5070 grid per state, on the
lattice the Albers assets already use, so all years of a state stack and
neighbouring states align pixel for pixel. Local GDAL warping costs no EECU.

Stages: ``submit`` (EE tasks, writes a manifest of description -> task id and
pixel count), ``report`` (EECU-h per task and per gigapixel from the EE task
list), ``fetch`` (gsutil rsync GCS -> disk), ``warp`` (native -> Albers COGs
with a state-polygon cutline, plus ``grids.json`` for provenance), ``check``
(edge zero fractions and value set, the QGIS reviewer's pass criterion),
``package`` (one zip per state with README, grids and boundary, plus SHA256
lists), ``stats`` (pixel counts per file, parallel), ``upload`` (Zenodo REST
API: create or resume a draft deposition, metadata, files with MD5 check; no
publish unless asked) and ``cancel``. ``submit --grid albers`` does the re-gridding on EE instead; kept
as an option, not the default path.

The export region must be the asset's exact pixel rectangle (see
``native_grid``): the 2026-09-22 QGIS review found a 1-10 px frame of 0 around
every file exported with ``img.geometry()`` as the region.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone

import ee

from irrmapper.assets.ops import list_assets

DEFAULT_COLLECTION = "projects/ee-dgketchum/assets/IrrMapper/version1_2"
DEFAULT_BUCKET = "wudr"
DEFAULT_PREFIX = "irrmapper_zenodo_v1_2"
DEFAULT_BOUNDARIES = "users/dgketchum/boundaries"
DEFAULT_TIF_DIR = "/nas/irrmapper/zenodo_v1_2/tif"
DEFAULT_ALBERS_DIR = "/nas/irrmapper/zenodo_v1_2/tif_albers"
DEFAULT_BOUNDARY_DIR = "/nas/irrmapper/zenodo_v1_2/boundaries"
DEFAULT_DEPOSIT_DIR = "/nas/irrmapper/zenodo_v1_2/deposit"
ZENODO_API = {
    "zenodo": "https://zenodo.org/api",
    "sandbox": "https://sandbox.zenodo.org/api",
}
NODATA = 255
ALBERS = "EPSG:5070"
SCALE = 30
LATTICE = 15  # pixel edges of the Albers assets fall at 15 mod 30 m
GRIDS = ("native", "albers")
STATES = ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"]


def _description(state, year):
    return "IrrMapper_v1_2_{}_{}".format(state, year)


def native_grid(asset_id, band="classification"):
    """Grid of the asset's own band: (crs, crs_transform, region, pixels).

    The collection is not uniform (census 2026-09-22): 162 assets sit on a
    30 m EPSG:5070 lattice with pixel edges at 15 mod 30 m, the other 288
    are EPSG:4326 at ~0.000269 deg. A native-grid export reproduces that mix.
    """
    img = ee.Image(asset_id).select(band)
    info = img.getInfo()["bands"][0]
    w, h = info["dimensions"]
    crs, t = info["crs"], info["crs_transform"]
    x0, y1 = t[2], t[5]
    x1, y0 = x0 + t[0] * w, y1 + t[4] * h
    # Do not use img.geometry() as the region: the stored footprint is a
    # lon/lat polygon whose edges bow outside the pixel rectangle once
    # projected (47-288 m for MT/WY), so EE pads the export by several pixels
    # and writes that outside-footprint padding as masked, i.e. 0. The exact
    # planar pixel rectangle reproduces the asset's own extent instead.
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], ee.Projection(crs), False)
    return crs, t, region, w * h


def albers_grid(roi, pad_pixels=2):
    """One 30 m EPSG:5070 grid per state, on the lattice the Albers assets use.

    Snaps the state's bounding box outward to pixel edges at 15 mod 30 m and
    pads it, so every year of a state shares an identical grid and neighboring
    states align pixel for pixel. Assets stored in EPSG:4326 are resampled
    (nearest neighbour) by the export.
    """
    geom = ee.FeatureCollection(roi).geometry()
    coords = geom.bounds(1, ee.Projection(ALBERS)).coordinates().getInfo()[0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pad = pad_pixels * SCALE
    x0 = math.floor((min(xs) - LATTICE) / SCALE) * SCALE + LATTICE - pad
    x1 = math.ceil((max(xs) - LATTICE) / SCALE) * SCALE + LATTICE + pad
    y0 = math.floor((min(ys) - LATTICE) / SCALE) * SCALE + LATTICE - pad
    y1 = math.ceil((max(ys) - LATTICE) / SCALE) * SCALE + LATTICE + pad
    w = int(round((x1 - x0) / SCALE))
    h = int(round((y1 - y0) / SCALE))
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], ALBERS, False)
    return ALBERS, [SCALE, 0, x0, 0, -SCALE, y1], region, w * h


def export_on_grid(
    asset_id,
    roi,
    grid,
    description,
    bucket,
    prefix,
    cloud_optimized=True,
    band="classification",
):
    """Start one export of ``asset_id`` on ``grid``; return (task, pixels)."""
    crs, transform, region, pixels = grid
    src = ee.Image(asset_id)
    state = ee.FeatureCollection(roi).geometry()
    # EE writes masked pixels as 0 regardless of the noData tag, so fill the
    # outside-state mask with NODATA explicitly after the clip. The clip
    # shrinks the footprint to the state, and unmask() defaults to
    # sameFootprint=True, so the fill must be told to extend past it.
    img = (
        src.select(band)
        .eq(0)
        .unmask(0)
        .clip(state)
        .unmask(NODATA, sameFootprint=False)
        .toByte()
    )
    task = ee.batch.Export.image.toCloudStorage(
        image=img,
        description=description,
        bucket=bucket,
        fileNamePrefix="{}/{}".format(prefix, description),
        region=region,
        crs=crs,
        crsTransform=transform,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": cloud_optimized, "noData": NODATA},
    )
    task.start()
    return task, pixels


def submit(
    states,
    years,
    manifest_path,
    collection=DEFAULT_COLLECTION,
    bucket=DEFAULT_BUCKET,
    prefix=DEFAULT_PREFIX,
    boundaries=DEFAULT_BOUNDARIES,
    cloud_optimized=True,
    grid="native",
):
    if grid not in GRIDS:
        raise ValueError("grid must be one of {}".format(GRIDS))
    manifest = _load_manifest(manifest_path)
    present = set(list_assets(collection))
    for state in states:
        roi = "{}/{}".format(boundaries, state)
        state_grid = albers_grid(roi) if grid == "albers" else None
        for year in years:
            desc = _description(state, year)
            if desc in manifest:
                print("skip {} (already submitted)".format(desc))
                continue
            asset_id = "{}/{}_{}".format(collection, state, year)
            if asset_id not in present:
                # CA has no 1985 image; the collection is otherwise complete
                print("skip {} (no asset {})".format(desc, asset_id))
                continue
            g = state_grid if state_grid is not None else native_grid(asset_id)
            task, pixels = export_on_grid(
                asset_id, roi, g, desc, bucket, prefix, cloud_optimized
            )
            manifest[desc] = {
                "asset": asset_id,
                "task": task.id,
                "grid": grid,
                "crs": g[0] if grid == "albers" else "native",
                "pixels": pixels,
                "uri": "gs://{}/{}/{}.tif".format(bucket, prefix, desc),
                "submitted": datetime.now(timezone.utc).isoformat(),
            }
            print("{}  {}  {:.3f} Gpx".format(desc, task.id, pixels / 1e9))
            _write_manifest(manifest, manifest_path)
    return manifest


def report(manifest_path):
    """Join the manifest to the EE task list and print EECU per task and per Gpx."""
    manifest = _load_manifest(manifest_path)
    by_desc = {}
    for op in ee.data.listOperations():
        meta = op.get("metadata", {})
        by_desc.setdefault(meta.get("description"), meta)

    rows, eecu_total, px_total = [], 0.0, 0
    for desc, rec in sorted(manifest.items()):
        meta = by_desc.get(desc)
        state = meta.get("state") if meta else "NOT_IN_TASK_LIST"
        eecu_s = (meta or {}).get("batchEecuUsageSeconds")
        eecu_h = eecu_s / 3600.0 if eecu_s is not None else None
        rows.append((desc, state, rec["pixels"] / 1e9, eecu_h))
        if state == "SUCCEEDED" and eecu_h is not None:
            eecu_total += eecu_h
            px_total += rec["pixels"]

    print(
        "{:<28} {:<12} {:>8} {:>10} {:>12}".format(
            "task", "state", "Gpx", "EECU-h", "EECU-h/Gpx"
        )
    )
    for desc, state, gpx, eecu_h in rows:
        eecu_txt = (
            "{:10.3f}".format(eecu_h) if eecu_h is not None else "{:>10}".format("-")
        )
        rate_txt = (
            "{:12.3f}".format(eecu_h / gpx)
            if eecu_h is not None and gpx
            else "{:>12}".format("-")
        )
        print(
            "{:<28} {:<12} {:8.3f} {} {}".format(desc, state, gpx, eecu_txt, rate_txt)
        )

    if px_total:
        rate = eecu_total / (px_total / 1e9)
        print(
            "\nsucceeded: {:.3f} EECU-h over {:.3f} Gpx = {:.3f} EECU-h/Gpx".format(
                eecu_total, px_total / 1e9, rate
            )
        )
    return rows


def fetch(bucket, prefix, tif_dir):
    """Mirror the exported GeoTIFFs from GCS into ``tif_dir`` (gsutil rsync)."""
    os.makedirs(tif_dir, exist_ok=True)
    cmd = [
        "gsutil",
        "-m",
        "-q",
        "rsync",
        "-x",
        r".*\.aux\.xml$",
        "gs://{}/{}/".format(bucket, prefix),
        tif_dir,
    ]
    subprocess.run(cmd, check=True)
    n = len([f for f in os.listdir(tif_dir) if f.endswith(".tif")])
    print("{} GeoTIFFs in {}".format(n, tif_dir))
    return n


def state_grids(states, boundaries=DEFAULT_BOUNDARIES, path=None):
    """Per-state Albers grids as plain data; cached in ``path`` when given."""
    grids = {}
    if path and os.path.exists(path):
        with open(path) as f:
            grids = json.load(f)
    for state in states:
        if state in grids:
            continue
        crs, transform, region, pixels = albers_grid("{}/{}".format(boundaries, state))
        coords = region.coordinates().getInfo()[0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        grids[state] = {
            "crs": crs,
            "transform": transform,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "width": int(round((max(xs) - min(xs)) / SCALE)),
            "height": int(round((max(ys) - min(ys)) / SCALE)),
        }
        if path:
            _write_json(grids, path)
    return grids


def fetch_boundaries(
    states, boundaries=DEFAULT_BOUNDARIES, out_dir=DEFAULT_BOUNDARY_DIR
):
    """Write each state's EE boundary polygon to ``out_dir/{ST}.fgb`` (EPSG:4326).

    These are the very polygons the export clipped with, so a gdalwarp cutline
    on them keeps the footprint where EE put it.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for state in states:
        path = os.path.join(out_dir, state + ".fgb")
        if not os.path.exists(path):
            fc = ee.FeatureCollection("{}/{}".format(boundaries, state)).getInfo()
            geoms = [shape(f["geometry"]) for f in fc["features"]]
            # geometry must come in the dict: GeoDataFrame(data, geometry=list)
            # yields a column shapely predicates reject (geopandas 1.1.2)
            gdf = gpd.GeoDataFrame(
                {"STUSPS": [state] * len(geoms), "geometry": geoms}, crs="EPSG:4326"
            )
            gdf.to_file(path, driver="FlatGeobuf", engine="fiona")
        paths[state] = path
    return paths


def edge_check(path, n=6):
    """Fraction of 0 in the outermost ``n`` rows/columns, and the value set.

    Index 0 is the outermost row/column. Pixels outside the state must be
    NODATA, so the fractions should be near 0 except where the state line
    itself touches the bounding box (CO's 41N top, WY's corners, MT's 49N).
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
        edges = {
            "top": ds.read(1, window=Window(0, 0, w, n)),
            "bottom": ds.read(1, window=Window(0, h - n, w, n))[::-1],
            "left": ds.read(1, window=Window(0, 0, n, h)).T,
            "right": ds.read(1, window=Window(w - n, 0, n, h)).T[::-1],
        }
        values = set()
        for _, win in ds.block_windows(1):
            values.update(np.unique(ds.read(1, window=win)).tolist())
    zero_frac = {
        k: [round(float((a[i] == 0).mean()), 3) for i in range(n)]
        for k, a in edges.items()
    }
    return zero_frac, sorted(values)


def cancel(manifest_path):
    """Cancel every task in the manifest that is still queued or running."""
    manifest = _load_manifest(manifest_path)
    ids = [rec["task"] for rec in manifest.values()]
    counts = {}
    for i in range(0, len(ids), 50):
        for status in ee.data.getTaskStatus(ids[i : i + 50]):
            state = status["state"]
            if state in ("READY", "RUNNING"):
                ee.data.cancelTask(status["id"])
                state = "cancelled:" + state
            counts[state] = counts.get(state, 0) + 1
    print(counts)
    return counts


def warp_one(src, dst, grid, cutline=None):
    """gdalwarp ``src`` onto ``grid`` (nearest neighbour, byte, 255 nodata, COG).

    With ``cutline`` (a vector file), everything outside the polygon becomes
    NODATA regardless of what the source holds: a safety net against masked
    padding written as 0, and it trims the 1-2 px band along shared borders
    where both neighbours carry data, so a 255-nodata mosaic is seamless.
    """
    x0, y0, x1, y1 = grid["bounds"]
    cmd = [
        "gdalwarp",
        "-overwrite",
        "-t_srs",
        grid["crs"],
        "-tr",
        str(SCALE),
        str(SCALE),
        "-te",
        str(x0),
        str(y0),
        str(x1),
        str(y1),
        "-r",
        "near",
        "-srcnodata",
        str(NODATA),
        "-dstnodata",
        str(NODATA),
        "-ot",
        "Byte",
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-of",
        "COG",
        "-co",
        "COMPRESS=LZW",
        "-co",
        "BLOCKSIZE=256",
    ]
    if cutline:
        cmd += ["-cutline", cutline]
    cmd += [src, dst]
    subprocess.run(cmd, check=True, capture_output=True)


def warp(
    manifest_path,
    tif_dir,
    out_dir,
    states=None,
    boundaries=DEFAULT_BOUNDARIES,
    overwrite=False,
    cutline=True,
    boundary_dir=DEFAULT_BOUNDARY_DIR,
    jobs=1,
):
    """Re-grid every native export in the manifest onto its state's Albers grid.

    Writes ``out_dir/{desc}.tif`` and ``out_dir/grids.json`` (the per-state
    grids, for provenance). Skips entries whose native GeoTIFF has not been
    fetched yet and, unless ``overwrite``, entries already warped. With
    ``cutline`` the state polygon is applied again locally (see ``warp_one``).
    ``jobs`` gdalwarp processes run at once: a cutline warp is single-threaded
    in practice (MT 2021: 250 s at 102% CPU, 56 s without the cutline), so on
    a many-core box the run scales almost linearly with ``jobs``.
    """
    manifest = _load_manifest(manifest_path)
    os.makedirs(out_dir, exist_ok=True)
    wanted = {
        d: r for d, r in manifest.items() if states is None or d.split("_")[3] in states
    }
    wanted_states = sorted({d.split("_")[3] for d in wanted})
    grids = state_grids(wanted_states, boundaries, os.path.join(out_dir, "grids.json"))
    cutlines = (
        fetch_boundaries(wanted_states, boundaries, boundary_dir) if cutline else {}
    )
    todo, missing, skipped = [], [], 0
    for desc in sorted(wanted):
        src = os.path.join(tif_dir, desc + ".tif")
        dst = os.path.join(out_dir, desc + ".tif")
        if not os.path.exists(src):
            missing.append(desc)
            continue
        if os.path.exists(dst) and not overwrite:
            skipped += 1
            continue
        todo.append((desc, src, dst))

    def _run(item):
        desc, src, dst = item
        t0 = time.time()
        state = desc.split("_")[3]
        warp_one(src, dst, grids[state], cutlines.get(state))
        return desc, time.time() - t0

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for desc, seconds in pool.map(_run, todo):
            done += 1
            print("{}  {:.0f} s".format(desc, seconds), flush=True)
    print(
        "warped {}, already present {}, native not fetched {}".format(
            done, skipped, len(missing)
        )
    )
    if missing:
        print("missing: {} ... {}".format(missing[0], missing[-1]))
    return done, skipped, missing


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(
    albers_dir=DEFAULT_ALBERS_DIR,
    out_dir=DEFAULT_DEPOSIT_DIR,
    boundary_dir=DEFAULT_BOUNDARY_DIR,
    states=STATES,
    overwrite=False,
):
    """Zip the Albers GeoTIFFs one archive per state for the Zenodo deposit.

    ``out_dir/IrrMapper_v1_2_{ST}.zip`` holds every year of the state plus
    ``README.md``, ``grids.json`` and the state's boundary polygon. The
    GeoTIFFs are already LZW-compressed, so members are stored, not deflated.
    Also writes ``SHA256SUMS_tif.txt`` (every GeoTIFF) and ``SHA256SUMS_zip.txt``
    (every archive) in ``out_dir``. ``README.md`` must already be in ``out_dir``.
    """
    readme = os.path.join(out_dir, "README.md")
    grids = os.path.join(albers_dir, "grids.json")
    for required in (readme, grids):
        if not os.path.exists(required):
            raise FileNotFoundError(required)
    tif_sums = {}
    zip_sums = {}
    for state in states:
        tifs = sorted(
            f
            for f in os.listdir(albers_dir)
            if f.startswith("IrrMapper_v1_2_{}_".format(state)) and f.endswith(".tif")
        )
        if not tifs:
            raise FileNotFoundError(
                "no GeoTIFFs for {} in {}".format(state, albers_dir)
            )
        for f in tifs:
            tif_sums[f] = _sha256(os.path.join(albers_dir, f))
        dst = os.path.join(out_dir, "IrrMapper_v1_2_{}.zip".format(state))
        if os.path.exists(dst) and not overwrite:
            print("skip {} (exists)".format(os.path.basename(dst)))
        else:
            t0 = time.time()
            with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as z:
                z.write(readme, "README.md")
                z.write(grids, "grids.json")
                z.write(
                    os.path.join(boundary_dir, state + ".fgb"),
                    "boundary_{}.fgb".format(state),
                )
                for f in tifs:
                    z.write(os.path.join(albers_dir, f), f)
            print(
                "{}  {} GeoTIFFs  {:.0f} MB  {:.0f} s".format(
                    os.path.basename(dst),
                    len(tifs),
                    os.path.getsize(dst) / 1e6,
                    time.time() - t0,
                ),
                flush=True,
            )
        zip_sums[os.path.basename(dst)] = _sha256(dst)
    for name, sums in (
        ("SHA256SUMS_tif.txt", tif_sums),
        ("SHA256SUMS_zip.txt", zip_sums),
    ):
        with open(os.path.join(out_dir, name), "w") as f:
            for k in sorted(sums):
                f.write("{}  {}\n".format(sums[k], k))
    print("{} GeoTIFFs in {} archives".format(len(tif_sums), len(zip_sums)))
    return tif_sums, zip_sums


def file_stats(path, grids=None):
    """Pixel counts of 0/1/255 and grid conformance of one Albers GeoTIFF."""
    import numpy as np
    import rasterio

    desc = os.path.basename(path)[:-4]
    state, year = desc.split("_")[3], int(desc.split("_")[4])
    counts = np.zeros(256, dtype=np.int64)
    with rasterio.open(path) as ds:
        for _, win in ds.block_windows(1):
            counts += np.bincount(ds.read(1, window=win).ravel(), minlength=256)
        t = ds.transform
        rec = {
            "state": state,
            "year": year,
            "crs": ds.crs.to_string(),
            "dtype": ds.dtypes[0],
            "nodata": ds.nodata,
            "width": ds.width,
            "height": ds.height,
            "x0": t.c,
            "y1": t.f,
        }
    other = int(counts.sum() - counts[0] - counts[1] - counts[255])
    rec.update(
        {
            "n_irrigated": int(counts[1]),
            "n_not_irrigated": int(counts[0]),
            "n_nodata": int(counts[255]),
            "n_other_values": other,
            "irrigated_km2": float(counts[1] * SCALE * SCALE / 1e6),
            "state_km2": float((counts[0] + counts[1]) * SCALE * SCALE / 1e6),
            "irrigated_frac": float(counts[1] / max(1, counts[0] + counts[1])),
        }
    )
    g = (grids or {}).get(state)
    rec["grid_ok"] = bool(
        g
        and (rec["width"], rec["height"]) == (g["width"], g["height"])
        and t.c == g["transform"][2]
        and t.f == g["transform"][5]
        and rec["crs"] == g["crs"]
        and rec["nodata"] == NODATA
        and rec["dtype"] == "uint8"
        and other == 0
    )
    return rec


def stats(albers_dir=DEFAULT_ALBERS_DIR, out_csv=None, jobs=8):
    """Per-file pixel stats for every Albers GeoTIFF, in parallel; CSV + table.

    Returns the list of records sorted by state and year. Each is checked
    against ``grids.json`` (``grid_ok``). Workers are plain processes; run the
    CLI under ``nice`` to stay polite on a shared box.
    """
    import csv

    with open(os.path.join(albers_dir, "grids.json")) as f:
        grids = json.load(f)
    paths = sorted(
        os.path.join(albers_dir, p)
        for p in os.listdir(albers_dir)
        if p.endswith(".tif")
    )
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        for rec in pool.map(file_stats, paths, [grids] * len(paths)):
            rows.append(rec)
            print(
                "{}_{}  irrigated {:.0f} km2  {}".format(
                    rec["state"],
                    rec["year"],
                    rec["irrigated_km2"],
                    "ok" if rec["grid_ok"] else "GRID MISMATCH",
                ),
                flush=True,
            )
    rows.sort(key=lambda r: (r["state"], r["year"]))
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote {}".format(out_csv))
    bad = [r for r in rows if not r["grid_ok"]]
    print("{} files, {} grid/value mismatches".format(len(rows), len(bad)))
    return rows


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(
    deposit_dir=DEFAULT_DEPOSIT_DIR,
    deposition=None,
    sandbox=False,
    publish=False,
    token=None,
):
    """Create or resume a Zenodo deposition, set metadata, upload the deposit files.

    Uses the token in ``$ZENODO_TOKEN`` (``$ZENODO_SANDBOX_TOKEN`` with
    ``sandbox``) and ``deposit_dir/zenodo_metadata.json``. Uploads every
    ``.zip``, ``README.md`` and ``SHA256SUMS_*.txt`` in ``deposit_dir`` to the
    deposition's file bucket, skipping names already present with a matching
    MD5. Never publishes unless ``publish`` is set; prints the draft URL so
    the record can be reviewed in the browser first.
    """
    import requests

    env = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN"
    token = token or os.environ.get(env)
    if not token:
        raise RuntimeError(
            "set {} (a personal access token with deposit:write)".format(env)
        )
    api = ZENODO_API["sandbox" if sandbox else "zenodo"]
    auth = {"Authorization": "Bearer {}".format(token)}
    with open(os.path.join(deposit_dir, "zenodo_metadata.json")) as f:
        metadata = json.load(f)

    if deposition is None:
        r = requests.post(
            "{}/deposit/depositions".format(api), json={}, headers=auth, timeout=60
        )
        r.raise_for_status()
        deposition = r.json()["id"]
        print("created deposition {}".format(deposition))
    url = "{}/deposit/depositions/{}".format(api, deposition)
    r = requests.put(url, json=metadata, headers=auth, timeout=60)
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    present = {
        f["filename"]: f["checksum"].replace("md5:", "") for f in dep.get("files", [])
    }

    names = sorted(
        n
        for n in os.listdir(deposit_dir)
        if n.endswith(".zip") or n == "README.md" or n.startswith("SHA256SUMS_")
    )
    for name in names:
        path = os.path.join(deposit_dir, name)
        local = _md5(path)
        if present.get(name) == local:
            print("skip {} (already uploaded)".format(name))
            continue
        t0 = time.time()
        with open(path, "rb") as f:
            r = requests.put(
                "{}/{}".format(bucket, name), data=f, headers=auth, timeout=3600
            )
        r.raise_for_status()
        remote = r.json()["checksum"].replace("md5:", "")
        if remote != local:
            raise RuntimeError(
                "MD5 mismatch for {}: {} != {}".format(name, remote, local)
            )
        print(
            "{}  {:.0f} MB  {:.0f} s  md5 ok".format(
                name, os.path.getsize(path) / 1e6, time.time() - t0
            ),
            flush=True,
        )

    if publish:
        r = requests.post("{}/actions/publish".format(url), headers=auth, timeout=60)
        r.raise_for_status()
        print("published: {}".format(r.json().get("doi_url") or r.json().get("doi")))
    else:
        print("draft, not published: {}".format(dep["links"].get("html", url)))
    return deposition


def _load_manifest(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


_write_manifest = _write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="start exports and record them in the manifest")
    s.add_argument("--states", nargs="+", default=STATES)
    s.add_argument("--years", nargs="+", type=int, required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--collection", default=DEFAULT_COLLECTION)
    s.add_argument("--bucket", default=DEFAULT_BUCKET)
    s.add_argument("--prefix", default=DEFAULT_PREFIX)
    s.add_argument("--boundaries", default=DEFAULT_BOUNDARIES)
    s.add_argument(
        "--grid",
        choices=GRIDS,
        default="native",
        help="native: each asset's own grid (mixed EPSG:5070/4326); "
        "albers: one padded 30 m EPSG:5070 grid per state",
    )
    s.add_argument("--no-cog", action="store_true")
    s.add_argument("--project", default="ee-dgketchum")

    r = sub.add_parser("report", help="EECU usage of the manifest tasks")
    r.add_argument("--manifest", required=True)
    r.add_argument("--project", default="ee-dgketchum")

    f = sub.add_parser("fetch", help="mirror exported GeoTIFFs from GCS to disk")
    f.add_argument("--tif-dir", default=DEFAULT_TIF_DIR)
    f.add_argument("--bucket", default=DEFAULT_BUCKET)
    f.add_argument("--prefix", default=DEFAULT_PREFIX)
    f.add_argument("--project", default="ee-dgketchum")

    w = sub.add_parser(
        "warp", help="re-grid native exports onto per-state Albers grids"
    )
    w.add_argument("--manifest", required=True)
    w.add_argument("--tif-dir", default=DEFAULT_TIF_DIR)
    w.add_argument("--out-dir", default=DEFAULT_ALBERS_DIR)
    w.add_argument("--states", nargs="+", default=None)
    w.add_argument("--boundaries", default=DEFAULT_BOUNDARIES)
    w.add_argument("--overwrite", action="store_true")
    w.add_argument("--no-cutline", action="store_true")
    w.add_argument("--boundary-dir", default=DEFAULT_BOUNDARY_DIR)
    w.add_argument("--jobs", type=int, default=1, help="concurrent gdalwarp processes")
    w.add_argument("--project", default="ee-dgketchum")

    c = sub.add_parser("check", help="edge zero fractions and value set of GeoTIFFs")
    c.add_argument("paths", nargs="+")
    c.add_argument("-n", type=int, default=6, help="outer rows/columns to test")

    p = sub.add_parser("package", help="zip the Albers GeoTIFFs per state for Zenodo")
    p.add_argument("--albers-dir", default=DEFAULT_ALBERS_DIR)
    p.add_argument("--out-dir", default=DEFAULT_DEPOSIT_DIR)
    p.add_argument("--boundary-dir", default=DEFAULT_BOUNDARY_DIR)
    p.add_argument("--states", nargs="+", default=STATES)
    p.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("stats", help="pixel counts and grid check of every Albers file")
    t.add_argument("--albers-dir", default=DEFAULT_ALBERS_DIR)
    t.add_argument("--out-csv", default=None)
    t.add_argument("--jobs", type=int, default=8)

    u = sub.add_parser("upload", help="create/resume a Zenodo deposition and upload")
    u.add_argument("--deposit-dir", default=DEFAULT_DEPOSIT_DIR)
    u.add_argument("--deposition", type=int, default=None, help="resume this id")
    u.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    u.add_argument("--publish", action="store_true", help="publish after upload")

    x = sub.add_parser("cancel", help="cancel queued/running tasks in the manifest")
    x.add_argument("--manifest", required=True)
    x.add_argument("--project", default="ee-dgketchum")

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.bucket, args.prefix, args.tif_dir)
        return
    if args.cmd == "upload":
        upload(args.deposit_dir, args.deposition, args.sandbox, args.publish)
        return
    if args.cmd == "stats":
        stats(args.albers_dir, args.out_csv, args.jobs)
        return
    if args.cmd == "package":
        package(
            args.albers_dir,
            args.out_dir,
            args.boundary_dir,
            states=args.states,
            overwrite=args.overwrite,
        )
        return
    if args.cmd == "check":
        for path in args.paths:
            zero_frac, values = edge_check(path, args.n)
            print(os.path.basename(path), "values", values)
            for k, v in zero_frac.items():
                print("  {:<7} {}".format(k, " ".join("{:.2f}".format(f) for f in v)))
        return
    ee.Initialize(project=args.project)
    if args.cmd == "warp":
        warp(
            args.manifest,
            args.tif_dir,
            args.out_dir,
            states=args.states,
            boundaries=args.boundaries,
            overwrite=args.overwrite,
            cutline=not args.no_cutline,
            boundary_dir=args.boundary_dir,
            jobs=args.jobs,
        )
    elif args.cmd == "cancel":
        cancel(args.manifest)
    elif args.cmd == "submit":
        submit(
            args.states,
            args.years,
            args.manifest,
            collection=args.collection,
            bucket=args.bucket,
            prefix=args.prefix,
            boundaries=args.boundaries,
            cloud_optimized=not args.no_cog,
            grid=args.grid,
        )
    else:
        report(args.manifest)


if __name__ == "__main__":
    main()
