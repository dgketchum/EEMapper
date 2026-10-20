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
(edge zero fractions and value set, the QGIS reviewer's pass criterion) and
``cancel``. ``submit --grid albers`` does the re-gridding on EE instead; kept
as an option, not the default path.

The export region must be the asset's exact pixel rectangle (see
``native_grid``): the 2026-09-22 QGIS review found a 1-10 px frame of 0 around
every file exported with ``img.geometry()`` as the region.
"""

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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

    x = sub.add_parser("cancel", help="cancel queued/running tasks in the manifest")
    x.add_argument("--manifest", required=True)
    x.add_argument("--project", default="ee-dgketchum")

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.bucket, args.prefix, args.tif_dir)
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
