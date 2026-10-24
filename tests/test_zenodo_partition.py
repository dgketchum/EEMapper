"""``partition_one``: exact partition of the lattice by the TIGER state-ID raster."""

import json
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from irrmapper.postproc.zenodo import (  # noqa: E402
    NODATA,
    partition,
    partition_one,
    state_id_raster,
)

# two 'states' side by side on a 30 m lattice (edges at 15 mod 30): A owns
# x in [15, 315), B owns x in [315, 615); both span y in [15, 315)
STATES = ["AA", "BB"]
NEIGHBOURS = [("AA", "BB")]
POLY = {"AA": box(15, 15, 315, 315), "BB": box(315, 15, 615, 315)}


def _write(path, arr, x0, y1):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="uint8",
        crs="EPSG:5070",
        transform=from_origin(x0, y1, 30, 30),
        nodata=NODATA,
    ) as ds:
        ds.write(arr, 1)


@pytest.fixture
def raw(tmp_path):
    """Raw per-state warps with a 1 px margin that spill across the line.

    AA's raw covers x [-15, 375): its own 10 columns, one margin column left,
    and two columns past the line (the generalised outline spilled). BB's raw
    covers x [255, 645): two columns short of the line on its west side, i.e.
    BB's export did not cover BB's own columns x [315, 375) in row 0; there
    the neighbour's (AA's) spill holds the value.
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    bdir = tmp_path / "bounds"
    raw_dir.mkdir()
    bdir.mkdir()
    for st, g in POLY.items():
        gpd.GeoDataFrame({"STUSPS": [st], "geometry": [g]}, crs="EPSG:5070").to_file(
            bdir / f"{st}.fgb", driver="FlatGeobuf", engine="fiona"
        )
    # AA raw: 11 rows x 13 cols, origin x=-15, y1=315; rows 0-9 have their
    # centres inside the polygons, row 10 (centre y=0) is the margin
    a = np.zeros((11, 13), dtype="uint8")
    a[:, 0] = NODATA  # margin column outside the state
    a[-1, :] = NODATA  # bottom margin row
    a[:, 11:13] = 1  # AA's spill over BB's first two columns: irrigated
    a[3, 5] = 1  # one irrigated pixel inside AA
    _write(raw_dir / "IrrMapper_v1_2_AA_2021.tif", a, -15, 315)
    # BB raw: 11 rows x 13 cols, origin x=255; cols 0-1 are AA's territory
    # (BB spilled west), cols 2-3 (x 315-375) nodata: BB's export fell short
    b = np.zeros((11, 13), dtype="uint8")
    b[:, 0:2] = 0  # BB's spill into AA, not irrigated
    b[:, 2:4] = NODATA  # hole inside BB
    b[-1, :] = NODATA
    b[5, 7] = 1
    _write(raw_dir / "IrrMapper_v1_2_BB_2021.tif", b, 255, 315)
    (raw_dir / "grids.json").write_text(json.dumps({"AA": {}, "BB": {}}))
    return raw_dir, out_dir, bdir


def test_state_id_raster_partitions_shared_edge():
    ids = state_id_raster(POLY, from_origin(-15, 315, 30, 30), (11, 22), STATES)
    assert set(np.unique(ids)) == {0, 1, 2}
    # column 0 (x -15..15) and row 10 (centre y=0) are outside both
    assert (ids[:10, 0] == 0).all()
    assert (ids[10, :] == 0).all()
    assert (ids[:10, 1:11] == 1).all()
    assert (ids[:10, 11:21] == 2).all()
    assert (ids[:10, 21] == 0).all()
    # a centre on the shared edge is claimed exactly once
    assert ((ids == 1).sum() + (ids == 2).sum()) == (ids > 0).sum()


def test_partition_one_fills_from_neighbour_and_cuts_spill(raw):
    raw_dir, out_dir, bdir = raw
    out_dir.mkdir()
    desc, n_in, n_hole, filled, left = partition_one(
        "IrrMapper_v1_2_BB_2021",
        str(raw_dir),
        str(out_dir),
        str(bdir),
        NEIGHBOURS,
        STATES,
    )
    assert desc == "IrrMapper_v1_2_BB_2021"
    assert n_in == 100  # 10 x 10 pixels of BB inside the raw window
    assert n_hole == 20  # cols x 315-375 over BB's 10 rows
    assert filled == 20  # all of them covered by AA's spill
    assert left == 0
    with rasterio.open(out_dir / "IrrMapper_v1_2_BB_2021.tif") as ds:
        out = ds.read(1)
        assert ds.nodata == NODATA
        assert ds.driver == "GTiff"
    assert (out[:, 0:2] == NODATA).all()  # BB's spill into AA removed
    assert (out[:10, 2:4] == 1).all()  # hole filled with AA's irrigated spill
    assert out[5, 7] == 1
    assert (out[10, :] == NODATA).all()
    assert (out[:, 12] == NODATA).all()


def test_partition_is_exact(raw):
    raw_dir, out_dir, bdir = raw
    report = partition(
        str(raw_dir),
        str(out_dir),
        str(bdir),
        jobs=1,
        neighbours=NEIGHBOURS,
        all_states=STATES,
    )
    assert set(report) == {"IrrMapper_v1_2_AA_2021", "IrrMapper_v1_2_BB_2021"}
    assert os.path.exists(out_dir / "grids.json")
    with rasterio.open(out_dir / "IrrMapper_v1_2_AA_2021.tif") as da:
        a = da.read(1)
    with rasterio.open(out_dir / "IrrMapper_v1_2_BB_2021.tif") as db:
        b = db.read(1)
    # the shared window is AA's cols 9-12 / BB's cols 0-3 (x 255-375)
    va = a[:, 9:13] != NODATA
    vb = b[:, 0:4] != NODATA
    assert not (va & vb).any(), "a pixel carries data in both states"
    assert (va | vb)[:10].all(), "a pixel inside the union is nodata"
    assert (a[:10, 11:13] == NODATA).all()  # AA's spill cut away
    assert int((a == 1).sum()) == 1  # only AA's own irrigated pixel remains
