"""Phase 1 safety-net tests for map/distribute_points.py.

Covers the parts of PointsRunspec that are testable without real training
data: construction with no vector paths, the pure static _random_points
sampler, and _get_polygons reading/filtering synthetic polygon shapefiles.

NOT testable in this environment (documented, not forced):
    create_sample_points / save_sample_points end-to-end need realistic
    labeled polygon inputs per class; only the component pieces are covered.

All fixtures are synthetic shapefiles under pytest tmp_path.
"""

import os

import numpy as np
import fiona
from shapely.geometry import Polygon, mapping

# Importing distribute_points sets os.environ['GDAL_DATA'] to a bogus relative
# miniconda path at import time (line 17), which can break fiona .prj writing.
# Import, then strip the bogus value so fiona uses its bundled GDAL data.
import map.distribute_points as dp

if os.environ.get("GDAL_DATA") == "miniconda3/envs/gcs/share/gdal/":
    os.environ.pop("GDAL_DATA")


CRS = fiona.crs.from_epsg(5070)


def _write_polys(path, polys, props=None, prop_schema=None):
    prop_schema = prop_schema or {"id": "int"}
    schema = {"geometry": "Polygon", "properties": prop_schema}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=CRS, schema=schema) as dst:
        for i, p in enumerate(polys):
            record_props = props[i] if props else {"id": i}
            dst.write({"geometry": mapping(p), "properties": record_props})


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_construct_with_no_paths_does_not_sample():
    # With no *_path / intersect / exclude kwargs and none of the class-count
    # flags set, __init__ opens no vector files and generates no points.
    prs = dp.PointsRunspec(buffer=0)
    assert prs.intersect is None
    assert prs.exclude is None
    assert prs.object_id == 0
    # extracted_points starts empty with the fixed schema columns
    assert list(prs.extracted_points.columns) == [
        "FID",
        "X",
        "Y",
        "POINT_TYPE",
        "YEAR",
    ]
    assert len(prs.extracted_points) == 0


# --------------------------------------------------------------------------- #
# _random_points (pure static method)
# --------------------------------------------------------------------------- #
def test_random_points_shape_and_bounds():
    np.random.seed(0)
    # coords are (min_x, min_y, max_x, max_y); n controls resolution.
    x_range, y_range = dp.PointsRunspec._random_points((0, 0, 10, 20), 5)
    # each axis gets linspace(min, max, num=2*n) then shuffled -> length 2*n
    assert len(x_range) == 10
    assert len(y_range) == 10
    # linspace preserves the endpoints regardless of the shuffle
    assert x_range.min() == 0 and x_range.max() == 10
    assert y_range.min() == 0 and y_range.max() == 20
    # values are exactly the linspace set (shuffle only reorders)
    assert set(x_range) == set(np.linspace(0, 10, num=10))


# --------------------------------------------------------------------------- #
# _get_polygons
# --------------------------------------------------------------------------- #
def test_get_polygons_returns_all_shapes(tmp_path):
    prs = dp.PointsRunspec(buffer=0)
    shp = str(tmp_path / "polys.shp")
    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
    ]
    _write_polys(shp, polys)

    got = prs._get_polygons(shp)
    assert len(got) == 2
    assert all(isinstance(g, Polygon) for g in got)


def test_get_polygons_with_attribute_returns_pairs(tmp_path):
    prs = dp.PointsRunspec(buffer=0)
    shp = str(tmp_path / "polys_attr.shp")
    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
    ]
    _write_polys(
        shp,
        polys,
        props=[{"YEAR": 2000}, {"YEAR": 2001}],
        prop_schema={"YEAR": "int"},
    )

    got = prs._get_polygons(shp, attr="YEAR")
    # each entry is a (geometry, attribute_value) pair
    assert [(type(g).__name__, y) for g, y in got] == [
        ("Polygon", 2000),
        ("Polygon", 2001),
    ]


def test_get_polygons_intersect_filter(tmp_path):
    prs = dp.PointsRunspec(buffer=0)
    shp = str(tmp_path / "polys2.shp")
    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # overlaps the intersect
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),  # does not
    ]
    _write_polys(shp, polys)

    inter = str(tmp_path / "inter.shp")
    _write_polys(inter, [Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])])

    # set the intersect filter directly (as __init__ would from kwargs)
    prs.intersect = inter
    prs.intersect_buffer = None
    got = prs._get_polygons(shp)
    assert len(got) == 1


# --------------------------------------------------------------------------- #
# _add_entry
# --------------------------------------------------------------------------- #
def test_add_entry_accumulates_points():
    prs = dp.PointsRunspec(buffer=0)
    prs.year = 2000
    prs._add_entry((1.0, 2.0), val=0)
    prs._add_entry((3.0, 4.0), val=2)
    df = prs.extracted_points
    assert df.shape[0] == 2
    assert list(df.columns) == ["FID", "X", "Y", "POINT_TYPE", "YEAR"]
    assert df.iloc[0].to_dict() == {
        "FID": 0,
        "X": 1.0,
        "Y": 2.0,
        "POINT_TYPE": 0,
        "YEAR": 2000,
    }
    assert df.iloc[1]["FID"] == 1
    assert df.iloc[1]["POINT_TYPE"] == 2
    assert prs.object_id == 2
