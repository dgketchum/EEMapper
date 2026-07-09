"""Phase 1 safety-net tests for map/shape_ops.py.

Covers the pure vector utilities that do NOT shell out to ogr2ogr:
    - get_area (via captured stdout, since it prints rather than returns)
    - popper_test (compactness filter)
    - fiona_merge (centroid latitude filter + FID/SOURCE schema)

The ogr2ogr wrappers (to_aea / to_geographic) and the disk-path __main__ block
are intentionally skipped -- they require the /usr/bin/ogr2ogr subprocess and
absolute data paths.

All fixtures are synthetic shapefiles written under pytest tmp_path.
"""

import os

import fiona
from shapely.geometry import Polygon, MultiPolygon, mapping

# Importing shape_ops has an import-time side effect: it sets
# os.environ['GDAL_DATA'] to a bogus relative miniconda path (line 41), which
# can break fiona's ability to write .prj files. Import it, then strip that
# bogus value so fiona falls back to its wheel-bundled GDAL data.
import map.shape_ops as so

if os.environ.get("GDAL_DATA") == "miniconda3/envs/gcs/share/gdal/":
    os.environ.pop("GDAL_DATA")


CRS = fiona.crs.from_epsg(5070)  # a projected (meter) CRS so areas are meaningful
SQUARE_1KM = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])  # area = 1e6 m^2


def _write_shp(path, geoms, geom_type="Polygon", crs=CRS, props_schema=None):
    props_schema = props_schema or {"id": "int"}
    schema = {"geometry": geom_type, "properties": props_schema}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        for i, g in enumerate(geoms):
            dst.write({"geometry": mapping(g), "properties": {"id": i}})


# --------------------------------------------------------------------------- #
# get_area
# --------------------------------------------------------------------------- #
def test_get_area_polygon_only_prints_zero_without_intersect(tmp_path, capsys):
    # LOCKED-IN QUIRK: for plain Polygon features, get_area only accumulates
    # area inside the `if intersect_shape:` branch. With no intersect_shape the
    # running total stays 0.0, so a Polygon-only shapefile prints '0.0'.
    shp = str(tmp_path / "squares.shp")
    _write_shp(shp, [SQUARE_1KM], geom_type="Polygon")
    ret = so.get_area(shp)
    assert ret is None  # get_area returns nothing; it only prints
    assert capsys.readouterr().out.strip() == "0.0"


def test_get_area_multipolygon_accumulates_without_intersect(tmp_path, capsys):
    # For MultiPolygon features the area IS accumulated even without an
    # intersect_shape. Two disjoint 1 km^2 parts -> (1e6 + 1e6) / 1e6 = 2.0.
    part2 = Polygon(
        [(5000, 5000), (6000, 5000), (6000, 6000), (5000, 6000)]
    )  # also 1e6 m^2
    mp = MultiPolygon([SQUARE_1KM, part2])
    shp = str(tmp_path / "multi.shp")
    _write_shp(shp, [mp], geom_type="MultiPolygon")
    so.get_area(shp)
    assert capsys.readouterr().out.strip() == "2.0"


# --------------------------------------------------------------------------- #
# popper_test (compactness filter)
# --------------------------------------------------------------------------- #
def test_popper_test_keeps_square(tmp_path):
    # popper = 4*pi*area / perimeter^2. For a square this is pi/4 ~= 0.7854,
    # which falls inside the default open interval (min_thresh=0.78, 0.79).
    unit_square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    src = str(tmp_path / "square.shp")
    out = str(tmp_path / "square_out.shp")
    _write_shp(src, [unit_square], geom_type="Polygon")

    so.popper_test(src, out)

    with fiona.open(out) as f:
        feats = list(f)
    assert len(feats) == 1
    assert abs(feats[0]["properties"]["popper"] - 0.7853981633974483) < 1e-9


def test_popper_test_excludes_elongated_rectangle(tmp_path):
    # A 10x1 rectangle has popper ~= 0.26, below min_thresh=0.78 -> excluded.
    rect = Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
    src = str(tmp_path / "rect.shp")
    out = str(tmp_path / "rect_out.shp")
    _write_shp(src, [rect], geom_type="Polygon")

    so.popper_test(src, out)

    with fiona.open(out) as f:
        assert len(list(f)) == 0


# --------------------------------------------------------------------------- #
# fiona_merge (centroid latitude filter + FID/SOURCE schema)
# --------------------------------------------------------------------------- #
def test_fiona_merge_filters_high_latitude_and_sets_source(tmp_path):
    # fiona_merge drops any feature whose centroid |y| > 50.0 and rewrites the
    # schema to just FID + SOURCE (SOURCE = first 10 chars of the file stem).
    crs = fiona.crs.from_epsg(4326)
    near_equator = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # centroid y=0.5 kept
    high_lat = Polygon([(0, 60), (1, 60), (1, 61), (0, 61)])  # centroid y=60.5 dropped

    f1 = str(tmp_path / "aaa.shp")
    f2 = str(tmp_path / "bbb.shp")
    _write_shp(f1, [near_equator, high_lat], crs=crs)
    _write_shp(f2, [near_equator], crs=crs)

    out = str(tmp_path / "merged.shp")
    so.fiona_merge(out, [f1, f2])

    with fiona.open(out) as f:
        feats = list(f)
    # 2 kept from f1's 1 valid + f2's 1 valid (high_lat dropped)
    assert len(feats) == 2
    for feat in feats:
        assert set(feat["properties"].keys()) == {"FID", "SOURCE"}
    assert {feat["properties"]["SOURCE"] for feat in feats} == {"aaa", "bbb"}
