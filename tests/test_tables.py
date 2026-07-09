"""Phase 1 safety-net tests for map/tables.py.

Focus: concatenate_band_extract (the core per-year band-CSV concatenator /
class-remapper) plus the small pure helpers to_polygon and join_tables.

All fixtures are synthetic and written under pytest tmp_path -- no GCS, no
network, no real data paths.
"""

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from map.tables import concatenate_band_extract, to_polygon, join_tables


def _write_band_csv(path, rows):
    """Write a tiny CSV mimicking the real GCS band-extract export.

    The real files carry a 'system:index' and '.geo' column (which
    concatenate_band_extract drops), an integer POINT_TYPE and YEAR, plus
    float feature columns including the greenness metric nd_max_cy.
    """
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_rows():
    # Rows chosen to exercise every branch of the class-filtering logic.
    # Columns match a real band-extract row.
    return [
        # POINT_TYPE 0 with nd_max_cy >= 0.68 -> kept as 0
        {
            "system:index": "a",
            ".geo": "{}",
            "POINT_TYPE": 0,
            "YEAR": 2020,
            "nd_max_cy": 0.90,
            "B2": 0.1,
        },
        # POINT_TYPE 0 with nd_max_cy < 0.68 -> reclassed to 9 -> dropped
        {
            "system:index": "b",
            ".geo": "{}",
            "POINT_TYPE": 0,
            "YEAR": 2020,
            "nd_max_cy": 0.50,
            "B2": 0.2,
        },
        # POINT_TYPE 1 -> kept
        {
            "system:index": "c",
            ".geo": "{}",
            "POINT_TYPE": 1,
            "YEAR": 2020,
            "nd_max_cy": 0.50,
            "B2": 0.3,
        },
        # POINT_TYPE 4 (fallow) with nd_max_cy > 0.6 -> reclassed to 9 -> dropped
        {
            "system:index": "d",
            ".geo": "{}",
            "POINT_TYPE": 4,
            "YEAR": 2020,
            "nd_max_cy": 0.70,
            "B2": 0.4,
        },
        # POINT_TYPE 4 (fallow) with nd_max_cy <= 0.6 -> survives filtering
        {
            "system:index": "e",
            ".geo": "{}",
            "POINT_TYPE": 4,
            "YEAR": 2020,
            "nd_max_cy": 0.30,
            "B2": 0.5,
        },
        # POINT_TYPE 2 -> kept
        {
            "system:index": "f",
            ".geo": "{}",
            "POINT_TYPE": 2,
            "YEAR": 2020,
            "nd_max_cy": 0.50,
            "B2": 0.6,
        },
        # POINT_TYPE 3 -> kept
        {
            "system:index": "g",
            ".geo": "{}",
            "POINT_TYPE": 3,
            "YEAR": 2020,
            "nd_max_cy": 0.50,
            "B2": 0.7,
        },
    ]


def _run_base(tmp_path):
    root = tmp_path / "in"
    out = tmp_path / "out"
    root.mkdir()
    out.mkdir()
    rows = _base_rows()
    # split across two per-year files to exercise concatenation
    _write_band_csv(root / "bands_TEST_2020.csv", rows[:4])
    _write_band_csv(root / "bands_TEST_2021.csv", rows[4:])
    concatenate_band_extract(str(root), str(out), glob="bands_TEST")
    return pd.read_csv(out / "bands_TEST.csv")


def test_concatenate_drops_index_and_geo_columns(tmp_path):
    df = _run_base(tmp_path)
    # 'system:index' and '.geo' are always dropped
    assert "system:index" not in df.columns
    assert ".geo" not in df.columns
    # the surviving columns are exactly the feature + label columns
    assert set(df.columns) == {"POINT_TYPE", "YEAR", "nd_max_cy", "B2"}


def test_concatenate_row_count_after_class_filtering(tmp_path):
    df = _run_base(tmp_path)
    # of the 7 input rows, two are reclassed to 9 and dropped:
    #   - the POINT_TYPE 0 row with nd_max_cy < 0.68
    #   - the POINT_TYPE 4 row with nd_max_cy > 0.6
    # leaving 5 rows.
    assert df.shape[0] == 5
    # class 9 must never appear in the output
    assert 9 not in df["POINT_TYPE"].values


def test_concatenate_point_type_is_integer(tmp_path):
    df = _run_base(tmp_path)
    # POINT_TYPE and YEAR are forced to int; feature columns to float.
    assert np.issubdtype(df["POINT_TYPE"].dtype, np.integer)
    assert np.issubdtype(df["YEAR"].dtype, np.floating) or np.issubdtype(
        df["YEAR"].dtype, np.integer
    )
    assert np.issubdtype(df["nd_max_cy"].dtype, np.floating)


def test_concatenate_class_four_remapped_to_irrigated(tmp_path):
    df = _run_base(tmp_path)
    counts = df["POINT_TYPE"].value_counts().to_dict()
    # Invariant regardless of pandas version: classes {1, 4} together account
    # for exactly the two surviving irrigated/fallow rows (one POINT_TYPE 1
    # input row + one surviving POINT_TYPE 4 row).
    assert counts.get(1, 0) + counts.get(4, 0) == 2

    # tables.py does `df['POINT_TYPE'][df['POINT_TYPE'] == 4] = 1`, a chained
    # assignment that remaps fallow(4) -> irrigated(1). This only works on
    # pandas < 3 (Copy-on-Write makes it a silent no-op on 3.x), which is why
    # pyproject.toml pins pandas>=2,<3. If this test fails with class 4
    # surviving, the pin was lifted without rewriting the remap.
    assert counts.get(4, 0) == 0
    assert counts.get(1, 0) == 2
    # the classes present are exactly the four survivors after remap
    assert set(counts) == {0, 1, 2, 3}


def test_concatenate_output_written_to_glob_named_file(tmp_path):
    root = tmp_path / "in"
    out = tmp_path / "out"
    root.mkdir()
    out.mkdir()
    _write_band_csv(root / "bands_TEST_2020.csv", _base_rows())
    concatenate_band_extract(str(root), str(out), glob="bands_TEST")
    # with no sample/select, the output file is named '<glob>.csv'
    assert (out / "bands_TEST.csv").exists()


# --------------------------------------------------------------------------- #
# to_polygon
# --------------------------------------------------------------------------- #
def test_to_polygon_non_list_returns_nan():
    # anything that is not a list short-circuits to nan
    assert to_polygon("not a list") != to_polygon("not a list")  # nan != nan
    result = to_polygon(5)
    assert isinstance(result, float) and result != result


def test_to_polygon_valid_ring_builds_polygon():
    poly = to_polygon([[(0, 0), (1, 0), (1, 1), (0, 1)]])
    assert isinstance(poly, Polygon)
    assert poly.area == 1.0


# --------------------------------------------------------------------------- #
# join_tables
# --------------------------------------------------------------------------- #
def test_join_tables_concatenates_rows(tmp_path):
    one = tmp_path / "one.csv"
    two = tmp_path / "two.csv"
    out = tmp_path / "joined.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(one, index=False)
    pd.DataFrame({"a": [7, 8], "b": [9, 10]}).to_csv(two, index=False)

    join_tables(str(one), str(two), str(out))

    res = pd.read_csv(out)
    # rows are concatenated (3 + 2 == 5)
    assert res.shape[0] == 5
    # join_tables writes with the default (numeric) index, which reads back as
    # an 'Unnamed: 0' column alongside the original 'a' and 'b'.
    assert "a" in res.columns and "b" in res.columns
    assert list(res["a"]) == [1, 2, 3, 7, 8]
