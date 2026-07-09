"""Phase 1 safety-net tests for map/models.py.

Covers the pure confusion-matrix helpers (producer/consumer) with exact
values, plus structural / smoke tests for the scikit-learn Random Forest
helpers using small synthetic DataFrames.

The RandomForestClassifier calls in models.py do NOT pass random_state, so the
importance *values* are not reproducible across runs; we therefore assert only
structure (types, ordering, feature set) and seed numpy for speed/stability.
All fixtures are synthetic under tmp_path -- no Earth Engine, no real data.
"""

import numpy as np
import pandas as pd

from map.models import consumer, producer, random_forest, find_rf_variable_importance


# --------------------------------------------------------------------------- #
# confusion-matrix helpers (pure)
# --------------------------------------------------------------------------- #
def test_consumer_is_row_normalized_diagonal():
    # consumer accuracy = diagonal / row sum
    cf = np.array([[8, 2], [1, 9]])
    result = consumer(cf)
    assert result[0] == 8 / 10
    assert result[1] == 9 / 10


def test_producer_is_column_normalized_diagonal():
    # producer accuracy = diagonal / column sum
    cf = np.array([[8, 2], [1, 9]])
    result = producer(cf)
    assert result[0] == 8 / 9
    assert result[1] == 9 / 11


def test_consumer_producer_length_matches_matrix():
    cf = np.array([[5, 1, 0], [2, 6, 1], [0, 1, 7]])
    assert len(consumer(cf)) == 3
    assert len(producer(cf)) == 3


# --------------------------------------------------------------------------- #
# helpers to build a synthetic training frame
# --------------------------------------------------------------------------- #
def _synthetic_frame(n=80, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            # random_forest computes a geometry from lon/lat (or Lon_GCS/LAT_GCS);
            # the columns must exist or the function raises KeyError.
            "lon": rng.uniform(-116, -112, n),
            "lat": rng.uniform(44, 48, n),
            "B2": rng.random(n),
            "B3": rng.random(n),
            "nd_max_cy": rng.random(n),
            "YEAR": np.full(n, 2020),
            "POINT_TYPE": rng.integers(0, 2, n),
        }
    )


# --------------------------------------------------------------------------- #
# random_forest (smoke)
# --------------------------------------------------------------------------- #
def test_random_forest_accepts_dataframe_and_returns_none():
    np.random.seed(0)
    df = _synthetic_frame()
    # random_forest fits, prints a confusion matrix, and explicitly returns None.
    result = random_forest(df.copy())
    assert result is None


# --------------------------------------------------------------------------- #
# find_rf_variable_importance (structure)
# --------------------------------------------------------------------------- #
def test_find_rf_variable_importance_structure(tmp_path):
    np.random.seed(0)
    csv = tmp_path / "bands.csv"
    _synthetic_frame().to_csv(csv, index=False)

    master = find_rf_variable_importance(str(csv))

    # returns a list of (feature_name, cumulative_importance) pairs
    assert isinstance(master, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in master)
    assert all(isinstance(f, str) for f, _ in master)
    assert all(isinstance(float(v), float) for _, v in master)

    # one entry per feature column (YEAR and POINT_TYPE are dropped before fit)
    feature_names = {f for f, _ in master}
    assert feature_names == {"lon", "lat", "B2", "B3", "nd_max_cy"}

    # importances are summed over 10 iterations and sorted descending
    values = [v for _, v in master]
    assert values == sorted(values, reverse=True)
