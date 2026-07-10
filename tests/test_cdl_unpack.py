"""Regression: reduce_classification must unpack get_cdl's 3-tuple.

get_cdl returns (cultivated, crop, simple_crop); reduce_classification
unpacked two values, so any cdl_mask=True run raised ValueError before it
touched Earth Engine. These tests run the cdl_mask branches with the EE
namespace mocked.
"""

from unittest import mock

from map.call_ee import reduce_classification


def _cdl_triple():
    return (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())


def test_reduce_classification_cdl_mask_unpacks_get_cdl():
    """cdl_mask=True (no min_years) reaches export without a ValueError."""
    with (
        mock.patch("map.call_ee.ee") as mock_ee,
        mock.patch("map.call_ee.get_cdl") as mock_cdl,
    ):
        mock_cdl.return_value = _cdl_triple()
        reduce_classification(
            "users/x/coll",
            "users/x/shapes",
            years=[2015],
            description="test",
            cdl_mask=True,
            props=["GEOID"],
        )
        mock_cdl.assert_called_once_with(2015)
        cultivated = mock_cdl.return_value[0]
        cultivated.eq.assert_called_once_with(1)
        assert mock_ee.batch.Export.table.toCloudStorage.called


def test_reduce_classification_cdl_mask_with_min_years():
    """The cdl_mask + min_years branch unpacks the 3-tuple too."""
    with (
        mock.patch("map.call_ee.ee") as mock_ee,
        mock.patch("map.call_ee.get_cdl") as mock_cdl,
    ):
        mock_cdl.return_value = _cdl_triple()
        reduce_classification(
            "users/x/coll",
            "users/x/shapes",
            years=[2015],
            description="test",
            cdl_mask=True,
            min_years=3,
            props=["GEOID"],
        )
        mock_cdl.assert_called_once_with(2015)
        assert mock_ee.batch.Export.table.toCloudStorage.called
