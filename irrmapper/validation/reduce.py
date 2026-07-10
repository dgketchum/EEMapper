"""Zonal reduction of classifications for comparison against county/HUC/state
statistics (e.g. NASS), with optional CDL cultivated masking.
"""

import ee

from irrmapper.ingest.cdl import get_cdl


def reduce_classification(asset, shapes, years=None, description=None, cdl_mask=False,
                          min_years=0, props=None, acres=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS bucket.
    :param shapes: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    sum_mask = None
    fc = ee.FeatureCollection(shapes)
    # fc = fc.filterMetadata('GEOID', 'equals', '30073')
    if not props:
        props = list(fc.first().propertyNames().getInfo())

    if min_years > 0:
        coll = ee.ImageCollection(asset)
        sum = ee.ImageCollection(coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])).sum()
        sum_mask = sum.lt(min_years)

    first, input_bands = True, None
    for yr in years:
        coll = ee.ImageCollection(asset).filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr))
        tot = coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])

        if cdl_mask and min_years > 0:
            cultivated, _, _ = get_cdl(yr)
            cdl_crop_mask = cultivated.eq(1)
            tot = tot.mask(cdl_crop_mask).mask(sum_mask)

        elif min_years > 0:
            tot = tot.mask(sum_mask)

        elif cdl_mask:
            cultivated, _, _ = get_cdl(yr)
            cdl_crop_mask = cultivated.eq(1)
            tot = tot.mask(cdl_crop_mask)

        _rname = 'IM_{}'.format(yr)
        props.append(_rname)
        tot = tot.multiply(ee.Image.pixelArea())

        if acres:
            tot = tot.divide(4046.86)

        if first:
            input_bands = tot.rename(_rname)
            first = False
        else:
            tot = tot.rename(_rname)
            input_bands = input_bands.addBands([tot])

    data = input_bands.reduceRegions(collection=fc,
                                     reducer=ee.Reducer.sum(),
                                     scale=30)

    if len(years) == 1:
        description = '{}_{}'.format(description, years[0])
        props[-1] = 'sum'
    if len(years) > 1:
        description = '{}_{}_{}'.format(description, years[0], years[-1])

    task = ee.batch.Export.table.toCloudStorage(
        data,
        description=description,
        bucket='wudr',
        fileNamePrefix=description,
        fileFormat='CSV',
        selectors=props)
    task.start()
    print(description)
