"""Dissertation-era Earth Engine analysis functions quarantined from
map/call_ee.py during the Phase 3 restructure (2026-07). Not part of the
irrmapper package; kept runnable as scripts against the installed package.
See notes/phase3_package_restructure.md.
"""

import os
from pprint import pprint

import ee

from irrmapper.auth import is_authorized
from irrmapper.features.stack import stack_bands
from irrmapper.ingest.landsat import landsat_masked
from irrmapper.postproc.rasters import export_raster

BOUNDARIES = 'users/dgketchum/boundaries'
IRRIGATION_TABLE = 'users/dgketchum/western_states_irr/NV_agpoly'
RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
E_STATES = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

# list of years we have verified irrigated fields
YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019]

TEST_YEARS = [2005]
ALL_YEARS = [x for x in range(1986, 2021)]


def attribute_irrigation(collection):
    """
    Extracts fraction of vector classified as irrigated. Been using this to attribute irrigation to
    field polygon coverages.
    :return:
    """
    fc = ee.FeatureCollection(IRRIGATION_TABLE)
    for state in TARGET_STATES:
        for yr in range(2011, 2021):
            images = os.path.join(collection, '{}_{}'.format(state, yr))
            coll = ee.Image(images)
            tot = coll.select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])
            means = tot.reduceRegions(collection=fc,
                                      reducer=ee.Reducer.mean(),
                                      scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                means,
                description='{}_{}'.format(state, yr),
                bucket='wudr',
                fileNamePrefix='attr_{}_{}'.format(state, yr),
                fileFormat='CSV')

            print(state, yr)
            task.start()


def get_ndvi_cultivation_data_polygons(table, years, region, input_props,
                                       bucket='wudr', southern=False, id_col='FID'):
    """
    Extracts specified data to a polygon shapefile layer.
    :return:
    """
    fc = ee.FeatureCollection(table)
    roi = ee.FeatureCollection(region)
    props = [id_col]
    input_bands = None
    first = True
    for year in years:
        rname_ = ['{}_{}'.format(x, year) for x in input_props]
        [props.append(rn) for rn in rname_]
        if first:
            input_bands = stack_bands(year, roi, southern).select(input_props)
            input_bands = input_bands.rename(rname_)
            first = False
        else:
            add_bands_ = stack_bands(year, roi, southern).select(input_props)
            add_bands_ = add_bands_.rename(rname_)
            input_bands = input_bands.addBands(add_bands_)

    means = input_bands.reduceRegions(collection=fc,
                                      reducer=ee.Reducer.mean(),
                                      scale=30)

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description='{}'.format(os.path.basename(table)),
        bucket=bucket,
        fileNamePrefix='attr_{}_{}'.format(os.path.basename(table), years[0]),
        fileFormat='CSV',
        selectors=props)

    print(os.path.basename(table))
    task.start()


def wrs_analysis(irrmapper, table, desc, bucket, debug=False):
    irr_coll = ee.ImageCollection(irrmapper)

    irr_coll = irr_coll.map(lambda img: img.lt(1))
    img = irr_coll.sum()

    fc = ee.FeatureCollection(table)
    # fc = fc.filterMetadata('FID', 'equals', 253)
    data = img.reduceRegions(collection=fc,
                             reducer=ee.Reducer.mode(),
                             scale=30)

    if debug:
        p = data.first().getInfo()['properties']
        pprint('propeteries {}'.format(p))

    task = ee.batch.Export.table.toCloudStorage(
        data,
        description=desc,
        bucket=bucket,
        fileNamePrefix=desc,
        fileFormat='CSV')
    task.start()
    print(desc)


def filter_irrigated(asset, yr, region, filter_type='irrigated', addl_yr=None):
    """
    Takes a field polygon vector and filters it based on NDVI rules.

    :filter_type: filter_low is to filter out low ndvi fields (thus saving and returning high-ndvi fields,
            likely irrigated), filter_high filters out high-ndvi feilds, leaving likely fallowed fields
    :return:
    """
    filt_fc = None

    # filter out any weird geometries
    plots = ee.FeatureCollection(asset)
    plots = plots.map(lambda x: x.set('geo_type', x.geometry().type()))
    plots = plots.filter(ee.Filter.eq('geo_type', 'Polygon'))

    roi = ee.FeatureCollection(region)
    if filter_type == 'irrigated':

        summer_s, late_summer_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
        late_summer_s_, summer_e = '{}-07-01'.format(yr), '{}-10-31'.format(yr)

        lsSR_masked = landsat_masked(yr, roi)

        early_nd = ee.Image(lsSR_masked.filterDate(summer_s, late_summer_e).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd')
        early_nd_max = early_nd.select('nd').reduce(ee.Reducer.intervalMean(0.0, 15.0))
        early_int_mean = early_nd_max.reduceRegions(collection=plots,
                                                    reducer=ee.Reducer.median(),
                                                    scale=30.0)
        early_int_mean = early_int_mean.select('median')

        late_nd = ee.Image(lsSR_masked.filterDate(late_summer_s_, summer_e).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_1')
        late_nd_max = late_nd.select('nd_1').reduce(ee.Reducer.intervalMean(0.0, 15.0))

        combo = late_nd_max.reduceRegions(collection=early_int_mean,
                                          reducer=ee.Reducer.mean(),
                                          scale=30.0)

        filt_fc = combo  # .filter(ee.Filter.Or(ee.Filter.gt('median', 0.9), ee.Filter.gt('mean', 0.8)))
        desc = '{}_{}_irr'.format(os.path.basename(region), yr)

    elif filter_type == 'dryland':

        summer_s, late_summer_e = '{}-07-01'.format(yr), '{}-10-31'.format(yr)
        late_summer_s_, late_summer_e_ = '{}-07-01'.format(addl_yr), '{}-10-31'.format(addl_yr)

        lsSR_masked = landsat_masked(yr, roi)
        early_nd = ee.Image(lsSR_masked.filterDate(summer_s, late_summer_e).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd')
        early_nd_max = early_nd.select('nd').reduce(ee.Reducer.intervalMean(0.0, 15.0))
        early_int_mean = early_nd_max.reduceRegions(collection=plots,
                                                    reducer=ee.Reducer.mean(),
                                                    scale=30.0)
        early_int_mean = early_int_mean.select(['mean', 'MGRS_TILE', 'system:index', 'popper'],
                                               ['nd_e', 'MGRS_TILE', 'system:index', 'popper'])

        lsSR_masked = landsat_masked(addl_yr, roi)
        late_nd = ee.Image(lsSR_masked.filterDate(late_summer_s_, late_summer_e_).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_1')
        late_nd_max = late_nd.select('nd_1').reduce(ee.Reducer.intervalMean(0.0, 15.0))

        combo = late_nd_max.reduceRegions(collection=early_int_mean,
                                          reducer=ee.Reducer.mean(),
                                          scale=30.0)

        filt_fc = combo.filter(ee.Filter.Or(ee.Filter.lt('nd_e', 0.7), ee.Filter.lt('mean', 0.7)))
        desc = '{}_dry'.format(os.path.basename(region))

    else:
        raise NotImplementedError('must choose from filter_low or filter_high')

    task = ee.batch.Export.table.toCloudStorage(filt_fc,
                                                description=desc,
                                                bucket='wudr',
                                                fileFormat='SHP')
    print(yr, filter_type)
    task.start()


def export_resmaple_irr_frequency():
    bounds = 'users/dgketchum/boundaries/impacts_basins_join'
    im = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
    remap = ee.ImageCollection(im) \
        .filterDate('1991-01-01', '2020-12-31') \
        .map(lambda x: x.select('classification')
             .remap([0, 1, 2, 3], [1, 0, 0, 0]))
    sum_ = remap.sum().rename('sum')

    roi = ee.FeatureCollection(os.path.join(bounds)).geometry()
    i = sum_.clip(roi).int()

    task = ee.batch.Export.image.toCloudStorage(
        image=i,
        description='{}'.format('irr_freq_1991_2020'),
        bucket='wudr',
        fileNamePrefix='{}'.format('irr_freq_1991_2020'),
        scale=30,
        maxPixels=1e13)
    task.start()
    print(bounds)


def export_resmaple_irr_change():
    bounds = 'users/dgketchum/boundaries/western_17_counties'
    bounds = ee.FeatureCollection(bounds)  # .filterMetadata('GEOID', 'equals', '16001')
    roi = bounds.geometry()

    im = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp')

    remap = ee.ImageCollection(im) \
        .filterDate('1991-01-01', '2020-12-31') \
        .map(lambda x: x.select('classification')
             .remap([0, 1, 2, 3], [1, 0, 0, 0]))
    sum_ = remap.sum().rename('sum')
    mask = sum_.gt(5)

    early_coll = im.filterDate('1991-01-01', '1995-12-31').select('classification')
    early_rm = early_coll.map(lambda x: x.lt(1))
    early_sum = early_rm.sum().gte(2)

    late_coll = im.filterDate('2016-01-01', '2020-12-31').select('classification')
    late_rm = late_coll.map(lambda x: x.lt(1))
    late_sum = late_rm.sum().gte(2)

    difference = late_sum.subtract(early_sum)
    difference = difference.mask(mask).add(2)

    # proj_ = ee.Projection('EPSG:4326')
    # difference = difference.reproject(proj_)

    i = difference.clip(roi).int()

    task = ee.batch.Export.image.toCloudStorage(
        image=i,
        description='{}'.format('delta_irr_1990_2020'),
        bucket='wudr',
        fileNamePrefix='{}'.format('delta_irr_1990_2020'),
        scale=30,
        maxPixels=1e13)

    task.start()


if __name__ == '__main__':
    is_authorized(project='ee-dgketchum')
    out_c = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
    years_ = list(range(2020, 2026))

    geo_ = 'users/dgketchum/boundaries/ID'
    export_raster(out_c, geo_, years=years_, min_years=3, debug=False, state='ID',
                  export_freq=False)

# ========================= EOF ====================================================================
