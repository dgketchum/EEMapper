import os
import sys
from datetime import datetime, date

from numpy import ceil, linspace
from pprint import pprint

import ee
from numpy import ceil, linspace

sys.path.insert(0, os.path.abspath('..'))
from map.assets import list_assets, copy_asset
from map.ee_utils import get_world_climate, landsat_composites, landsat_masked
from map.cdl import get_cdl

sys.setrecursionlimit(2000)

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
            cultivated, _ = get_cdl(yr)
            cdl_crop_mask = cultivated.eq(1)
            tot = tot.mask(cdl_crop_mask).mask(sum_mask)

        elif min_years > 0:
            tot = tot.mask(sum_mask)

        elif cdl_mask:
            cultivated, _ = get_cdl(yr)
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


def export_raster(roi=None, min_years=3, debug=False):
    irr_min_yr_mask = None
    roi = ee.FeatureCollection(roi).first()

    irr_coll = ee.ImageCollection(RF_ASSET)

    coll = irr_coll.filterDate('1987-01-01', '2009-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))

    if min_years:
        irr_min_yr_mask = remap.sum().gte(min_years)
        sum = remap.sum().mask(irr_min_yr_mask)
    else:
        sum = remap.sum()

    sum = sum.clip(roi.geometry()).toInt()

    desc = 'irrmapper_freq_1987_2009_no_mask_07MAY2024'
    task = ee.batch.Export.image.toCloudStorage(
        image=sum,
        description=desc,
        bucket='wudr',
        fileNamePrefix=desc,
        region=roi.geometry(),
        scale=30,
        maxPixels=1e13,
        crs='EPSG:5071',
        fileFormat='GeoTIFF')
    print(desc)
    task.start()

    coll = irr_coll.filterDate('2009-01-01', '2009-12-31').select('classification')
    if irr_min_yr_mask:
        remap = coll.map(lambda img: img.lt(1)).mosaic().mask(irr_min_yr_mask).toInt()
    else:
        remap = coll.map(lambda img: img.lt(1)).mosaic().toInt()
    remap = remap.clip(roi.geometry())

    if debug:
        pt = ee.FeatureCollection(ee.Geometry.Point([-112.6152495034253, 48.689606909150044]))
        data = remap.sampleRegions(collection=pt, scale=30)
        data = data.getInfo()

    desc = 'irrmapper_status_2009'
    task = ee.batch.Export.image.toCloudStorage(
        image=remap,
        description=desc,
        bucket='wudr',
        fileNamePrefix=desc,
        region=roi.geometry(),
        scale=30,
        maxPixels=1e13,
        crs='EPSG:5071',
        fileFormat='GeoTIFF')
    print(desc)
    # task.start()


def export_special(input_coll, out_coll, roi, description):
    fc = ee.FeatureCollection(roi)
    ned = ee.Image('USGS/NED')
    slope = ee.Terrain.products(ned).select('slope')

    for year in range(2022, 2024):
        start, end = '{}-03-01'.format(year), '{}-12-30'.format(year)
        ndvi = landsat_composites(year, start, end, fc, 'gs', composites_only=True).select('nd_max_gs')

        cropland = get_cdl(year)[1].select('cropland')

        target = ee.Image(os.path.join(input_coll, '{}_{}'.format(description, year)))
        props = target.getInfo()['properties']
        target = target.select('classification').clip(fc.geometry())

        sum_coll = ee.ImageCollection(input_coll)
        remap = ee.ImageCollection(sum_coll).map(lambda x: x.select('classification').remap([0, 1, 2, 3],
                                                                                            [1, 0, 0, 0]))
        sum = remap.sum().rename('sum')
        if description == 'MT':
            if year < 2010:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2009').filterBounds(fc)
            elif year < 2012:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2011').filterBounds(fc)
            elif year < 2014:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2013').filterBounds(fc)
            elif year < 2019:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2015').filterBounds(fc)
            else:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2019').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 6
            if year < 2008:

                expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                              ': (IRR == 0) && (SUM < {t}) ? 1' \
                              ': (IRR == 0) && (SLOPE > 10) ? 3' \
                              ': IRR'.format(t=threshold)

                target = expr.expression(expression_,
                                         {'IRR': expr.select('classification'),
                                          'SUM': expr.select('sum'),
                                          'NDVI': expr.select('nd_max_gs'),
                                          'SLOPE': expr.select('slope')})
            else:
                expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                              ': (IRR == 0) && (SUM < {t}) ? 1' \
                              ': (IRR == 0) && (SLOPE > 10) ? 3' \
                              ': IRR'.format(t=threshold)

                target = expr.expression(expression_,
                                         {'IRR': expr.select('classification'),
                                          'SUM': expr.select('sum'),
                                          'NDVI': expr.select('nd_max_gs'),
                                          'SLOPE': expr.select('slope')})

                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})

        elif description == 'ID':
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2011 else (2023 - year - 1)
            if threshold < 0:
                threshold = 0
            threshold = 5

            expression_ = ' (IRR == 0) && (NDVI < 0.68) && (SUM > {t}) ? 1' \
                          ': (IRR != 0) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs')})

            expression_ = ' (IRR == 0) && (SLOPE > 6) ? 3' \
                          ': IRR'.format(t=threshold)

            target = target.expression(expression_,
                                       {'IRR': target.select('classification'),
                                        'SLOPE': expr.select('slope')})

            if year > 2010:
                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) && (SUM > {t}) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})

        elif description in ['WA', 'OR', 'NM', 'NV']:
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2016 else (2025 - year - 1)
            if threshold < 0:
                threshold = 0

            expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': (IRR == 0) && (SLOPE > 10) ? 3' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs'),
                                      'SLOPE': expr.select('slope')})

            if year > 2011:
                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) && (SUM > {t}) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})

        elif description in ['CO', 'WY']:
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2016 else (2025 - year - 1)
            if threshold < 0:
                threshold = 0

            threshold = 3
            expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': (IRR == 0) && (SLOPE > 10) ? 3' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs'),
                                      'SLOPE': expr.select('slope')})

        elif description in ['CA']:
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2016 else (2021 - year - 1)
            if threshold < 0:
                threshold = 0

            expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': (IRR == 0) && (SLOPE > 10) ? 3' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs'),
                                      'SLOPE': expr.select('slope')})

            if year > 2011:
                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) && (SUM > {t}) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})
        else:
            src = os.path.join(input_coll, '{}_{}'.format(description, year))
            dst = os.path.join(out_coll, '{}_{}'.format(description, year))
            print('No rule written for this state, copying')
            copy_asset(src, dst)
            continue

        props.update({'post_process': expression_})
        target.set(props)
        target = target.rename('classification')

        desc = '{}_{}'.format(description, year)
        _id = os.path.join(out_coll, desc)
        task = ee.batch.Export.image.toAsset(
            target,
            description=desc,
            pyramidingPolicy={'.default': 'mode'},
            assetId=_id,
            scale=30,
            maxPixels=1e13)

        task.start()
        print(year, _id)


def export_classification(out_name, table, asset_root, region, years,
                          export='asset', bag_fraction=0.5, input_props=None, southern=False):
    """
    Trains a Random Forest classifier using a training table input, creates a stack of raster images of the same
    features, and classifies it.  I run this over a for-loop iterating state by state.
    :param region:
    :param asset_root:
    :param out_name:
    :param asset:
    :param export:
    :param bag_fraction:
    :return:
    """
    fc = ee.FeatureCollection(table)
    roi = ee.FeatureCollection(region)

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=150,
        minLeafPopulation=1,
        bagFraction=bag_fraction).setOutputMode('CLASSIFICATION')

    if not input_props:
        input_props = fc.first().propertyNames().remove('YEAR').remove('POINT_TYPE').remove('system:index')
    else:
        input_props = ee.List(input_props)

    trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

    for yr in years:
        input_bands = stack_bands(yr, roi, southern)

        b, p = input_bands.bandNames().getInfo(), input_props.getInfo()
        check = [x for x in p if x not in b]
        if check:
            pprint(check)
            revised = [f for f in p if f not in check]
            input_props = ee.List(revised)
            trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

        annual_stack = input_bands.select(input_props)
        classified_img = annual_stack.unmask().classify(trained_model).int().set({
            'system:index': ee.Date('{}-01-01'.format(yr)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-01-01'.format(yr)).millis(),
            'system:time_end': ee.Date('{}-12-31'.format(yr)).millis(),
            'date_ingested': str(date.today()),
            'image_name': out_name,
            'training_data': table,
            'bag_fraction': bag_fraction,
            'class_key': '0: irrigated, 1: rainfed, 2: uncultivated, 3: wetland'})

        classified_img = classified_img.clip(roi.geometry())

        if export == 'asset':
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                assetId=os.path.join(asset_root, '{}_{}'.format(out_name, yr)),
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)

        elif export == 'cloud':
            task = ee.batch.Export.image.toCloudStorage(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                bucket='wudr',
                fileNamePrefix='{}_{}'.format(yr, out_name),
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)
        else:
            raise NotImplementedError('choose asset or cloud for export')

        task.start()
        print(os.path.join(asset_root, '{}_{}'.format(out_name, yr)))


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


def request_validation_extract(roi, file_prefix='validation'):
    """
    This takes a sample points set and extracts the classification result.  This is a roundabout cross-validation.
    Rather than using holdouts in the Random Forest classifier, we just run all the training data to train the
    classifier, and come back later with this function and a seperate set of points (with known classes) to
    independently test classifier accuracy.
    Other options to achieve this is to use out-of-bag cross validation, or set up a sckikit-learn RF classifier and
    use k-folds cross validation.
    :param file_prefix:
    :return:
    """
    roi = ee.FeatureCollection(roi)
    plots = ee.FeatureCollection(None).filterBounds(roi)
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    for yr in YEARS:
        yr_img = [x for x in image_list if x.endswith(str(yr))]
        coll = ee.ImageCollection(yr_img)
        classified = coll.mosaic().select('classification')

        filtered = plots.filter(ee.Filter.eq('YEAR', yr))

        plot_sample_regions = classified.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR', 'FID'],
            scale=30)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print(yr)


def request_band_extract(file_prefix, points_layer, region, years, filter_bounds=False, buffer=None,
                         southern=False, filter_years=True, diagnose=False, properties=None):
    """
    Extract raster values from a points kml file in Fusion Tables. Send annual extracts .csv to GCS wudr bucket.
    Concatenate them using map.tables.concatenate_band_extract().
    :param region:
    :param points_layer:
    :param file_prefix:
    :param filter_bounds: Restrict extract to within a geographic extent.
    :return:
    """
    roi = ee.FeatureCollection(region)
    if buffer:
        roi = ee.Feature(roi.first()).buffer(buffer)
        roi = ee.FeatureCollection([roi])
    plots = ee.FeatureCollection(points_layer)
    for yr in years:
        stack = stack_bands(yr, roi, southern)

        if filter_bounds:
            plots = plots.filterBounds(roi)

        if filter_years:
            filtered = plots.filter(ee.Filter.eq('YEAR', yr))
        else:
            filtered = plots

        # if tables are coming out empty, use this to find missing bands
        if diagnose:
            filtered = ee.FeatureCollection([filtered.first()])
            bad_ = []
            bands = stack.bandNames().getInfo()
            for b in bands:
                stack_ = stack.select([b])

                def sample_regions(i, points):
                    red = ee.Reducer.toCollection(i.bandNames())
                    reduced = i.reduceRegions(points, red, 30, stack_.select(b).projection())
                    fc = reduced.map(lambda f: ee.FeatureCollection(f.get('features'))
                                     .map(lambda q: q.copyProperties(f, None, ['features'])))
                    return fc.flatten()

                data = sample_regions(stack_, filtered)
                try:
                    print(b, data.getInfo()['features'][0]['properties'][b])
                except Exception as e:
                    print(b, 'not there', e)
                    bad_.append(b)
            print(bad_)
            return None

        props = ['POINT_TYPE', 'YEAR']
        if properties:
            props = properties

        plot_sample_regions = stack.sampleRegions(
            collection=filtered,
            properties=props,
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print('{}_{}'.format(file_prefix, yr))


def stack_bands(yr, roi, southern=False):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """

    water_year_start = '{}-10-01'.format(yr - 1)

    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    prev_s, prev_e = '{}-05-01'.format(yr - 1), '{}-09-30'.format(yr - 1),
    p_summer_s, p_summer_e = '{}-07-15'.format(yr - 1), '{}-09-30'.format(yr - 1)

    pprev_s, pprev_e = '{}-05-01'.format(yr - 2), '{}-09-30'.format(yr - 2),
    pp_summer_s, pp_summer_e = '{}-07-15'.format(yr - 2), '{}-09-30'.format(yr - 2)

    if southern:
        periods = [('gs', winter_s, fall_e),
                   ('1', winter_s, spring_e),
                   ('2', late_spring_s, late_spring_e),
                   ('3', summer_s, summer_e),
                   # modify to run in September
                   # ('4', fall_s, fall_e),

                   ('m1', prev_s, prev_e),
                   ('3_m1', p_summer_s, p_summer_e),

                   ('m2', pprev_s, pprev_e),
                   ('3_m2', pp_summer_s, pp_summer_e)]
    else:
        periods = [('gs', spring_e, fall_s),
                   ('1', spring_s, spring_e),
                   ('2', late_spring_s, late_spring_e),
                   ('3', summer_s, summer_e),
                   # modify to run in September
                   ('4', fall_s, fall_e),

                   ('m1', prev_s, prev_e),
                   ('3_m1', p_summer_s, p_summer_e),

                   ('m2', pprev_s, pprev_e),
                   ('3_m2', pp_summer_s, pp_summer_e)]

    first = True
    for name, start, end in periods:
        prev = 'm' in name
        bands = landsat_composites(yr, start, end, roi, name, composites_only=prev)
        if first:
            input_bands = bands
            proj = bands.select('B2_gs').projection().getInfo()
            first = False
        else:
            input_bands = input_bands.addBands(bands)

    integrated_composite_bands = []

    for feat in ['nd', 'gi', 'nw', 'evi']:
        # modify to run in September
        # periods = [x for x in range(2, 5)]
        periods = [x for x in range(2, 4)]
        c_bands = ['{}_{}'.format(feat, p) for p in periods]
        b = input_bands.select(c_bands).reduce(ee.Reducer.sum()).rename('{}_int'.format(feat))

        integrated_composite_bands.append(b)

    input_bands = input_bands.addBands(integrated_composite_bands)

    for s, e, n, m in [(spring_s, late_spring_e, 'spr', (3, 8)),
                       (water_year_start, spring_e, 'wy_spr', (10, 5)),
                       (water_year_start, summer_e, 'wy_smr', (10, 9))]:
        nldas = ee.ImageCollection('NASA/NLDAS/FORA0125_H002').filterBounds(roi).filterDate(s, e)
        nldas = nldas.select('total_precipitation', 'potential_evaporation', 'temperature')

        temp = ee.Image(nldas.select('temperature').mean())
        temp = temp.resample('bilinear').reproject(crs=proj['crs'], scale=30)

        ai_sum = nldas.select('total_precipitation', 'potential_evaporation').reduce(ee.Reducer.sum()).rename(
            'prec_tot_{}'.format(n), 'pet_tot_{}'.format(n)).resample('bilinear').reproject(crs=proj['crs'],
                                                                                            scale=30)
        wd_estimate = ai_sum.select('prec_tot_{}'.format(n)).subtract(ai_sum.select(
            'pet_tot_{}'.format(n))).rename('cwd_{}'.format(n))

        worldclim_prec = get_world_climate(proj=proj, months=m, param='prec')
        anom_prec = ai_sum.select('prec_tot_{}'.format(n)).subtract(worldclim_prec)
        worldclim_temp = get_world_climate(proj=proj, months=m, param='tavg')
        anom_temp = temp.subtract(worldclim_temp).rename('an_temp_{}'.format(n))

        input_bands = input_bands.addBands([temp, ai_sum, wd_estimate, anom_temp, anom_prec])

    coords = ee.Image.pixelLonLat().rename(['lon', 'lat']).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                  scale=30)
    ned = ee.Image('CGIAR/SRTM90_V4')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect').reduceResolution(
        ee.Reducer.mean()).reproject(crs=proj['crs'], scale=30)

    landforms = ee.Image('CSP/ERGo/1_0/Global/SRTM_landforms').rename('landforms')
    globcover = ee.Image('ESA/GLOBCOVER_L4_200901_200912_V2_3').select('landcover').rename('globcover')
    esacov = ee.ImageCollection('ESA/WorldCover/v100').first().rename('esacov')

    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    input_bands = input_bands.addBands([coords, terrain, tpi_1250, tpi_250, tpi_150, anom_prec, anom_temp])

    gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
    occ_pos = gsw.select('occurrence').gt(0)
    water = occ_pos.unmask(0).rename('gsw')

    input_bands = input_bands.addBands([landforms, globcover, esacov, water])

    input_bands = input_bands.clip(roi)

    standard_names = []
    temp_ct = 1
    prec_ct = 1
    names = input_bands.bandNames().getInfo()
    for name in names:
        if 'tavg' in name and 'tavg' in standard_names:
            standard_names.append('tavg_{}'.format(temp_ct))
            temp_ct += 1
        elif 'prec' in name and 'prec' in standard_names:
            standard_names.append('prec_{}'.format(prec_ct))
            prec_ct += 1
        elif 'nd_cy' in name:
            standard_names.append('nd_max_cy')
        else:
            standard_names.append(name)

    input_bands = input_bands.rename(standard_names)
    return input_bands


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


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


if __name__ == '__main__':
    is_authorized()
    out_c = 'users/dgketchum/IrrMapper/IrrMapper_sw'
    geo_ = 'users/dgketchum/boundaries/blackfeet_res'

    export_raster(geo_, min_years=None, debug=False)

# ========================= EOF ====================================================================
