# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===============================================================================

import os
import sys
from datetime import datetime, date
from numpy import ceil, linspace
from pprint import pprint

import ee

from map.assets import list_assets
from map.ee_utils import get_world_climate, ls57mask, ls8mask, ndvi5
from map.ee_utils import ndvi7, ndvi8, ls5_edge_removal, period_stat, daily_landsat

sys.setrecursionlimit(2000)

GEO_DOMAIN = 'users/dgketchum/boundaries/western_11_union'
BOUNDARIES = 'users/dgketchum/boundaries'
ASSET_ROOT = 'users/dgketchum/IrrMapper/version_2'
IRRIGATION_TABLE = 'users/dgketchum/western_states_irr/NV_agpoly'

RF_TRAINING_DATA = 'projects/ee-dgketchum/assets/bands/IrrMapper_RF_training_sample'
RF_TRAINING_POINTS = 'projects/ee-dgketchum/assets/points/IrrMapper_training_data_points'

HUC_6 = 'users/dgketchum/usgs_wbd/huc6_semiarid_clip'
HUC_8 = 'users/dgketchum/usgs_wbd/huc8_semiarid_clip'
COUNTIES = 'users/dgketchum/boundaries/western_counties'

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
IRR = {}
# list of years we have verified irrigated fields
YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997, 1998,
         2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

TEST_YEARS = [2005]
ALL_YEARS = [x for x in range(1986, 2019)]


def reduce_classification(tables, years=None, description=None, cdl_mask=False, min_years=0):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    sum_mask = None
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')
    fc = ee.FeatureCollection(tables)

    if min_years > 0:
        coll = ee.ImageCollection(image_list)
        sum = ee.ImageCollection(coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])).sum()
        sum_mask = sum.lt(min_years)

    # first = True
    for yr in years:
        if yr not in [2002, 2007, 2012]:
            yr_img = [x for x in image_list if x.endswith(str(yr))]
            coll = ee.ImageCollection(yr_img)
            tot = coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])

        if cdl_mask and min_years > 0:
            # cultivated/uncultivated band only available 2013 to 2017
            cdl = ee.Image('USDA/NASS/CDL/2013')
            cultivated = cdl.select('cultivated')
            cdl_crop_mask = cultivated.eq(2)
            tot = tot.mask(cdl_crop_mask).mask(sum_mask)

        elif min_years > 0:
            tot = tot.mask(sum_mask)

        elif cdl_mask:
            cdl = ee.Image('USDA/NASS/CDL/2013')
            cultivated = cdl.select('cultivated')
            cdl_crop_mask = cultivated.eq(2)
            tot = tot.mask(cdl_crop_mask)

        tot = tot.multiply(ee.Image.pixelArea())
        reduce = tot.reduceRegions(collection=fc,
                                   reducer=ee.Reducer.sum(),
                                   scale=30)
        task = ee.batch.Export.table.toCloudStorage(
            reduce,
            description='{}_area_{}_'.format(description, yr),
            bucket='wudr',
            fileNamePrefix='{}_area_{}_'.format(description, yr),
            fileFormat='CSV')
        task.start()
        print(yr)


def get_sr_series(tables, out_name, max_sample=500):
    """This assumes 'YEAR' parameter is already in str(YEAR.js.milliseconds)"""

    pt_ct = 0
    for year in YEARS:
        for state in TARGET_STATES:

            name_prefix = '{}_{}_{}'.format(out_name, state, year)
            local_file = os.path.join('/home/dgketchum/IrrigationGIS/EE_extracts/to_concatenate',
                                      '{}.csv'.format(name_prefix))
            if os.path.isfile(local_file):
                continue
            else:
                print(local_file)

            roi = ee.FeatureCollection(os.path.join(BOUNDARIES, state))

            start = '{}-01-01'.format(year)
            d = datetime.strptime(start, '%Y-%m-%d')
            epoch = datetime.utcfromtimestamp(0)
            start_millisec = str(int((d - epoch).total_seconds() * 1000))

            table = ee.FeatureCollection(tables)
            table = table.filter(ee.Filter.eq('YEAR', start_millisec))
            table = table.filterBounds(roi)
            table = table.randomColumn('rnd')
            points = table.size().getInfo()
            print('{} {} {} points'.format(state, year, points))

            n_splits = int(ceil(points / float(max_sample)))
            ranges = linspace(0, 1, n_splits + 1)
            diff = ranges[1] - ranges[0]

            for enum, slice in enumerate(ranges[:-1], start=1):
                slice_table = table.filter(ee.Filter.And(ee.Filter.gte('rnd', slice),
                                                   ee.Filter.lt('rnd', slice + diff)))
                points = slice_table.size().getInfo()
                print('{} {} {} points'.format(state, year, points))

                name_prefix = '{}_{}_{}_{}'.format(out_name, state, enum, year)
                local_file = os.path.join('/home/dgketchum/IrrigationGIS/EE_extracts/to_concatenate',
                                          '{}.csv'.format(name_prefix))
                if os.path.isfile(local_file):
                    continue
                else:
                    print(local_file)

                pt_ct += points
                if points == 0:
                    continue

                ls_sr_masked = daily_landsat(year, roi)
                stats = ls_sr_masked.sampleRegions(collection=table,
                                                   properties=['POINT_TYPE', 'YEAR', 'LAT_GCS', 'Lon_GCS'],
                                                   scale=30,
                                                   tileScale=16)

                task = ee.batch.Export.table.toCloudStorage(
                    stats,
                    description=name_prefix,
                    bucket='wudr',
                    fileNamePrefix=name_prefix,
                    fileFormat='CSV')

                task.start()
    print('{} total points'.format(pt_ct))


def attribute_irrigation():
    """
    Extracts fraction of vector classified as irrigated. Been using this to attribute irrigation to
    field polygon coverages.
    :return:
    """
    fc = ee.FeatureCollection(IRRIGATION_TABLE)
    for state in TARGET_STATES:
        for yr in range(1986, 2019):
            images = os.path.join(ASSET_ROOT, '{}_{}'.format(state, yr))
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


def export_raster(roi, description):
    fc = ee.FeatureCollection(roi)
    mask = fc.geometry().bounds().getInfo()['coordinates']
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    for yr in range(1986, 2019):
        yr_img = [x for x in image_list if x.endswith(str(yr))]
        coll = ee.ImageCollection(yr_img)
        img = ee.ImageCollection(coll.mosaic().select('classification'))
        img = img.first()
        task = ee.batch.Export.image.toDrive(
            img,
            description='IrrMapper_V2_{}_{}'.format(description, yr),
            folder='Irrigation',
            region=mask,
            scale=30,
            maxPixels=1e13,
            fileNamePrefix='IrrMapper_V2_{}_{}'.format(description, yr))
        task.start()
        print(yr)


def export_special(roi, description):
    fc = ee.FeatureCollection(roi)
    roi_mask = fc.geometry().bounds().getInfo()['coordinates']
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    # years = [str(x) for x in range(1986, 1991)]
    # target_images = [x for x in image_list if x.endswith(years[0])]
    # target = ee.ImageCollection(image_list)
    # target = target.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])
    # range_images = [x for x in image_list if x.endswith(tuple(years))]

    coll = ee.ImageCollection(image_list)
    sum = ee.ImageCollection(coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])).sum().toDouble()
    sum_mask = sum.lt(3)

    img = sum.mask(sum_mask).toDouble()

    task = ee.batch.Export.image.toDrive(
        sum,
        description='IrrMapper_V2_sum_years',
        # folder='Irrigation',
        region=roi_mask,
        scale=30,
        maxPixels=1e13,
        # fileNamePrefix='IrrMapper_V2_{}_{}'.format(description, period)
    )
    task.start()


def export_classification(out_name, asset_root, region, export='asset'):
    """
    Trains a Random Forest classifier using a training table input, creates a stack of raster images of the same
    features, and classifies it.  I run this over a for-loop iterating state by state.
    :param region:
    :param asset_root:
    :param out_name:
    :param asset:
    :param export:
    :return:
    """
    fc = ee.FeatureCollection(None)
    roi = ee.FeatureCollection(region)
    mask = roi.geometry().bounds().getInfo()['coordinates']

    classifier = ee.Classifier.randomForest(
        numberOfTrees=100,
        variablesPerSplit=0,
        minLeafPopulation=1,
        outOfBagMode=False).setOutputMode('CLASSIFICATION')

    input_props = fc.first().propertyNames().remove('YEAR').remove('POINT_TYPE').remove('system:index')

    feature_bands = sorted([b for b in fc.first().getInfo()['properties']])
    feature_bands.remove('POINT_TYPE')
    feature_bands.remove('YEAR')

    trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

    for yr in TEST_YEARS:
        input_bands = stack_bands(yr, roi)
        annual_stack = input_bands.select(input_props)
        classified_img = annual_stack.classify(trained_model).int().set({
            'system:index': ee.Date('{}-01-01'.format(yr)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-01-01'.format(yr)).millis(),
            'geography': out_name,
            'class_key': '0: irrigated, 1: rainfed, 2: uncultivated, 3: wetland'})

        if export == 'asset':
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                assetId=os.path.join(asset_root, '{}_{}'.format(out_name, yr)),
                fileNamePrefix='{}_{}'.format(yr, out_name),
                region=mask,
                scale=30,
                maxPixels=1e13)
        elif export == 'cloud':
            task = ee.batch.Export.image.toCloudStorage(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                bucket='wudr',
                fileNamePrefix='{}_{}'.format(yr, out_name),
                region=mask,
                scale=30,
                maxPixels=1e13)
        else:
            raise NotImplementedError('choose asset or cloud for export')

        task.start()
        print(yr)


def filter_irrigated(filter_type='filter_low'):
    """
    Takes a field polygon vector and filters it based on NDVI rules. At present, the function keeps features
    where the lower 15 percentile reach NDVI greater than 0.5 in either early or late summer.

    :filter_type: filter_low is to filter out low ndvi fields (thus saving and returning high-ndvi fields,
            likely irrigated), filter_high filters out high-ndvi feilds, leaving likely fallowed fields
    :return:
    """
    for k, v in IRR.items():
        plots = ee.FeatureCollection(v[0])

        for year in v[1]:
            # pprint(plots.first().getInfo())
            start = '{}-01-01'.format(year)

            early_summer_s = ee.Date(start).advance(5, 'month')
            early_summer_e = ee.Date(start).advance(7, 'month')
            late_summer_s = ee.Date(start).advance(7, 'month')
            late_summer_e = ee.Date(start).advance(10, 'month')

            if year <= 2011:
                collection = ndvi5()
            elif year == 2012:
                collection = ndvi7()
            else:
                collection = ndvi8()

            early_collection = period_stat(collection, early_summer_s, early_summer_e)
            late_collection = period_stat(collection, late_summer_s, late_summer_e)
            summer_collection = period_stat(collection, early_summer_s, late_summer_e)

            if filter_type == 'filter_low':
                early_nd_max = early_collection.select('nd_mean').reduce(ee.Reducer.intervalMean(0.0, 15.0))
                early_int_mean = early_nd_max.reduceRegions(collection=plots,
                                                            reducer=ee.Reducer.mean(),
                                                            scale=30.0)

                s_nd_max = late_collection.select('nd_mean').reduce(ee.Reducer.intervalMean(0.0, 15.0))
                combo_mean = s_nd_max.reduceRegions(collection=early_int_mean,
                                                    reducer=ee.Reducer.mean(),
                                                    scale=30.0)
                filt_fc = combo_mean.filter(ee.Filter.Or(ee.Filter.gt('mean', v[2]), ee.Filter.gt('mean', v[2])))

            elif filter_type == 'filter_high':

                irrmapper = ee.ImageCollection(ASSET_ROOT)
                img = ee.Image()
                for y in range(1986, 2019):
                    i = irrmapper.filterDate('{}-01-01'.format(y), '{}-12-31'.format(y)).mosaic()
                    i = i.remap([0, 1, 2, 3], [1, 0, 0, 0]).rename('irr')
                    img = img.addBands(i)

                img = img.reduce(ee.Reducer.sum())
                equipped = img.gte(10).rename('equip')
                equip = equipped.reduceRegions(collection=plots,
                                               reducer=ee.Reducer.mode(),
                                               scale=30.0)

                summer_nd_max = summer_collection.select('nd_max')
                early_int_mean = summer_nd_max.reduceRegions(collection=equip,
                                                             reducer=ee.Reducer.mean(),
                                                             scale=30.0)

                filt_fc = early_int_mean.filter(ee.Filter.And(ee.Filter.lt('mean', 0.65), ee.Filter.eq('mode', 1)))

            else:
                raise NotImplementedError('must choose from filter_low or filter_high')

            task = ee.batch.Export.table.toCloudStorage(filt_fc,
                                                        folder='Irrigation',
                                                        description='{}_{}_{}'.format(k, filter_type, year),
                                                        bucket='wudr',
                                                        fileNamePrefix='{}_{}_{}'.format(k, filter_type, year),
                                                        fileFormat='KML')

            task.start()
            print(k, year, filter_type, filt_fc.size().getInfo())


def request_validation_extract(file_prefix='validation'):
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
    roi = ee.FeatureCollection(GEO_DOMAIN)
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


def request_band_extract(file_prefix, points_layer, region, filter_bounds=False):
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
    plots = ee.FeatureCollection(points_layer)
    for yr in ALL_YEARS:
        stack = stack_bands(yr, roi)
        start = '{}-01-01'.format(yr)
        d = datetime.strptime(start, '%Y-%m-%d')
        epoch = datetime.utcfromtimestamp(0)
        start_millisec = (d - epoch).total_seconds() * 1000

        if filter_bounds:
            plots = plots.filterBounds(roi)

        filtered = plots.filter(ee.Filter.eq('YEAR', ee.Number(start_millisec)))

        plot_sample_regions = stack.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR'],
            scale=30,
            tileScale=2)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print(yr)


def stack_bands(yr, roi):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param roi:
    :return:
    """
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)
    water_year_start = '{}-10-01'.format(yr - 1)

    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-01'.format(yr)
    summer_s, summer_e = '{}-07-01'.format(yr), '{}-09-01'.format(yr)
    fall_s, fall_e = '{}-09-01'.format(yr), '{}-11-01'.format(yr)

    l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls57mask)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls8mask)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))
    lsSR_spr_mn = ee.Image(lsSR_masked.filterDate(spring_s, spring_e).mean())
    lsSR_lspr_mn = ee.Image(lsSR_masked.filterDate(late_spring_s, late_spring_e).mean())
    lsSR_sum_mn = ee.Image(lsSR_masked.filterDate(summer_s, fall_s).mean())
    lsSR_fal_mn = ee.Image(lsSR_masked.filterDate(fall_s, fall_e).mean())

    proj = lsSR_sum_mn.select('B2').projection().getInfo()
    input_bands = lsSR_spr_mn.addBands([lsSR_lspr_mn, lsSR_sum_mn, lsSR_fal_mn])

    nd_list_ = []
    for pos, year in zip(['m2', 'm1', 'cy'], range(yr - 2, yr + 1)):
        if year <= 2011:
            collection = ndvi5()
        elif year == 2012:
            collection = ndvi7()
        else:
            collection = ndvi8()

        nd_collection = period_stat(collection, spring_s.replace('{}'.format(yr), '{}'.format(year)),
                                    fall_e.replace('{}'.format(yr), '{}'.format(year)))
        s_nd_max = nd_collection.select('nd_max').rename('nd_max_{}'.format(pos))
        nd_list_.append(s_nd_max)

    input_bands = input_bands.addBands(nd_list_)

    for s, e, n in [(spring_s, spring_e, 'espr'),
                    (late_spring_s, late_spring_e, 'lspr'),
                    (summer_s, summer_e, 'smr'),
                    (fall_s, fall_e, 'fl'),
                    (water_year_start, spring_e, 'wy_espr'),
                    (water_year_start, late_spring_e, 'wy_espr'),
                    (water_year_start, summer_e, 'wy_smr'),
                    (water_year_start, fall_e, 'wy')]:
        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
            roi).filterDate(s, e).select('pr', 'eto', 'tmmn', 'tmmx')
        temp_reducer = ee.Reducer.mean()
        t_names = ['tmax'.format(n), 'tmin'.format(n)]
        temp_perc = gridmet.select('tmmn', 'tmmx').reduce(temp_reducer).rename(t_names).resample(
            'bilinear').reproject(crs=proj['crs'], scale=30)

        precip_reducer = ee.Reducer.sum()
        precip_sum = gridmet.select('pr', 'eto').reduce(precip_reducer).rename(
            'precip_total_{}'.format(n), 'pet_total_{}'.format(n)).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                  scale=30)
        wd_estimate = precip_sum.select('precip_total_{}'.format(n)).subtract(precip_sum.select(
            'pet_total_{}'.format(n))).rename('wd_est_{}'.format(n))
        input_bands = input_bands.addBands([temp_perc, precip_sum, wd_estimate])

    temp_reducer = ee.Reducer.percentile([10, 50, 90])
    t_names = ['tmmn_p10_cy', 'tmmn_p50_cy', 'tmmn_p90_cy', 'tmmx_p10_cy', 'tmmx_p50_cy', 'tmmx_p90_cy']
    temp_perc = gridmet.select('tmmn', 'tmmx').reduce(temp_reducer).rename(t_names).resample(
        'bilinear').reproject(crs=proj['crs'], scale=30)

    precip_reducer = ee.Reducer.sum()
    precip_sum = gridmet.select('pr', 'eto').reduce(precip_reducer).rename(
        'precip_total_cy', 'pet_total_cy').resample('bilinear').reproject(crs=proj['crs'], scale=30)
    wd_estimate = precip_sum.select('precip_total_cy').subtract(precip_sum.select(
        'pet_total_cy')).rename('wd_est_cy')

    coords = ee.Image.pixelLonLat().rename(['Lon_GCS', 'LAT_GCS']).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                  scale=30)
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect').reduceResolution(
        ee.Reducer.mean()).reproject(crs=proj['crs'], scale=30)

    world_climate = get_world_climate(proj=proj)
    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    static_input_bands = coords.addBands([temp_perc, wd_estimate, terrain, tpi_1250, tpi_250, tpi_150, world_climate])

    nlcd = ee.Image('USGS/NLCD/NLCD2011').select('landcover').reproject(crs=proj['crs'], scale=30).rename('nlcd')
    cdl = ee.Image('USDA/NASS/CDL/2017').select('cultivated').remap([1, 2], [0, 1]).reproject(crs=proj['crs'],
                                                                                              scale=30).rename('cdl')
    static_input_bands = static_input_bands.addBands([nlcd, cdl])

    input_bands = input_bands.addBands(static_input_bands).clip(roi)

    # standardize names to match EE javascript output
    standard_names = []
    temp_ct = 1
    prec_ct = 1
    names = input_bands.bandNames().getInfo()
    pprint(sorted(names))
    for name in names:
        if 'B' in name and '_1_1' in name:
            replace_ = name.replace('_1_1', '_2')
            standard_names.append(replace_)
        elif 'B' in name and '_2' in name:
            replace_ = name.replace('_2', '_3')
            standard_names.append(replace_)
        elif 'tavg' in name and 'tavg' in standard_names:
            standard_names.append('tavg_{}'.format(temp_ct))
            temp_ct += 1
        elif 'prec' in name and 'prec' in standard_names:
            standard_names.append('prec_{}'.format(prec_ct))
            prec_ct += 1
        else:
            standard_names.append(name)

    input_bands = input_bands.rename(standard_names)
    return input_bands


def is_authorized():
    try:
        ee.Initialize()  # investigate (use_cloud_api=True)
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


if __name__ == '__main__':
    is_authorized()
    get_sr_series(RF_TRAINING_POINTS, 'sr_series', max_sample=100)
# ========================= EOF ====================================================================
