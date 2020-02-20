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

## UCRB is going with the pure state input data for now.
# 'UCRB_WY': ('ft:1M0GDErc0dgoYajU_HStZBkp-hBL4kUiZufFdtWHG', [1989, 1996, 2010, 2013, 2016], 0.5),  # a.k.a. 2000
# 'UCRB_UT_CO': ('ft:1Av2WlcPRBd7JZqYOU73VCLOJ-b5q6H5u6Bboebdv', [1998, 2003, 2006, 2013, 2016], 0.5),  # a.k.a. 2005
# 'UCRB_UT': ('ft:144ymxhlcv8lj1u_BYQFEC1ITmiISW52q5JvxSVyk', [1998, 2003, 2006, 2013, 2016], 0.5),  # a.k.a. 2006
# 'UCRB_NM': ('ft:1pBSJDPdFDHARbdc5vpT5FzRek-3KXLKjNBeVyGdR', [1987, 2001, 2004, 2007, 2016], 0.4),  # a.k.a. 2009
# ===============================================================================

import os
from datetime import datetime
from pprint import pprint

import ee

from map.assets import list_assets

ROI = 'users/dgketchum/boundaries/western_11_union'
BOUNDARIES = 'users/dgketchum/boundaries/MT'

ASSET_ROOT = 'projects/ee-dgketchum/IrrMapper/version_3'
IRRIGATION_TABLE = 'users/dgketchum/western_states_irr/NV_agpoly'
COUNTIES = 'users/dgketchum/boundaries/western_counties'

STATES = ['AZ', 'CA', 'NV', 'CO', 'ID', 'MT', 'NM', 'OR', 'UT', 'WA', 'WY']  #
TARGET_STATES = ['AZ', 'CA', 'NM', 'NV', 'UT']

POINTS_MT = 'users/dgketchum/point_sample/points_rdgp_20FEB2020'
TABLE_MT = 'ft:1-2IMLOk64CGhr1Lz53am02pnFw4-ReDioNSKTYU-'

# list of years we have verified irrigated fields
YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997, 1998,
         2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

RDGP_YEARS = [2002, 2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015, 2016]

TEST_YEARS = [2013]
# TEST_YEARS = [x for x in range(2018, 2019)]
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
    alt_image_list = list_assets('users/dpendergraph/IrrMapper_v3')
    removal = []
    for i in image_list:
        if 'MT_' in i:
            removal.append(i)
    [image_list.remove(r) for r in removal]
    image_list = image_list + alt_image_list
    fc = ee.FeatureCollection(tables)

    if min_years > 0:
        coll = ee.ImageCollection(image_list)
        sum = ee.ImageCollection(coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])).sum()
        sum_mask = sum.lt(min_years)

    for yr in years:
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


def get_ndvi_stats(tables, years, out_name):
    fc = ee.FeatureCollection(tables)
    i = get_ndvi_series(years, fc)
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    for yr in years:
        coll = ee.ImageCollection(image_list).filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr))
        remap = coll.mosaic().select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])
        cls = remap.rename('irr_{}'.format(yr))
        i = i.addBands(cls)

    stats = i.reduceRegions(collection=fc,
                            reducer=ee.Reducer.mean(),
                            scale=30)

    task = ee.batch.Export.table.toCloudStorage(
        stats,
        description='{}'.format(out_name),
        bucket='wudr',
        fileNamePrefix='{}'.format(out_name),
        fileFormat='KML')

    task.start()


def attribute_irrigation():
    """
    Extracts fraction of vector classified as irrigated. Been using this to attribute irrigation to
    field polygon coverages.
    :return:
    """
    fc = ee.FeatureCollection(None)
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
                description='lcrb_{}_{}'.format(state, yr),
                bucket='wudr',
                fileNamePrefix='lcrb_attr_{}_{}'.format(state, yr),
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
    :param export:
    :return:
    """
    fc = ee.FeatureCollection(TABLE_MT)
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
    roi = ee.FeatureCollection(ROI)
    plots = ee.FeatureCollection(VALIDATION_POINTS).filterBounds(roi)
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    for yr in YEARS:
        yr_img = [x for x in image_list if x.endswith(str(yr))]
        coll = ee.ImageCollection(yr_img)
        classified = coll.mosaic().select('classification')

        start = '{}-01-01'.format(yr)
        d = datetime.strptime(start, '%Y-%m-%d')
        epoch = datetime.utcfromtimestamp(0)
        start_millisec = (d - epoch).total_seconds() * 1000
        filtered = plots.filter(ee.Filter.eq('YEAR', ee.Number(start_millisec)))

        plot_sample_regions = classified.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR'],
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
    print(plots.size().getInfo())
    for yr in RDGP_YEARS:
        stack = stack_bands(yr, roi)

        if filter_bounds:
            plots = plots.filterBounds(roi)

        filtered = plots.filter(ee.Filter.eq('YEAR', yr))
        print(filtered.size().getInfo())

        plot_sample_regions = stack.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR'],
            scale=30,
            tileScale=2)
        
        if not plot_sample_regions.first().getInfo():
            print('none for ', yr)
            pass
        
        else:
            task = ee.batch.Export.table.toCloudStorage(
                plot_sample_regions,
                description='{}_{}'.format(file_prefix, yr),
                bucket='wudr',
                fileNamePrefix='{}_{}'.format(file_prefix, yr),
                fileFormat='CSV')
    
            task.start()
            pprint(yr)


def get_ndvi_series(years, roi):
    """ Stack NDVI bands """
    ndvi_l5, ndvi_l7, ndvi_l8 = ndvi5(), ndvi7(), ndvi8()
    ndvi = ee.ImageCollection(ndvi_l5.merge(ndvi_l7).merge(ndvi_l8)).filterBounds(roi)

    def ndvi_means(date):
        etc = ndvi.filterDate(ee.Date(date).advance(4, 'month'),
                              ee.Date(date).advance(9, 'month')).toBands()
        stats = ee.Image(etc.reduce(ee.Reducer.mean()).rename('nd_mean_{}'.format(date[:4])))
        return stats

    def ndvi_max(date):
        etc = ndvi.filterDate(ee.Date(date).advance(4, 'month'),
                              ee.Date(date).advance(9, 'month')).toBands()
        stats = ee.Image(etc.reduce(ee.Reducer.max()).rename('nd_max_{}'.format(date[:4])))
        return stats

    def ndvi_min(date):
        etc = ndvi.filterDate(ee.Date(date).advance(4, 'month'),
                              ee.Date(date).advance(9, 'month')).toBands()
        stats = ee.Image(etc.reduce(ee.Reducer.min()).rename('nd_min_{}'.format(date[:4])))
        return stats

    bands_list = []
    for yr in years:
        d = '{}-01-01'.format(yr)

        bands_mean = ndvi_means(d)
        bands_list.append(bands_mean.rename('nd_mean_{}'.format(yr)))

        bands_max = ndvi_max(d)
        bands_list.append(bands_max.rename('nd_max_{}'.format(yr)))

        bands_min = ndvi_min(d)
        bands_list.append(bands_min.rename('nd_min_{}'.format(yr)))

    i = ee.Image(bands_list)
    return i


def add_doy(image):
    """ Add day-of-year image """
    mask = ee.Date(image.get('system:time_start'))
    day = ee.Image.constant(image.date().getRelative('day', 'year')).clip(image.geometry())
    i = image.addBands(day.rename('DOY')).int().updateMask(mask)
    return i


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


def get_world_climate(proj):
    n = list(range(1, 13))
    months = [str(x).zfill(2) for x in n]
    parameters = ['tavg', 'tmin', 'tmax', 'prec']
    combinations = [(m, p) for m in months for p in parameters]

    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).select(p).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                scale=30) for m, p in
         combinations]
    # not sure how to do this without initializing the image with a constant
    i = ee.Image(l)
    return i


def get_qa_bits(image, start, end, qa_mask):
    pattern = 0
    for i in range(start, end - 1):
        pattern += 2 ** i
    return image.select([0], [qa_mask]).bitwiseAnd(pattern).rightShift(start)


def mask_quality(image):
    QA = image.select('pixel_qa')
    shadow = get_qa_bits(QA, 3, 3, 'cloud_shadow')
    cloud = get_qa_bits(QA, 5, 5, 'cloud')
    cirrus_detected = get_qa_bits(QA, 9, 9, 'cirrus_detected')
    return image.updateMask(shadow.eq(0)).updateMask(cloud.eq(0).updateMask(cirrus_detected.eq(0)))


def ls57mask(img):
    sr_bands = img.select('B1', 'B2', 'B3', 'B4', 'B5', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_sel = img_masked.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'], ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    mask_mult = mask_sel.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def ls8mask(img):
    sr_bands = img.select('B2', 'B3', 'B4', 'B5', 'B6', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_mult = img_masked.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def ndvi5():
    l = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').map(lambda x: x.select().addBands(
        x.normalizedDifference(['B4', 'B3'])))
    return l


def ndvi7():
    l = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').map(lambda x: x.select().addBands(
        x.normalizedDifference(['B4', 'B3'])))
    return l


def ndvi8():
    l = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').map(lambda x: x.select().addBands(
        x.normalizedDifference(['B5', 'B4'])))
    return l


def ls5_edge_removal(lsImage):
    inner_buffer = lsImage.geometry().buffer(-3000)
    buffer = lsImage.clip(inner_buffer)
    return buffer


def period_stat(collection, start, end):
    c = collection.filterDate(start, end)
    return c.reduce(
        ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(),
                                  sharedInputs=True))


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
    request_band_extract(file_prefix='RDGP_19FEB2020', points_layer=POINTS_MT,
                         region=ROI, filter_bounds=False)
# ========================= EOF ====================================================================
