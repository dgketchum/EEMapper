import os
import sys
from datetime import datetime, date
from numpy import ceil, linspace
from pprint import pprint

import ee

sys.path.insert(0, os.path.abspath('..'))
from map.assets import list_assets
from map.ee_utils import get_world_climate, landsat_composites, landsat_masked, daily_landsat
from map.tables import SELECT
from shape_ops import count_points

sys.setrecursionlimit(2000)

GEO_DOMAIN = 'users/dgketchum/boundaries/western_states_expanded_union'
BOUNDARIES = 'users/dgketchum/boundaries'
ASSET_ROOT = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper_test'
IRRIGATION_TABLE = 'users/dgketchum/western_states_irr/NV_agpoly'
FILTER_TARGET = 'users/dgketchum/to_filter/MT_2012'
RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper_RF'

RF_TRAINING_DATA = 'projects/ee-dgketchum/assets/bands/bands_3DEC2020_COWY'
RF_TRAINING_POINTS = 'projects/ee-dgketchum/assets/points/train_pts_7DEC2020_CIMOW'

HUC_6 = 'users/dgketchum/usgs_wbd/huc6_semiarid_clip'
HUC_8 = 'users/dgketchum/usgs_wbd/huc8_semiarid_clip'
COUNTIES = 'users/dgketchum/boundaries/western_counties'
MT_BASINS = 'users/dgketchum/boundaries/MT_Admin_Basins'

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT',
                 'WA', 'WY']

other = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

IRR = {'ND': [2013, 2017],
       'SD': [2017, 2019],
       'NE': [2012, 2013, 2016],
       'KS': [2018, 2019],
       'OK': [2011, 2019]}

DRY = {
    # 'ND': [2006, 2012],
    # 'SD': [2006, 2012],
    # 'NE': [2002, 2012],
    # 'KS': [2002, 2012],
    'OK': [2001, 2011],
    'TX': [2011, 2020]}

# list of years we have verified irrigated fields
YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019]

TEST_YEARS = [2005]
ALL_YEARS = [x for x in range(1997, 2018)]


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
        out_desc = '{}_area_{}'.format(description, yr)
        task = ee.batch.Export.table.toCloudStorage(
            reduce,
            description=out_desc,
            bucket='wudr',
            fileNamePrefix=out_desc,
            fileFormat='CSV')
        task.start()
        print(out_desc)


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
        for yr in range(2011, 2021):
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


def export_raster():
    target_bn = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper_RF'
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    for yr in range(1987, 2022):
        images = [x for x in image_list if x.endswith(str(yr))]

        coll = ee.ImageCollection(images)

        _properties = {'image_id': 'IrrMapper_RF_{}'.format(yr), 'system:time_start': ee.Date.fromYMD(yr, 1, 1),
                       'system:time_end': ee.Date.fromYMD(yr, 12, 31)}

        img = coll.select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0])
        img = img.updateMask(img.neq(0)).rename('classification').set(_properties)

        id_ = os.path.join(target_bn, '{}'.format(yr))
        task = ee.batch.Export.image.toAsset(
            image=img,
            description='IrrMapper_RF_{}'.format(yr),
            assetId=id_,
            pyramidingPolicy={'.default': 'mode'},
            scale=30,
            maxPixels=1e13)
        task.start()
        print(yr)


def export_special(roi, description):
    fc = ee.FeatureCollection(roi)
    roi_mask = fc.geometry().bounds().getInfo()['coordinates']

    for year in [str(x) for x in range(1987, 2021)]:
        target = ee.Image(os.path.join(RF_ASSET, year))
        target = target.select('classification')

        task = ee.batch.Export.image.toDrive(
            target,
            description='IrrMapper_{}_{}'.format(description, year),
            region=roi_mask,
            scale=30,
            maxPixels=1e13,
            folder='UCRB',
            crs='EPSG:5070'
        )
        task.start()
        print(year)


def export_classification(out_name, table, asset_root, region, export='asset'):
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
    fc = ee.FeatureCollection(table)
    roi = ee.FeatureCollection(region)
    mask = roi.geometry().bounds().getInfo()['coordinates']

    classifier = ee.Classifier.randomForest(
        numberOfTrees=100,
        variablesPerSplit=0,
        minLeafPopulation=1,
        outOfBagMode=False).setOutputMode('CLASSIFICATION')

    input_props = fc.first().propertyNames().remove('YEAR').remove('POINT_TYPE').remove('system:index')

    trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

    for yr in ALL_YEARS:
        input_bands = stack_bands(yr, roi)
        annual_stack = input_bands.select(input_props)
        classified_img = annual_stack.classify(trained_model).int().set({
            'system:index': ee.Date('{}-01-01'.format(yr)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-01-01'.format(yr)).millis(),
            'system:time_end': ee.Date('{}-12-31'.format(yr)).millis(),
            'image_name': out_name,
            'class_key': '0: irrigated, 1: rainfed, 2: uncultivated, 3: wetland'})

        if export == 'asset':
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                assetId=os.path.join(asset_root, '{}_{}'.format(out_name, yr)),
                region=mask,
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)

        elif export == 'cloud':
            task = ee.batch.Export.image.toCloudStorage(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                bucket='wudr',
                fileNamePrefix='{}_{}'.format(yr, out_name),
                region=mask,
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
    for yr in YEARS:
        stack = stack_bands(yr, roi)

        if filter_bounds:
            plots = plots.filterBounds(roi)

        filtered = plots.filter(ee.Filter.eq('YEAR', yr))

        plot_sample_regions = stack.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR'],
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print(yr)
        exit()


def stack_bands(yr, roi):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param roi:
    :return:
    """

    water_year_start = '{}-10-01'.format(yr - 1)

    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-01'.format(yr)
    summer_s, summer_e = '{}-07-01'.format(yr), '{}-09-01'.format(yr)
    fall_s, fall_e = '{}-09-01'.format(yr), '{}-12-31'.format(yr)

    periods = [('cy', winter_s, fall_e),
               ('1', spring_s, spring_e),
               ('2', late_spring_s, late_spring_e),
               ('3', summer_s, summer_e),
               ('4', fall_s, fall_e)]

    first = True
    for name, start, end in periods:
        bands = landsat_composites(yr, start, end, roi, name)
        if first:
            input_bands = bands
            proj = bands.select('B2_cy').projection().getInfo()
            first = False
        else:
            input_bands = input_bands.addBands(bands)

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

    cdl_cult = ee.Image('USDA/NASS/CDL/2017').select('cultivated'). \
        remap([1, 2], [0, 1]).reproject(crs=proj['crs'], scale=30).rename('cdlclt')

    cdl_crop = ee.Image('USDA/NASS/CDL/2017').select('cropland').reproject(crs=proj['crs'],
                                                                           scale=30).rename('cdlcrp')

    gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
    occ_pos = gsw.select('occurrence').gt(0)
    water = occ_pos.unmask(0).rename('gsw')

    static_input_bands = static_input_bands.addBands([nlcd, cdl_cult, cdl_crop, water])

    input_bands = input_bands.addBands(static_input_bands).clip(roi)

    # standardize names to match EE javascript output
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
    pts = 'projects/ee-dgketchum/assets/points/train_pts_18JAN2021'
    request_band_extract('bands_18JAN2021', pts, GEO_DOMAIN, filter_bounds=False)
    # csv = 'projects/ee-dgketchum/assets/bands/state/bands_{}_10DEC2020'.format(s)
    # geo = os.path.join(BOUNDARIES, s)
    # export_classification(out_name='IM_{}'.format(s), table=csv, asset_root=ASSET_ROOT, region=geo)
# ========================= EOF ====================================================================
