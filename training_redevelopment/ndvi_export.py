import os
import sys
import time
import random

import ee
import geopandas as gpd

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)


def landsat_c2_sr(input_img):
    # credit: cgmorton; https://github.com/Open-ET/openet-core-beta/blob/master/openet/core/common.py

    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                    'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))

    prep_image = input_img \
        .select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                   0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _cloud_mask(i):
        qa_img = i.select(['QA_PIXEL'])
        cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))
        sat_mask = i.select(['QA_RADSAT']).gt(0)
        cloud_mask = cloud_mask.Or(sat_mask)

        cloud_mask = cloud_mask.Not().rename(['cloud_mask'])

        return cloud_mask

    mask = _cloud_mask(input_img)

    image = prep_image.updateMask(mask).copyProperties(input_img, ['system:time_start'])

    return image


def landsat_masked(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l4_coll = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l9_coll = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l9_coll).merge(l5_coll).merge(l4_coll))

    return lsSR_masked


def clustered_sample_ndvi(points, bucket=None, debug=False, check_dir=None, extract_random_modern=False):

    if extract_random_modern:
        points['YEAR'] = points['YEAR'].apply(lambda x: random.randint(2017, 2024))
        file_prefix = 'irrmapper_redev/ndvi_modern'
    else:
        file_prefix = 'irrmapper_redev/ndvi'

    mgrs_tiles = points['MGRS_TILE'].unique()

    for tile in mgrs_tiles:

        tile_df = points[points['MGRS_TILE'] == tile]
        states = tile_df['STUSPS'].unique()

        for state in states:

            state_df = tile_df[tile_df['STUSPS'] == state]
            years = state_df['YEAR'].unique()

            for year in years:

                if year != 2024:
                    continue

                year_df = state_df[state_df['YEAR'] == year]

                feature_coll = ee.FeatureCollection(year_df.__geo_interface__)

                first, bands = True, None
                selectors = ['FID', 'POINT_TYPE', 'YEAR', 'MGRS_TILE', 'STUSPS']

                desc = f'ndvi_{tile}_{state}_{year}'

                if check_dir:
                    f = os.path.join(check_dir, f'{desc}.csv')
                    if os.path.exists(f):
                        continue

                coll = landsat_masked(year, feature_coll).map(lambda x: x.normalizedDifference(['B5', 'B4']))

                ndvi_scenes = coll.aggregate_histogram('system:index').getInfo()

                for img_id in ndvi_scenes:

                    splt = img_id.split('_')
                    _name = '_ '.join(splt[-3:])

                    selectors.append(_name)

                    nd_img = coll.filterMetadata('system:index', 'equals', img_id).first().rename(_name)

                    nd_img = nd_img.clip(feature_coll.geometry())

                    if first:
                        bands = nd_img
                        first = False
                    else:
                        bands = bands.addBands([nd_img])

                if debug:
                    fc = ee.FeatureCollection([feature_coll.filterMetadata('FID', 'equals', 2).first()])
                    data = bands.reduceRegions(collection=fc,
                                               reducer=ee.Reducer.mean(),
                                               scale=30).getInfo()
                    print(data['features'])

                try:
                    data = bands.reduceRegions(collection=feature_coll,
                                               reducer=ee.Reducer.mean(),
                                               scale=30)
                except AttributeError as exc:
                    print(f'{desc} raised {exc}')
                    continue

                task = ee.batch.Export.table.toCloudStorage(
                    data,
                    description=desc,
                    bucket=bucket,
                    fileNamePrefix=f'{file_prefix}/{desc}',
                    fileFormat='CSV',
                    selectors=selectors)

                try:
                    task.start()
                except ee.ee_exception.EEException as e:
                    print('{}, waiting on '.format(e), desc, '......')
                    time.sleep(600)
                    task.start()

                print(desc)


def is_authorized():
    try:
        ee.Initialize(project='ee-dgketchum')
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/irrmapper'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/irrmapper'

    is_authorized()
    bucket_ = 'wudr'

    pt_wgs = os.path.join(d, 'EE_extracts/point_shp', 'state_wgs_mgrs')
    shp = os.path.join(pt_wgs, 'master_training_points.shp')
    df = gpd.read_file(shp)

    check_d = '/data/ssd2/irrmapper/ndvi/past_training/'
    # clustered_sample_ndvi(df, bucket=bucket_, check_dir=check_d)

    check_d = '/data/ssd2/irrmapper/ndvi/modern_update/'
    clustered_sample_ndvi(df, bucket=bucket_, check_dir=check_d, extract_random_modern=True)

# ========================= EOF ====================================================================
