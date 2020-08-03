import os
import ee
from pprint import pprint
from datetime import datetime

import tensorflow as tf
from map.openet.collection import get_target_dates, Collection
from map.trainer.shapefile_meta import shapefile_counts, SHP_TO_YEAR_AND_COUNT
from map.trainer.tc_ee import preprocess_data
from map.trainer.tc_ee import create_class_labels as ccl
from map.trainer.tc_ee import temporally_filter_features as tff

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

GS_BUCKET = 'wudr'
KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]

BOUNDARIES = 'users/dgketchum/boundaries'
MGRS = os.path.join(BOUNDARIES, 'MGRS_TILE')
MT = os.path.join(BOUNDARIES, 'MT')

# Tommy's training data
# gs://ee-irrigation-mapping/train-data-july9_1-578/
# gs://ee-irrigation-mapping/test-data-july23/
# gs://ee-irrigation-mapping/validation-data-july23/


def create_class_labels(shp_to_fc_):

    labels = ee.ImageCollection('USDA/NASS/CDL') \
                .filter(ee.Filter.date('2018-01-01', '2018-12-31')) \
                .first().select('cultivated').rename('constant')

    return labels


def filter_features(polygon_ds, yr_, roi):
    polygon_to_fc = {}
    polygon_mapping = shapefile_counts()
    polygon_mapping = {'{}_MT'.format(k): v for k, v in polygon_mapping.items()}

    for shapefile in polygon_ds:
        is_temporal = True
        basename = os.path.basename(shapefile)
        feature_collection = ee.FeatureCollection(shapefile).filterBounds(roi)

        if 'irrigated' in basename and 'unirrigated' not in basename:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", yr_))

        elif 'fallow' in basename:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", yr_))

        else:
            is_temporal = False
            size = feature_collection.size().getInfo()
            polygon_to_fc[shapefile] = (size, feature_collection)

        if is_temporal:
            valid_years = list(dict(polygon_mapping[basename].items()).keys())
            if yr_ in valid_years:
                size = feature_collection.size().getInfo()
                polygon_to_fc[shapefile] = (size, feature_collection)

    return polygon_to_fc


def get_sr_stack(yr, s, e, interval, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr, e, 1)
    target_interval = interval
    interp_days = 32

    target_dates = get_target_dates(s, e, interval_=target_interval)

    model_obj = Collection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        geometry=geo_,
        cloud_cover_max=60)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    scenes = model_obj.scenes(variables_)

    interp = interpolated.sort('system:time_start').toBands()
    # TODO: map rename function over collection to append date to var name
    return interp, interp.bandNames().getInfo()


def extract_data_over_shapefiles(label_polygons, year, out_folder,
                                 get_points=False, n_shards=4):

    s, e, interv_ = 5, 6, 60

    roi = ee.FeatureCollection(MT).geometry()

    # image_stack, bands = get_sr_stack(year, s, e, interv_, geo_=roi)
    # image_stack = preprocess_data(year).toBands()
    # TODO: check Tommy's image stack (fresh), try to get sr interp with tff
    features = image_stack.bandNames().getInfo() + ['constant']

    shp_to_fc = tff(label_polygons, year)
    class_labels = create_class_labels(shp_to_fc).float()

    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = dict(zip(features, columns))
    pprint(feature_dict)

    data_stack = ee.Image.cat([image_stack, class_labels]).float()
    kernel = ee.Kernel.square(KERNEL_SIZE / 2)
    data_stack = data_stack.neighborhoodToArray(kernel)
    for asset, fc_ in shp_to_fc.items():

        n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(asset)][year]

        polygons = fc_.toList(fc_.size())
        out_class_label = os.path.basename(asset)
        out_filename = '{}_{}_means'.format(out_folder, out_class_label)

        geometry_sample = ee.ImageCollection([])
        for i in range(n_features):

            sample = data_stack.sample(
                region=ee.Feature(polygons.get(i)).geometry(),
                scale=30,
                numPixels=1,
                tileScale=16,
                dropNulls=False)

            geometry_sample = geometry_sample.merge(sample)
            if (i + 1) % n_shards == 0:
                task = ee.batch.Export.table.toCloudStorage(
                    collection=geometry_sample,
                    description=out_filename,
                    bucket=GS_BUCKET,
                    fileNamePrefix=out_filename,
                    fileFormat='TFRecord',
                    selectors=features)

                task.start()
                print(year, asset)
                exit()

if __name__ == '__main__':
    ee.Initialize(use_cloud_api=True)

    root = 'users/dgketchum/training_polygons/'
    test = ['irrigated_test', 'fallow_test', 'uncultivated_test',
            'unirrigated_test', 'wetlands_test']
    test = [root + t for t in test]
    train = ['irrigated_train', 'fallow_train', 'uncultivated_train',
             'unirrigated_train', 'wetlands_train']
    train = [root + t for t in train]

    extract_data_over_shapefiles(train, year=2010, out_folder=GS_BUCKET, get_points=False)
# ========================= EOF ====================================================================
