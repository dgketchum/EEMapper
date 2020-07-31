import os
import ee
import time
from collections import OrderedDict
from pprint import pprint
from datetime import datetime

import tensorflow as tf
from map.openet.collection import get_target_bands, get_target_dates, Collection
from map.shapefile_meta import shapefile_counts
from map.ee_utils import assign_class_code

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
    class_labels = ee.Image(0).byte()

    for asset, (n, _fc) in shp_to_fc_.items():
        class_labels = class_labels.paint(_fc, assign_class_code(asset) + 1)

    labels = class_labels.updateMask(class_labels)

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
    interp_days = 0

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

    interp = interpolated.sort('system:time_start')
    # TODO: map rename function over collection to append date to var name
    return interp


def extract_data_over_shapefiles(label_polygons, year, out_folder,
                                 points_to_extract=None, n_shards=4):

    test_xy = [-121.5265, 38.7399]
    test_point = ee.Geometry.Point(test_xy)

    # study_area = ee.Geometry.Rectangle(-122.00, 38.60, -121.00, 39.0)
    study_area = ee.Geometry.Rectangle(
        test_xy[0] - 0.04, test_xy[1] - 0.02,
        test_xy[0] + 0.04, test_xy[1] + 0.02)
    study_region = study_area.bounds(1, 'EPSG:4326')

    s, e, interv_ = 5, 8, 60

    # roi = ee.FeatureCollection(MT).geometry()
    roi = study_region

    image_stack = get_sr_stack(year, s, e, interv_, geo_=roi)

    renames = ['{}_{}'.format(x[9:], x[:8]) for x in names]
    out_filename = '{}_s{}e{}int{}'.format('irri_test', s, e, interv_)

    asset_root = 'users/dgketchum/training_polygons'
    task = ee.batch.Export.image.toAsset(
        image=image_stack,
        description='{}_{}'.format(year, out_filename),
        assetId=os.path.join(asset_root, '{}_{}'.format(out_filename, year)),
        region=study_area,
        scale=30,
        maxPixels=1e13)

    task.start()
    print(year)
    exit()


    features = bands + ['constant']
    spect = {k: tf.io.FixedLenFeature(shape=[256, 256],
                                      dtype=tf.float32,
                                      default_value=None) for k in features}
    pprint(spect)

    shp_to_fc = filter_features(label_polygons, year, roi)
    class_labels = create_class_labels(shp_to_fc)

    data_stack = ee.Image.cat([image_stack, class_labels]).float()
    kernel = ee.Kernel.square(KERNEL_SIZE / 2)
    data_stack = data_stack.neighborhoodToArray(kernel)

    if points_to_extract is None:
        for asset, (n_features, fc_) in shp_to_fc.items():

            polygons = fc_.toList(fc_.size())
            out_class_label = os.path.basename(asset)
            out_filename = '{}_{}_s{}e{}int{}'.format(out_folder, out_class_label,
                                                      s, e, interv_)

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
    train = [root + t + '_MT' for t in train]

    extract_data_over_shapefiles(train, year=2010, out_folder=GS_BUCKET)
# ========================= EOF ====================================================================
