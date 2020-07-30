import os
import ee
# import tensorflow as tf
import time
# from pprint import pprint
from datetime import datetime
from map.openet.collection import get_target_bands, get_target_dates, Collection
from map.shapefile_meta import shapefile_counts
from map.ee_utils import assign_class_code

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

GS_BUCKET = 'wudr'
KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]

MGRS = 'users/dgketchum/boundaries/MGRS_TILE'


# Tommy's training data
# gs://ee-irrigation-mapping/train-data-july9_1-578/
# gs://ee-irrigation-mapping/test-data-july23/
# gs://ee-irrigation-mapping/validation-data-july23/


def create_class_labels(shp_to_fc_):
    class_labels = ee.Image(0).byte()

    for asset, (n, _fc) in shp_to_fc_.items():
        class_labels = class_labels.paint(_fc, assign_class_code(asset) + 1)

    return class_labels.updateMask(class_labels)


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


def get_sr_stack(yr, geo_):
    s = datetime(yr, 5, 1)
    e = datetime(yr, 10, 1)
    target_interval = 60
    interp_days = 32

    target_dates = get_target_dates(s, e, interval_=target_interval)

    model_obj = Collection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        geometry=geo_,
        cloud_cover_max=100)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
    interp = interpolated.toBands().rename(target_rename)
    return interp, target_rename


def extract_data_over_shapefiles(label_polygons, year, out_folder,
                                 points_to_extract=None, n_shards=1):

    roi = ee.FeatureCollection(MGRS).filter(ee.Filter.eq('MGRS_TILE', '12TVT')).geometry()

    # test_xy = [-111.8148, 47.52720]
    # roi = ee.Geometry.Rectangle(
    #     test_xy[0] - 0.15, test_xy[1] - 0.15,
    #     test_xy[0] + 0.15, test_xy[1] + 0.15)
    # roi = roi.bounds(1, 'EPSG:4326')

    image_stack, bands = get_sr_stack(year, geo_=roi)
    features = bands + ['irr']

    shp_to_fc = filter_features(label_polygons, year, roi)
    class_labels = create_class_labels(shp_to_fc)

    data_stack = ee.Image.cat([image_stack, class_labels]).float()
    # data_stack = image_stack.addBands([class_labels])
    kernel = ee.Kernel.square(KERNEL_SIZE / 2)
    data_stack = data_stack.neighborhoodToArray(kernel)

    if points_to_extract is None:
        for asset, (n_features, fc_) in shp_to_fc.items():

            polygons = fc_.toList(fc_.size())
            out_class_label = os.path.basename(asset)
            out_filename = out_folder + "_" + out_class_label + "_" + str(year)

            geometry_sample = ee.ImageCollection([])

            for i in range(n_features):

                sample = data_stack.sample(
                    region=ee.Feature(polygons.get(i)).geometry(),
                    scale=30,
                    numPixels=1,
                    tileScale=16)

                geometry_sample = geometry_sample.merge(sample)
                if (i + 1) % n_shards == 0:
                    task = ee.batch.Export.table.toCloudStorage(
                        collection=geometry_sample,
                        description=out_filename + str(time.time()),
                        bucket=GS_BUCKET,
                        fileNamePrefix=out_filename + str(time.time()),
                        fileFormat='TFRecord',
                        selectors=features)

                    task.start()
                    print(year, asset)
                    exit()


if __name__ == '__main__':
    ee.Initialize(use_cloud_api=True)

    boundary = 'users/dgketchum/boundaries/MT'

    root = 'users/dgketchum/training_polygons/'
    test = ['irrigated_test', 'fallow_test', 'uncultivated_test',
            'unirrigated_test', 'wetlands_test']
    test = [root + t for t in test]
    train = ['irrigated_train', 'fallow_train', 'uncultivated_train',
             'unirrigated_train', 'wetlands_train']
    train = [root + t + '_MT' for t in train]

    extract_data_over_shapefiles(train, year=2010, out_folder=GS_BUCKET)
# ========================= EOF ====================================================================
