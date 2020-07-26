import ee

ee.Initialize()
import tensorflow as tf
import time
import os
import numpy as np
from random import shuffle

from map.ee_utils import assign_class_code, daily_landsat

LC8_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
LC7_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
LC5_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
STD_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'wudr'


def create_class_labels(shapefile_to_feature_collection):
    class_labels = ee.Image(0).byte()
    for shapefile, feature_collection in shapefile_to_feature_collection.items():
        class_labels = class_labels.paint(feature_collection, assign_class_code(shapefile) + 1)
    return class_labels.updateMask(class_labels)


def extract_data_over_shapefiles(mask_shapefiles, year,
                                 out_folder, points_to_extract=None, n_shards=10):


    # Features dict for TFRecord
    features = [feat['id'] for feat in image_stack.getInfo()['bands']]
    # Add in the mask raster
    features = features + ['constant']
    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = dict(zip(features, columns))

    shapefile_to_feature_collection = temporally_filter_features(mask_shapefiles, year)
    if points_to_extract is not None:
        shapefile_to_feature_collection['points'] = points_to_extract
    class_labels = create_class_labels(shapefile_to_feature_collection)
    data_stack = ee.Image.cat([image_stack, class_labels]).float()
    data_stack = data_stack.neighborhoodToArray(KERNEL)

    if points_to_extract is None:
        # This extracts data over every polygon that
        # was passed in
        for shapefile, feature_collection in shapefile_to_feature_collection.items():

            polygons = feature_collection.toList(feature_collection.size())
            # Frustratingly using getInfo() for sizes causes
            # a EE out of user memory error, because it has to actually
            # request something from the server
            try:
                n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(shapefile)][year]
            except KeyError:
                n_features = SHP_TO_COUNT[os.path.basename(shapefile)]

            out_class_label = os.path.basename(shapefile)
            out_filename = out_class_label + "_" + str(year)
            geometry_sample = ee.ImageCollection([])
            indices = np.arange(n_features)
            print(out_class_label, year)
            indices = [int(i) for i in indices]
            feature_count = 0
            for i, idx in enumerate(indices):
                sample = data_stack.sample(
                    region=ee.Feature(polygons.get(idx)).geometry(),
                    scale=30,
                    numPixels=1,
                    tileScale=8
                )
                geometry_sample = geometry_sample.merge(sample)
                feature_count += 1
                if (feature_count + 1) % n_shards == 0:
                    task = ee.batch.Export.table.toCloudStorage(
                        collection=geometry_sample,
                        description=out_filename + str(time.time()),
                        bucket=GS_BUCKET,
                        fileNamePrefix=out_folder + out_filename + str(time.time()),
                        fileFormat='TFRecord',
                        selectors=features
                    )
                    try:
                        task.start()
                    except ee.ee_exception.EEException:
                        print('waiting to export, sleeping for 5 minutes. Holding at\
                                {} {}, index {}'.format(year, shapefile, i))
                        time.sleep(3000)
                        task.start()
                    geometry_sample = ee.ImageCollection([])
            # take care of leftovers
            print('n_extracted:', feature_count)
            task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                description=out_filename + str(time.time()),
                bucket=GS_BUCKET,
                fileNamePrefix=out_folder + out_filename + str(time.time()),
                fileFormat='TFRecord',
                selectors=features
            )
            task.start()
    else:
        # just extract data at points
        polygons = ee.FeatureCollection(points_to_extract)
        polygons = polygons.toList(polygons.size())
        n_features = polygons.size().getInfo()  # see if this works
        print(n_features, points_to_extract)
        geometry_sample = ee.ImageCollection([])
        out_filename = str(year)
        n_extracted = 0
        for i in range(n_features):
            sample = data_stack.sample(
                region=ee.Feature(polygons.get(i)).geometry(),
                scale=30,
                numPixels=1,
                tileScale=8
            )
            geometry_sample = geometry_sample.merge(sample)
            if (i + 1) % n_shards == 0:
                n_extracted += geometry_sample.size().getInfo()
                task = ee.batch.Export.table.toCloudStorage(
                    collection=geometry_sample,
                    bucket=GS_BUCKET,
                    description=out_filename + str(time.time()),
                    fileNamePrefix=out_folder + out_filename + str(time.time()),
                    fileFormat='TFRecord',
                    selectors=features
                )
                task.start()
                geometry_sample = ee.ImageCollection([])
        # take care of leftovers
        task = ee.batch.Export.table.toCloudStorage(
            collection=geometry_sample,
            bucket=GS_BUCKET,
            description=out_filename + str(time.time()),
            fileNamePrefix=out_folder + out_filename + str(time.time()),
            fileFormat='TFRecord',
            selectors=features
        )
        task.start()
        print(n_extracted, year)


if __name__ == '__main__':

    test_root = 'users/tcolligan0/test/'
    test = ['fallow_test', 'irrigated_test', 'uncultivated_test',
            'unirrigated_test', 'wetlands_test']
    test = [test_root + t for t in test]

    train = ['fallow_train', 'irrigated_train', 'uncultivated_train',
             'unirrigated_train', 'wetlands_train']
    train_root = 'users/tcolligan0/train/'
    train = [train_root + t for t in train]

    validation = ['fallow_validation', 'irrigated_validation', 'uncultivated_validation',
                  'unirrigated_validation', 'wetlands_validation']
    validation_root = 'users/tcolligan0/validation/'
    validation = [validation_root + t for t in validation]

    test_pts = 'users/tcolligan0/points_to_extract/test_regions_points'
    train_pts = 'users/tcolligan0/points_to_extract/train_regions_points'
    validation_pts = 'users/tcolligan0/points_to_extract/validation_regions_points'

    years = range(2003, 2015)
    # years = range(2013, 2016)
    # years = [2015]
    extract_test = False
    extract_train = True
    extract_validation = False
    if extract_test:
        for year in years:
            extract_data_over_shapefiles(test, year,
                                         out_folder='test-data-june30-578/', points_to_extract=test_pts)
    if extract_train:
        for year in years:
            extract_data_over_shapefiles(train, year,
                                         out_folder='train-data-july9_1-578/')  # , points_to_extract=train_pts)
    if extract_validation:
        for year in years:
            extract_data_over_shapefiles(validation, year,
                                         out_folder='validation-data-june30-578/', points_to_extract=validation_pts)

# ========================= EOF ====================================================================
