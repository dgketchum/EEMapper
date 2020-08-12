import ee

ee.Initialize()
import tensorflow as tf
import time
import os
from pprint import pprint
import numpy as np
from collections import OrderedDict
from datetime import datetime
from map.openet.collection import get_target_dates, Collection, get_target_bands
from map.trainer import extract_utils
from map.trainer.shapefile_meta import SHP_TO_YEAR_AND_COUNT

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'wudr'

BOUNDARIES = 'users/dgketchum/boundaries'
MGRS = os.path.join(BOUNDARIES, 'MGRS_TILE')
MT = os.path.join(BOUNDARIES, 'MT')

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']


def test_geo():
    fc = ee.Geometry([[-112.19186229407349,47.81781050704002],
                      [-112.0552198380188,47.81781050704002],
                      [-112.0552198380188,47.870346541256765],
                      [-112.19186229407349,47.870346541256765],[]])


def get_ancillary(yr):
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation') \
        .resample('bilinear').rename(['elev'])

    coords = ned.pixelLonLat().rename(['lon', 'lat'])
    return terrain, coords


def get_sr_stack(yr, s, e, interval, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr + 1, e, 1)
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

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
    interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
    return interp, target_rename


def extract_test_patches(mask_shapefiles, year,
                         out_folder, patch_shapefile):
    s, e, interval_ = 1, 1, 30
    roi = ee.FeatureCollection(MGRS).filter(ee.Filter.eq('MGRS_TILE', '12TVT')).geometry()
    image_stack, features = get_sr_stack(year, s, e, interval_, roi)

    features = features + ['lat', 'lon', 'elev', 'irr']
    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = dict(zip(features, columns))
    pprint(feature_dict)

    shapefile_to_feature_collection = extract_utils.temporally_filter_features(mask_shapefiles, year)
    coords, irr = extract_utils.create_class_labels(shapefile_to_feature_collection)
    data_stack = ee.Image.cat([image_stack, coords, irr]).float()

    patches = ee.FeatureCollection(patch_shapefile)
    patches = patches.toList(patches.size())

    out_filename = str(year)
    for idx in range(patches.size().getInfo()):
        patch = ee.Feature(patches.get(idx))
        task = ee.batch.Export.image.toCloudStorage(
            image=data_stack,
            bucket=GS_BUCKET,
            description=out_filename + str(time.time()),
            fileNamePrefix=out_folder + out_filename + str(time.time()),
            fileFormat='TFRecord',
            region=patch.geometry(),
            scale=30,
            formatOptions={'patchDimensions': KERNEL_SIZE,
                           'compressed': True})
        task.start()


def extract_data_over_shapefiles(mask_shapefiles, year,
                                 out_folder, points_to_extract=None,
                                 n_shards=10):
    roi = ee.FeatureCollection(MGRS).filter(ee.Filter.eq('MGRS_TILE', '12TVT')).geometry()
    # roi = ee.FeatureCollection(MT).geometry()

    s, e, interval_ = 1, 1, 30

    image_stack, features = get_sr_stack(year, s, e, interval_, roi)

    features = features + ['lat', 'lon', 'elev', 'slope', 'aspect', 'irr']

    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = OrderedDict(zip(features, columns))
    # pprint(feature_dict)

    shapefile_to_feature_collection = extract_utils.temporally_filter_features(mask_shapefiles, year)
    if points_to_extract is not None:
        shapefile_to_feature_collection['points'] = points_to_extract

    irr = extract_utils.create_class_labels(shapefile_to_feature_collection)
    terrain_, coords_ = get_ancillary(year)
    data_stack = ee.Image.cat([image_stack, terrain_, coords_, irr]).float()
    data_stack = data_stack.neighborhoodToArray(KERNEL)

    if points_to_extract is None:
        for shapefile, feature_collection in shapefile_to_feature_collection.items():
            polygons = feature_collection.toList(feature_collection.size())
            n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(shapefile)][year]
            out_class_label = os.path.basename(shapefile)
            out_filename = out_class_label + "_sr1_30d_" + str(year)
            geometry_sample = ee.ImageCollection([])
            if not n_features:
                continue
            n = 10
            if n_features < n:
                n = n_features
            indices = np.random.choice(n_features, size=n)
            indices = [int(i) for i in indices]
            print(out_class_label, year, n_features, len(indices))
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
                        exit()
                        continue
                    except ee.ee_exception.EEException:
                        print('waiting to export, sleeping for 50 minutes. Holding at\
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
    root = 'users/dgketchum/training_polygons/'
    test = ['irrigated_test', 'fallow_test',
            'unirrigated_test', 'wetlands_test']
    test = [root + t for t in test]
    train = ['irrigated_train', 'fallow_train',
             'unirrigated_train', 'wetlands_train']
    train = [root + t for t in train]

    year = 2010

    extract_data_over_shapefiles(train, year, out_folder=GS_BUCKET)
