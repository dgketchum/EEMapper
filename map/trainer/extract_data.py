import ee

ee.Initialize()
import tensorflow as tf
import time
import os
from collections import OrderedDict
from pprint import pprint
import numpy as np
from datetime import datetime
from map.openet.collection import get_target_dates, Collection, get_target_bands
# from map.trainer import extract_utils
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

root = 'users/dgketchum/training_points/'
train = ['irrigated', 'dryland', 'uncultivated']
AVAILABLE_TRAIN = [root + t for t in sorted(train)]


def test_geo():
    test_geo_ = ee.Geometry.Rectangle(-112.23391251229945, 47.9105255993934,
                                      -112.02860550546352, 47.81654904556808)
    test_geo_ = test_geo_.bounds(1, 'EPSG:4326')
    return test_geo_


def temporally_filter_features(year):
    shapefile_to_feature_collection = {}
    for shapefile in AVAILABLE_TRAIN:
        temporal_ = True
        bs = os.path.basename(shapefile)
        feature_collection = ee.FeatureCollection(shapefile)
        if 'irrigated' in bs:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", year))
        elif 'fallow' in bs:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", year))
        else:
            temporal_ = False
            shapefile_to_feature_collection[shapefile] = feature_collection

        if temporal_:
            shapefile_to_feature_collection[shapefile] = feature_collection

    return shapefile_to_feature_collection


def create_class_labels(shapefile_to_feature_collection):
    class_labels = ee.Image(0).byte()
    for shapefile, feature_collection in shapefile_to_feature_collection.items():
        class_labels = class_labels.paint(feature_collection,
                                          assign_class_code(shapefile) + 1)

    label = class_labels.updateMask(class_labels).rename('irr')

    return label


def assign_class_code(shapefile_path):
    shapefile_path = os.path.basename(shapefile_path)
    if 'irrigated' in shapefile_path:
        return 1
    if 'fallow' in shapefile_path:
        return 2
    if 'dryland' in shapefile_path:
        return 3
    if 'uncultivated' in shapefile_path:
        return 4
    if 'points' in shapefile_path:
        # annoying workaround for earthengine
        return 10
    else:
        raise NameError('shapefile path {} isn\'t named in assign_class_code'.format(shapefile_path))


def get_ancillary(yr):
    cdl = ee.ImageCollection('USDA/NASS/CDL') \
        .filter(ee.Filter.date('{}-01-01'.format(yr), '{}-12-31'.format(yr))) \
        .first().select('cultivated').rename('cdl')

    coords = cdl.pixelLonLat().rename(['lon', 'lat'])

    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect') \
        .resample('bilinear').rename(['elev', 'slope', 'aspect'])

    return terrain, coords, cdl


def get_sr_stack(yr, s, e, interval, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr + 1, e, 1)
    target_interval = interval
    interp_days = 0

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
    interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
    return interp, target_rename


def extract_data_at_points(point_layer, year,
                           out_folder, n_shards=3):

    # roi = ee.FeatureCollection(MGRS).filter(ee.Filter.eq('MGRS_TILE', '12TVT')).geometry()
    roi = ee.FeatureCollection(MT).geometry()
    pts = test_geo()

    s, e, interval_ = 1, 1, 30

    image_stack, features = get_sr_stack(year, s, e, interval_, roi)

    features = features + ['lat', 'lon', 'elev', 'slope', 'aspect', 'irr', 'cdl']

    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = OrderedDict(zip(features, columns))
    # pprint(feature_dict)

    shapefile_to_feature_collection = temporally_filter_features(year)

    irr = create_class_labels(shapefile_to_feature_collection)
    terrain_, coords_, cdl_ = get_ancillary(year)
    data_stack = ee.Image.cat([image_stack, terrain_, coords_, cdl_, irr]).float()
    data_stack = data_stack.neighborhoodToArray(KERNEL)

    # just extract data at points
    print(point_layer)
    points = ee.FeatureCollection(point_layer).filterBounds(pts)
    if 'irrigated' in point_layer:
        points = points.filter(ee.Filter.eq('YEAR', year))
    pprint(points.first().getInfo())
    points = points.toList(points.size())
    n_features = points.size().getInfo()
    pprint('{} features in geo in {}'.format(n_features, year))

    geometry_sample = ee.ImageCollection([])
    out_class_label = os.path.basename(point_layer)
    out_filename = out_class_label + "_sr1_30d_" + str(year)
    n_extracted = 0
    for i in range(n_features):
        sample = data_stack.sample(
            region=ee.Feature(points.get(i)).geometry(),
            scale=30,
            numPixels=1,
            tileScale=8
        )
        geometry_sample = geometry_sample.merge(sample)
        if (i + 1) % n_shards == 0:
            # n_extracted += geometry_sample.size().getInfo()
            # print('{} extracted'.format(n_extracted))
            task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                bucket=GS_BUCKET,
                description=out_filename + str(time.time()),
                fileNamePrefix=out_folder + out_filename + str(time.time()),
                fileFormat='TFRecord',
                selectors=features
            )
            task.start()
            print('{}th element out'.format(i))
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
    # print(n_extracted, year)


if __name__ == '__main__':

    year = 2013
    for t in AVAILABLE_TRAIN:
        extract_data_at_points(t, year, out_folder=GS_BUCKET)
