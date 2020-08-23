import ee

ee.Initialize()
import tensorflow as tf
import time
import os
from pprint import pprint
import sys
sys.path.append('/home/dgketchum/PycharmProjects/EEMapper')

from collections import OrderedDict
from datetime import datetime
from map.openet.collection import get_target_dates, Collection, get_target_bands

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'wudr'

BOUNDARIES = 'users/dgketchum/boundaries'
MGRS = os.path.join(BOUNDARIES, 'MGRS_TILE')
TRAINING_GRID = 'users/dgketchum/grids/train'
MT = os.path.join(BOUNDARIES, 'MT')

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

CLASSES = ['uncultivated', 'dryland', 'fallow', 'irrigated']


YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997, 1998,
         2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]


def masks(roi, year_):
    root = 'users/dgketchum/training_polygons/'

    irr_mask = ee.FeatureCollection(os.path.join(root, 'irrigated_mask')).filterBounds(roi)
    irr_mask = irr_mask.filter(ee.Filter.eq("YEAR", year_))

    fallow_mask = ee.FeatureCollection(os.path.join(root, 'fallow_mask')).filterBounds(roi)
    fallow_mask = fallow_mask.filter(ee.Filter.eq("YEAR", year_))

    dryland_mask = ee.FeatureCollection(os.path.join(root, 'dryland_mask')).filterBounds(roi)

    root = 'users/dgketchum/uncultivated/'
    w_wetlands = ee.FeatureCollection(os.path.join(root, 'west_wetlands'))
    c_wetlands = ee.FeatureCollection(os.path.join(root, 'central_wetlands'))
    e_wetlands = ee.FeatureCollection(os.path.join(root, 'east_wetlands'))
    usgs_pad = ee.FeatureCollection(os.path.join(root, 'usgs_pad'))
    rf_uncult = ee.FeatureCollection(os.path.join(root, 'west_wetlands'))
    uncultivated_mask = w_wetlands.merge(c_wetlands).merge(e_wetlands).merge(usgs_pad).merge(rf_uncult)
    uncultivated_mask = uncultivated_mask.filterBounds(roi)

    masks_ = {'irrigated': irr_mask,
              'fallow': fallow_mask,
              'dryland': dryland_mask,
              'uncultivated': uncultivated_mask}
    return masks_


def class_codes():
    return {'irrigated': 1,
            'fallow': 2,
            'dryland': 3,
            'uncultivated': 4}


def create_class_labels(name_fc):
    class_labels = ee.Image(0).byte()
    # paint irrigated last
    for name in ['uncultivated', 'dryland', 'fallow', 'irrigated']:
        class_labels = class_labels.paint(name_fc[name], class_codes()[name])
    label = class_labels.updateMask(class_labels).rename('irr')

    return label


def get_ancillary():
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation') \
        .resample('bilinear').rename(['elev'])

    coords = terrain.pixelLonLat().rename(['lon', 'lat'])
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
        geometry=geo_.buffer(100000),
        cloud_cover_max=60)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
    interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
    return interp, target_rename


def extract_by_feature(year, points_to_extract=None,
                       n_shards=4, feature_id=1440, max_sample=100):

    roi = ee.FeatureCollection(TRAINING_GRID).filter(ee.Filter.eq('FID', feature_id)).geometry()
    # roi = ee.FeatureCollection(MT).geometry()

    s, e, interval_ = 1, 1, 30

    image_stack, features = get_sr_stack(year, s, e, interval_, roi)

    features = features + ['lat', 'lon', 'elev', 'irr']

    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = OrderedDict(zip(features, columns))
    # pprint(feature_dict)
    masks_ = masks(roi, year)
    if masks_['irrigated'].size().getInfo() == 0:
        print('no irrigated in {} in {}'.format(year, fid))
        return

    irr = create_class_labels(masks_)
    terrain_, coords_ = get_ancillary()
    data_stack = ee.Image.cat([image_stack, terrain_, coords_, irr]).float()
    data_stack = data_stack.neighborhoodToArray(KERNEL)

    if not points_to_extract:
        for name, fc in masks_.items():
            feature_count = 0
            points = fc.toList(fc.size())
            class_ = os.path.basename(name)
            out_filename = '{}_{}_{}'.format(class_, str(year), feature_id)
            geometry_sample = ee.ImageCollection([])

            for i in range(n_shards):
                sample = data_stack.sample(
                    region=ee.Feature(points.get(i)).geometry(),
                    scale=30,
                    numPixels=1,
                    tileScale=8)

                geometry_sample = geometry_sample.merge(sample)
                feature_count += 1
                if (feature_count + 1) % n_shards == 0:
                    task = ee.batch.Export.table.toCloudStorage(
                        collection=geometry_sample,
                        description=out_filename + str(time.time()),
                        bucket=GS_BUCKET,
                        fileNamePrefix=out_filename + str(time.time()),
                        fileFormat='TFRecord',
                        selectors=features
                    )
                    try:
                        task.start()
                    except ee.ee_exception.EEException:
                        print('waiting to export, sleeping for 50 minutes. Holding at\
                                {} {}, index {}'.format(year, name, i))
                        time.sleep(3000)
                        task.start()
                    geometry_sample = ee.ImageCollection([])
            # take care of leftovers
            print('{} {} extracted'.format(name, feature_count))
            task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                description=out_filename + str(time.time()),
                bucket=GS_BUCKET,
                fileNamePrefix=out_filename + str(time.time()),
                fileFormat='TFRecord',
                selectors=features)
            task.start()

    else:
        for fc in points_to_extract:
            class_ = os.path.basename(fc)
            points = ee.FeatureCollection(fc).filterBounds(roi)
            n_features = points.size().getInfo()

            if n_features == 0:
                print('no irrigated in {} in {}'.format(year, fid))
                break

            points = points.toList(points.size())

            if n_features > max_sample:
                # Error: List.get: List index must be between 0 and 99, or -100 and -1. Found 115.
                points = points.slice(0, max_sample)
                print(n_features, fc)

            geometry_sample = ee.ImageCollection([])
            out_filename = '{}_{}_{}'.format(class_, str(year), feature_id)
            ct = 0
            for i in range(n_features):
                sample = data_stack.sample(
                    region=ee.Feature(points.get(i)).geometry(),
                    scale=30,
                    numPixels=1,
                    tileScale=8)

                geometry_sample = geometry_sample.merge(sample)
                ct += 1
                if (i + 1) % n_shards == 0:
                    task = ee.batch.Export.table.toCloudStorage(
                        collection=geometry_sample,
                        bucket=GS_BUCKET,
                        description=out_filename + str(time.time()),
                        fileNamePrefix=out_filename + str(time.time()),
                        fileFormat='TFRecord',
                        selectors=features
                    )

                    try:
                        task.start()
                    except ee.ee_exception.EEException:
                        print('waiting to export, sleeping for 50 minutes. Holding at\
                                {} {}, index {}'.format(year, class_, i))
                        time.sleep(3000)
                        task.start()

                    geometry_sample = ee.ImageCollection([])
                    ct = 0
            task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                bucket=GS_BUCKET,
                description=out_filename + str(time.time()),
                fileNamePrefix=out_filename + str(time.time()),
                fileFormat='TFRecord',
                selectors=features)
            task.start()
            print('exported', fid, year)


def subsample_fid():
    return [136, 671, 1942, 294, 151, 58, 278, 1621, 2237, 2244, 1224, 2024, 2308, 829,
            1467, 1440, 842, 403, 11, 1554, 2275, 2206, 1984, 2174, 562, 854, 1211, 1260,
            2192, 1688, 36, 87, 2171, 1400, 1324, 529, 133, 2175, 93, 2227, 1227, 527, 2243,
            1017, 2195, 365, 1484, 1555, 566, 2138, 908, 182, 1655, 2006, 1629, 2236, 113,
            1658, 1726, 1818, 808, 649, 2140, 1807, 2278, 1232, 509, 516, 280, 1014]


if __name__ == '__main__':
    pts_root = 'users/dgketchum/training_points'
    pts_training = [os.path.join(pts_root, x) for x in ['irrigated', 'uncultivated', 'dryland', 'fallow']]
    for yr_ in YEARS:
        for fid in subsample_fid():
            extract_by_feature(yr_, points_to_extract=pts_training, feature_id=fid, max_sample=100)
