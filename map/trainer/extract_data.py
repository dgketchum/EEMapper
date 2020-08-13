import ee

ee.Initialize()
import tensorflow as tf
import time
import os
from pprint import pprint

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
MT = os.path.join(BOUNDARIES, 'MT')

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']


def masks(roi):
    root = 'users/dgketchum/training_polygons/'
    irr_mask = ee.FeatureCollection(os.path.join(root, 'irrigated_mask')).filterBounds(roi)
    dryland_mask = ee.FeatureCollection(os.path.join(root, 'dryland_mask')).filterBounds(roi)

    root = 'users/dgketchum/uncultivated/'
    w_wetlands = ee.FeatureCollection(os.path.join(root, 'west_wetlands'))
    c_wetlands = ee.FeatureCollection(os.path.join(root, 'central_wetlands'))
    e_wetlands = ee.FeatureCollection(os.path.join(root, 'east_wetlands'))
    usgs_pad = ee.FeatureCollection(os.path.join(root, 'usgs_pad'))
    rf_uncult = ee.FeatureCollection(os.path.join(root, 'west_wetlands'))
    uncultivated_mask = w_wetlands.merge(c_wetlands).merge(e_wetlands).merge(usgs_pad).merge(rf_uncult)
    uncultivated_mask = uncultivated_mask.filterBounds(roi)

    return {'irrigated': irr_mask, 'dryland': dryland_mask, 'uncultivated': uncultivated_mask}


def temporally_filter_features(masks_, year_):

    shapefile_to_feature_collection = {}
    for name, fc in masks_.items():
        if 'irrigated' in name:
            feature_collection = fc.filter(ee.Filter.eq("YEAR", year_))
            shapefile_to_feature_collection[name] = feature_collection

        elif 'fallow' in name:
            fc = fc.filter(ee.Filter.eq("YEAR", year_))
            shapefile_to_feature_collection[name] = fc
        else:
            shapefile_to_feature_collection[name] = fc

    return shapefile_to_feature_collection


def create_class_labels(shapefile_to_feature_collection):
    class_labels = ee.Image(0).byte()
    for name, fc in shapefile_to_feature_collection.items():
        class_labels = class_labels.paint(fc, assign_class_code(name) + 1)

    label = class_labels.updateMask(class_labels).rename('irr')

    return label


def assign_class_code(shapefile_path):
    shapefile_path = os.path.basename(shapefile_path)
    if 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
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


def get_ancillary():
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


def extract_data_over_shapefiles(year, out_folder, points_to_extract=None, n_shards=15):

    roi = ee.FeatureCollection(MGRS).filter(ee.Filter.eq('MGRS_TILE', '12TVT')).geometry()
    # roi = ee.FeatureCollection(MT).geometry()

    s, e, interval_ = 1, 1, 30

    image_stack, features = get_sr_stack(year, s, e, interval_, roi)

    features = features + ['lat', 'lon', 'elev', 'irr']

    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = OrderedDict(zip(features, columns))
    # pprint(feature_dict)
    masks_ = masks(roi)
    name_fc = temporally_filter_features(masks_, year)
    if points_to_extract is not None:
        name_fc['points'] = points_to_extract

    irr = create_class_labels(name_fc)
    terrain_, coords_ = get_ancillary()
    data_stack = ee.Image.cat([image_stack, terrain_, coords_, irr]).float()
    data_stack = data_stack.neighborhoodToArray(KERNEL)

    if points_to_extract is None:
        for name, fc in name_fc.items():
            feature_count = 0
            polygons = fc.toList(fc.size())
            out_class_label = os.path.basename(name)
            out_filename = out_class_label + "_sr{}_s{}_e{}_".format(KERNEL_SIZE, s, e) + str(year)
            geometry_sample = ee.ImageCollection([])

            for i in range(n_shards):
                sample = data_stack.sample(
                    region=ee.Feature(polygons.get(i)).geometry(),
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
    extract_data_over_shapefiles(2010, out_folder=GS_BUCKET)
