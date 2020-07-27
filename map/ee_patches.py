import os
import ee
import tensorflow as tf
import time
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

# Tommy's training data
# gs://ee-irrigation-mapping/train-data-july9_1-578/
# gs://ee-irrigation-mapping/test-data-july23/
# gs://ee-irrigation-mapping/validation-data-july23/


def create_class_labels(shp_to_fc_, yr):

    class_labels = ee.Image(0).byte()

    cdl = ee.Image(ee.ImageCollection('USDA/NASS/CDL')
                   .filter(ee.Filter.date('{}-01-01'.format(yr), '{}-12-31'.format(yr)))
                   .first()
                   .select('cropland'))

    for shapefile, feature_collection in shp_to_fc_.items():
        class_labels = class_labels.paint(feature_collection, assign_class_code(shapefile) + 1)

    return class_labels.updateMask(class_labels), cdl


def temporally_filter_features(polygon_ds, yr_):

    polygon_to_fc = {}
    polygon_mapping = shapefile_counts()
    polygon_mapping = {'{}_MT'.format(k): v for k, v in polygon_mapping.items()}

    for shapefile in polygon_ds:
        is_temporal = True
        basename = os.path.basename(shapefile)
        feature_collection = ee.FeatureCollection(shapefile)

        if 'irrigated' in basename and 'unirrigated' not in basename:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", yr_))

        elif 'fallow' in basename:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", yr_))
        else:
            is_temporal = False
            polygon_to_fc[shapefile] = feature_collection

        if is_temporal:
            valid_years = list(dict(polygon_mapping[basename].items()).keys())
            if yr_ in valid_years:
                polygon_to_fc[shapefile] = feature_collection

    return polygon_to_fc


def get_sr_stack(yr, region):

    roi = ee.FeatureCollection(region)
    roi = roi.geometry()

    s = datetime(yr, 1, 1)
    e = datetime(yr + 1, 1, 1)
    target_interval = 15
    interp_days = 32

    target_dates = get_target_dates(s, e, interval_=target_interval)

    model_obj = Collection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        geometry=roi,
        cloud_cover_max=70)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']
    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)

    interp = interpolated.toBands().rename(target_rename)
    return interp, target_rename


def extract_data_over_shapefiles(label_polygons, year, extent, out_folder,
                                 points_to_extract=None, n_shards=10):

    polygon_mapping = shapefile_counts()
    polygon_mapping = {'{}_MT'.format(k): v for k, v in polygon_mapping.items()}

    image_stack, bands = get_sr_stack(year, region=extent)
    features = bands + ['irr', 'cdl']
    columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in features]
    feature_dict = dict(zip(features, columns))

    shp_to_fc = temporally_filter_features(label_polygons, year)
    class_labels, cdl_labels = create_class_labels(shp_to_fc, year)

    data_stack = ee.Image.cat([image_stack, class_labels, cdl_labels]).float()
    kernel = ee.Kernel.square(KERNEL_SIZE / 2)
    data_stack = data_stack.neighborhoodToArray(kernel)

    if points_to_extract is None:
        for shapefile, feature_collection in shp_to_fc.items():

            polygons = feature_collection.toList(feature_collection.size())
            n_features = polygon_mapping[os.path.basename(shapefile)][year]

            out_class_label = os.path.basename(shapefile)
            out_filename = out_folder + "_" + out_class_label + "_" + str(year)

            geometry_sample = ee.ImageCollection([])
            if 'irrigated' in out_class_label and 'unirrigated' not in out_class_label:
                rate = 1
            elif 'fallow' in out_class_label:
                rate = 1
            else:
                rate = 10

            print(year, shapefile, rate)
            for i in range(n_features):

                if i % rate != 0:
                    continue

                sample = data_stack.sample(
                    region=ee.Feature(polygons.get(i)).geometry(),
                    scale=30,
                    numPixels=1,
                    tileScale=16)

                geometry_sample = geometry_sample.merge(sample)
                if (i+1) % n_shards == 0:
                    task = ee.batch.Export.table.toCloudStorage(
                        collection=geometry_sample,
                        description=out_filename + str(time.time()),
                        bucket=GS_BUCKET,
                        fileNamePrefix=out_filename + str(time.time()),
                        fileFormat='TFRecord',
                        selectors=features)

                    # try:
                    task.start()
                    print('task start')

                    # except ee.ee_exception.EEException:
                    #     print('waiting to export, sleeping for 5 minutes. Holding at \n '
                    #           '{} {} {}, index {}'.format(year, shapefile, rate, i))
                    #     time.sleep(300)
                    #     task.start()
                    #     print('task start')

                    geometry_sample = ee.ImageCollection([])
                    break


if __name__ == '__main__':

    ee.Initialize(use_cloud_api=True)

    boundary = 'users/dgketchum/boundaries/MT'

    root = 'users/dgketchum/training_polygons/'
    test = ['fallow_test', 'irrigated_test', 'uncultivated_test',
            'unirrigated_test', 'wetlands_test']
    test = [root + t for t in test]
    train = ['fallow_train', 'irrigated_train', 'uncultivated_train',
             'unirrigated_train', 'wetlands_train']
    train = [root + t + '_MT' for t in train]

    extract_data_over_shapefiles(train, year=2010, extent=boundary, out_folder=GS_BUCKET)
# ========================= EOF ====================================================================
