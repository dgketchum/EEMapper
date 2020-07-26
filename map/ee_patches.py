import ee
import tensorflow as tf
import time
from datetime import datetime
from map.openet.collection import get_target_bands, get_target_dates, Collection

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

GS_BUCKET = 'wudr'
KERNEL_SIZE = 256
year = 2017
s = datetime(year, 3, 1)
e = datetime(year, 11, 1)
target_interval = 15
interp_days = 32

# Tommy's training data
# gs://ee-irrigation-mapping/train-data-july9_1-578/
# gs://ee-irrigation-mapping/test-data-july23/
# gs://ee-irrigation-mapping/validation-data-july23/


def create_class_labels():
    class_labels = ee.Image(ee.ImageCollection('USDA/NASS/CDL').select('cropland'))
    return class_labels


def test_geometries():
    test_xy = [ee.Feature(ee.Geometry.Point(-112.20878671819048, 47.5176895106640), {'POINT_TYPE': 0}),
               ee.Feature(ee.Geometry.Point(-111.61724610359778, 47.6963644206754), {'POINT_TYPE': 1}),
               ee.Feature(ee.Geometry.Point(-112.32898130320568, 47.4473514239175), {'POINT_TYPE': 2}),
               ee.Feature(ee.Geometry.Point(-111.74121361280626, 47.4209409545644), {'POINT_TYPE': 3})]

    fc = ee.FeatureCollection(test_xy)
    return fc


def get_sr_stack(target_ids):
    study_region = ee.Geometry.Rectangle([-112.5, 47.3, -111.5, 48.0])

    model_obj = Collection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        geometry=study_region,
        cloud_cover_max=70)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']
    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_ids)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)

    interp = interpolated.toBands().rename(target_rename)
    return interp


def extract_data_over_shapefiles(year, out_folder, n_shards=10):

    KERNEL = ee.Kernel.square(128/2)

    target_ids = get_target_dates(s, e, interval_=target_interval)
    image_stack = get_sr_stack(target_ids)

    class_labels = create_class_labels()
    data_stack = image_stack.addBands([class_labels])
    data_stack = data_stack.neighborhoodToArray(KERNEL)

    polygons = ee.FeatureCollection(test_geometries())
    polygons = polygons.toList(polygons.size())
    n_features = 4
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

    task = ee.batch.Export.table.toCloudStorage(
        collection=geometry_sample,
        bucket=GS_BUCKET,
        description=out_filename + str(time.time()),
        fileNamePrefix=out_folder + out_filename + str(time.time()),
        fileFormat='TFRecord',
        # selectors=features
    )
    task.start()
    print(n_extracted, year)


if __name__ == '__main__':
    ee.Initialize(use_cloud_api=True)
    extract_data_over_shapefiles(2017, out_folder=GS_BUCKET)
# ========================= EOF ====================================================================
