import ee
import geopandas as gpd
import os
from glob import glob
from collections import defaultdict
from pprint import pprint

from map.trainer.shapefile_meta import SHP_TO_YEAR_AND_COUNT

YEARS = [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015]

LC8_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
LC7_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
LC5_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
STD_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']


def temporally_filter_features(shapefiles, year):
    shapefile_to_feature_collection = {}
    for shapefile in shapefiles:
        is_temporal = True
        bs = os.path.basename(shapefile)
        feature_collection = ee.FeatureCollection(shapefile)
        if 'irrigated' in bs and 'unirrigated' not in bs:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", year))
        elif 'fallow' in bs:
            feature_collection = feature_collection.filter(ee.Filter.eq("YEAR", year))
        else:
            # don't need to temporally filter non-temporal land cover classes
            is_temporal = False
            shapefile_to_feature_collection[shapefile] = feature_collection

        if is_temporal:
            valid_years = list(dict(SHP_TO_YEAR_AND_COUNT[bs].items()).keys())
            if year in valid_years:
                shapefile_to_feature_collection[shapefile] = feature_collection

    return shapefile_to_feature_collection


def create_class_labels(shapefile_to_feature_collection):

    class_labels = ee.Image(0).byte()
    for shapefile, feature_collection in shapefile_to_feature_collection.items():
        class_labels = class_labels.paint(feature_collection,
                                          assign_class_code(shapefile) + 1)

    label = class_labels.updateMask(class_labels).rename('irr')

    return label


def ls8mask(img):
    sr_bands = img.select('B2', 'B3', 'B4', 'B5', 'B6', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_mult = img_masked.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def preprocess_data_l8_cloudmask(year):
    l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
    l8 = l8.map(ls8mask).select(LC8_BANDS, STD_NAMES)
    return temporalCollection(l8, ee.Date('{}-05-01'.format(year)), 6, 32, 'days')


def preprocess_data(year):
    l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').select(LC8_BANDS, STD_NAMES)
    l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').select(LC7_BANDS, STD_NAMES)
    l5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').select(LC7_BANDS, STD_NAMES)
    l7l8 = ee.ImageCollection(l7.merge(l8).merge(l5))

    return temporalCollection(l7l8, ee.Date('{}-05-01'.format(year)), 6, 32, 'days')


def temporalCollection(collection, start, count, interval, units):
    sequence = ee.List.sequence(0, ee.Number(count).subtract(1))
    originalStartDate = ee.Date(start)

    def filt(i):
        startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units)

        endDate = originalStartDate.advance(
            ee.Number(interval).multiply(ee.Number(i).add(1)), units)
        return collection.filterDate(startDate, endDate).reduce(ee.Reducer.mean())

    return ee.ImageCollection(sequence.map(filt))


def assign_class_code(shapefile_path):
    shapefile_path = os.path.basename(shapefile_path)
    if 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
        return 1
    if 'fallow' in shapefile_path:
        return 2
    if 'unirrigated' in shapefile_path:
        return 3
    if 'wetlands' in shapefile_path:
        return 4
    if 'uncultivated' in shapefile_path:
        return 5
    if 'points' in shapefile_path:
        # annoying workaround for earthengine
        return 10
    else:
        raise NameError('shapefile path {} isn\'t named in assign_class_code'.format(shapefile_path))


def is_temporal(shapefile_path):
    shapefile_path = os.path.basename(shapefile_path)
    if 'fallow' in shapefile_path:
        return True
    elif 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
        return True
    else:
        return False


def make_shape_to_year_and_count_dict(shapefile_dir):
    '''
    Uses local copies of shapefiles that have been
    uploaded to GEE.
    '''
    shape_to_year_and_count = defaultdict(dict)

    for f in glob(os.path.join(shapefile_dir, "*shp")):
        gdf = gpd.read_file(f)
        stripped_filename = os.path.splitext(os.path.basename(f))[0]
        for year in YEARS:
            if is_temporal(f):
                shape_to_year_and_count[stripped_filename][year] = sum(gdf['YEAR'] == year)
            else:
                shape_to_year_and_count[stripped_filename][year] = gdf.shape[0]

    return shape_to_year_and_count


if __name__ == '__main__':
    shapefile_train = '/home/thomas/irrigated-training-data/ee-dataset/train/'
    shapefile_test = '/home/thomas/irrigated-training-data/ee-dataset/test/'
    shapefile_valid = '/home/thomas/irrigated-training-data/ee-dataset/validation/'

    train = make_shape_to_year_and_count_dict(shapefile_train)
    test = make_shape_to_year_and_count_dict(shapefile_test)
    valid = make_shape_to_year_and_count_dict(shapefile_valid)
    merged = {**train, **test, **valid}  # wow python

    pprint(merged)
