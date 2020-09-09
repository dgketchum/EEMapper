import ee

ee.Initialize()
import time
import os
import sys

sys.path.append('/home/dgketchum/PycharmProjects/EEMapper')
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
        geometry=geo_,
        cloud_cover_max=100)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
    interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
    return interp, target_rename


class GEEExtractor:

    def __init__(self, year, feature_id):
        self.year = year
        self.out_gs_bucket = GS_BUCKET
        self.masks = None
        self.image_stack = None
        self.kernel_size = 256
        self.start, self.end, self.interval = 1, 1, 30
        self.fid = feature_id
        roi = ee.FeatureCollection(TRAINING_GRID).filter(ee.Filter.eq('FID', feature_id))
        l = roi.toList(roi.size().getInfo())
        self.patch = ee.Feature(l.get(0))
        self.geo = roi.geometry()

    def extract_patch(self):

        image_stack, features = self._get_sr_stack()
        masks_ = masks(self.geo, self.year)
        if masks_['irrigated'].size().getInfo() == 0:
            print('no irrigated in {} in {}'.format(self.year, fid))
            return

        irr = create_class_labels(masks_)
        terrain_, coords_ = get_ancillary()
        self.image_stack = ee.Image.cat([image_stack, terrain_, coords_, irr]).float()
        # data_stack = data_stack.neighborhoodToArray(KERNEL)

        projection = ee.Projection('EPSG:5070')
        self.image_stack = image_stack.reproject(projection, None, 30)
        out_filename = '{}_{}_1kmbuf_100mcp'.format(self.fid, self.year)
        self._create_and_start_image_task(out_filename)
        return True

    def _get_sr_stack(self):
        s = datetime(self.year, self.start, 1)
        e = datetime(self.year + 1, self.end, 1)
        target_interval = self.interval
        interp_days = 32

        target_dates = get_target_dates(s, e, interval_=target_interval)

        model_obj = Collection(
            collections=COLLECTIONS,
            start_date=s,
            end_date=e,
            geometry=self.geo.buffer(1000),
            cloud_cover_max=100)

        variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

        interpolated = model_obj.interpolate(variables=variables_,
                                             interp_days=interp_days,
                                             dates=target_dates)

        target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
        interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
        return interp, target_rename

    def _create_and_start_image_task(self, out_filename):
        task = ee.batch.Export.image.toCloudStorage(
            image=self.image_stack,
            bucket=self.out_gs_bucket,
            description=out_filename,
            fileNamePrefix=out_filename,
            fileFormat='TFRecord',
            region=self.patch.geometry(),
            crs='EPSG:5070',
            scale=30,
            formatOptions={'patchDimensions': 256,
                           'compressed': True,
                           'maskedThreshold': 0.99}, )
        self._start_task_and_handle_exception(task)

    def _start_task_and_handle_exception(self, task):
        try:
            task.start()
        except ee.ee_exception.EEException as e:
            print(e)
            print('waiting to export, sleeping for 50 minutes')
            time.sleep(3000)
            task.start()
        print(self.fid, self.year, 'export')
        return ee.ImageCollection([])

    def _create_filename(self, shapefile):
        return os.path.basename(shapefile) + str(self.year)


def subsample_fid():
    return [1621, 136, 671, 1942, 294, 151, 58, 278, 2237, 2244, 1224, 2024, 2308, 829,
            1467, 1440, 842, 403, 11, 1554, 2275, 2206, 1984, 2174, 562, 854, 1211, 1260,
            2192, 1688, 36, 87, 2171, 1400, 1324, 529, 133, 2175, 93, 2227, 1227, 527, 2243,
            1017, 2195, 365, 1484, 1555, 566, 2138, 908, 182, 1655, 2006, 1629, 2236, 113,
            1658, 1726, 1818, 808, 649, 2140, 1807, 2278, 1232, 509, 516, 280, 1014]


if __name__ == '__main__':
    for yr_ in YEARS[-5:-4]:
        for i, fid in enumerate(subsample_fid()):
            g = GEEExtractor(yr_, fid)
            g.extract_patch()
            if i < 7:
                exit()
