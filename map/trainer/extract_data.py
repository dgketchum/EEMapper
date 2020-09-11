import ee

ee.Initialize()
import time
import os
from pprint import pprint
import sys

sys.path.append('/home/dgketchum/PycharmProjects/EEMapper')

from collections import OrderedDict
from datetime import datetime
import numpy as np
from map.openet.collection import get_target_dates, Collection, get_target_bands

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'ts_data'

BOUNDARIES = 'users/dgketchum/boundaries'
MGRS = os.path.join(BOUNDARIES, 'MGRS_TILE')
EE_DATA = 'users/dgketchum/grids'
MT = os.path.join(BOUNDARIES, 'MT')

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

CLASSES = ['uncultivated', 'dryland', 'fallow', 'irrigated']

YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997, 1998,
         2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

STATE_YEARS = {'NM': [1987, 2001, 2004, 2007, 2016, 1988, 1994, 2002, 2010, 1989, 2014, 2008, 2009, 2011, 2012, 2013],
               'AZ': [2001, 2003, 2004, 2016, 2010, 2012, 2007],
               'MT': [2003, 2015, 2008, 2009, 2010, 2011, 2012, 2013],
               'CA': [1995, 1998, 2000, 2007, 2014, 2016, 2002, 2005, 2006, 2008, 2009, 2001, 2003],
               'CO': [1998, 2003, 2006, 2013, 2016, 2008, 2010, 2011],
               'WY': [1998, 2003, 2013, 2016, 2006],
               'ND': [2012, 2013, 2014, 2015, 2016, 2010, 2011, 2003, 2008, 2009],
               'SD': [2012, 2013, 2014, 2015, 2016, 2007, 2008, 2009],
               'NE': [2012, 2013, 2014, 2015, 2016, 1993, 2003, 2009],
               'KS': [2012, 2013, 2014, 2015, 2016, 2002, 2006, 2009],
               'TX': [2012, 2013, 2014, 2015, 2016, 2005, 2006, 2009],
               'OK': [2012, 2013, 2014, 2015, 2016, 2006, 2007, 2011],
               'ID': [1986, 1996, 2002, 2006, 2008, 2009, 2010, 2011, 1997, 1998, 2001, 2013, 1988, 2017, 2003],
               'NV': [2002, 2005, 2006, 2008, 2009, 2001, 2003, 2007],
               'UT': [2001, 2005, 2009, 2008, 2010, 2011, 2012, 2013, 1998, 2003, 2006, 2016],
               'OR': [1996, 1997, 2001, 2013, 1994, 2011],
               'WA': [1988, 1996, 1997, 1998, 2001, 2006]}

DEBUG_SELECT = {'train': [675, 715, 676, 716, 677, 696, 717],
                'test': [192, 202, 220],
                'val': [212, 225, 239]}


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
    rf_uncult = ee.FeatureCollection(os.path.join(root, 'uncultivated'))
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


def extract_by_feature(feature_id=1440, split=None):

    grid = os.path.join(EE_DATA, '{}_grid'.format(split))
    roi = ee.FeatureCollection(grid).filter(ee.Filter.eq('FID', feature_id))

    points = os.path.join(EE_DATA, '{}_pts'.format(split))
    points = ee.FeatureCollection(points).filter(ee.Filter.eq('POLYFID', feature_id))
    points = points.toList(points.size())

    geo = roi.geometry()
    size = roi.size().getInfo()
    if size != 1:
        raise TypeError
    info_ = roi.first().getInfo()
    state, irr = info_['properties']['STUSPS'], info_['properties']['IRR']
    years = STATE_YEARS[state]

    for year in years:

        s, e, interval_ = 1, 1, 30
        image_stack, features = get_sr_stack(year, s, e, interval_, geo)

        masks_ = masks(geo, year)
        irr = create_class_labels(masks_)
        terrain_, coords_ = get_ancillary()
        image_stack = ee.Image.cat([image_stack, terrain_, coords_, irr]).float()

        if irr and masks_['irrigated'].size().getInfo() == 0:
            print('no irrigated in {} in {}'.format(year, fid))
            continue

        projection = ee.Projection('EPSG:5070')
        image_stack = image_stack.reproject(projection, None, 30)

        if not split:
            raise NotImplementedError

        out_filename = '{}_{}_{}_{}'.format(split, state, feature_id, year)
        data_stack = image_stack.neighborhoodToArray(KERNEL)
        geometry_sample = ee.ImageCollection([])

        for i in range(9):
            sample = data_stack.sample(
                region=ee.Feature(points.get(i)).geometry(),
                scale=30,
                numPixels=1,
                tileScale=8)
            geometry_sample = geometry_sample.merge(sample)

        task = ee.batch.Export.table.toCloudStorage(
            collection=geometry_sample,
            bucket=GS_BUCKET,
            description=out_filename,
            fileNamePrefix=out_filename,
            fileFormat='TFRecord',
            selectors=features)

        try:
            task.start()
        except ee.ee_exception.EEException:
            print('waiting to export, sleeping for 50 minutes. Holding at\
                    {}, feature {}'.format(year, feature_id))
            time.sleep(3000)
            task.start()
        print('exported', state, feature_id, year)


if __name__ == '__main__':
    for split, fids in DEBUG_SELECT.items():
        for fid in fids:
            extract_by_feature(fid, split)
# =====================================================================================================================
