# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import os
import time

import ee

from datetime import datetime, timedelta


def add_doy(image):
    """ Add day-of-year image """
    mask = ee.Date(image.get('system:time_start'))
    day = ee.Image.constant(image.date().getRelative('day', 'year')).clip(image.geometry())
    i = image.addBands(day.rename('DOY')).int().updateMask(mask)
    return i


def get_world_climate(proj):
    n = list(range(1, 13))
    months = [str(x).zfill(2) for x in n]
    parameters = ['tavg', 'tmin', 'tmax', 'prec']
    combinations = [(m, p) for m in months for p in parameters]

    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).select(p).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                scale=30) for m, p in
         combinations]
    # not sure how to do this without initializing the image with a constant
    i = ee.Image(l)
    return i


def daily_landsat(year, roi):

    start = '{}-01-01'.format(year)
    end_date = '{}-01-01'.format(year + 1)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
        ee.FeatureCollection(roi).geometry()).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
        ee.FeatureCollection(roi).geometry()).filterDate(start, end_date).map(ls57mask)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
        ee.FeatureCollection(roi).geometry()).filterDate(start, end_date).map(ls8mask)

    ls_sr_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))

    d1 = datetime(year, 1, 1)
    d2 = datetime(year + 1, 1, 1)
    d_times = [(d1 + timedelta(days=x), d1 + timedelta(days=x + 1)) for x in range((d2-d1).days)]
    date_tups = [(x.strftime('%Y-%m-%d'), y.strftime('%Y-%m-%d')) for x, y in d_times]
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    l, empty = [], []
    final = False
    for s, e in date_tups:
        if s == '{}-12-31'.format(year):
            e = '{}-01-01'.format(year + 1)
            final = True
        dt = datetime.strptime(s, '%Y-%m-%d')
        doy = dt.strftime('%j')
        rename_bands = ['{}{}{}'.format(year, doy, b) for b in bands]
        b = ls_sr_masked.filterDate(s, e).mosaic().rename(rename_bands)

        try:
            _ = b.getInfo()['bands'][0]
        except IndexError:
            empty.append(s)
            continue

        b = b.unmask(-99)
        l.append(b)
        if final:
            break

    print('{} empty dates : {}'.format(len(empty), empty))
    i = ee.Image(l)
    return i


def get_qa_bits(image, start, end, qa_mask):
    pattern = 0
    for i in range(start, end - 1):
        pattern += 2 ** i
    return image.select([0], [qa_mask]).bitwiseAnd(pattern).rightShift(start)


def mask_quality(image):
    QA = image.select('pixel_qa')
    shadow = get_qa_bits(QA, 3, 3, 'cloud_shadow')
    cloud = get_qa_bits(QA, 5, 5, 'cloud')
    cirrus_detected = get_qa_bits(QA, 9, 9, 'cirrus_detected')
    return image.updateMask(shadow.eq(0)).updateMask(cloud.eq(0).updateMask(cirrus_detected.eq(0)))


def ls57mask(img):
    sr_bands = img.select('B1', 'B2', 'B3', 'B4', 'B5', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_sel = img_masked.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'], ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    mask_mult = mask_sel.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


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


def ndvi5():
    l = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').map(ls5_edge_removal).map(lambda x: x.select().addBands(
        x.normalizedDifference(['B4', 'B3'])))
    return l


def ndvi7():
    l = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').map(lambda x: x.select().addBands(
        x.normalizedDifference(['B4', 'B3'])))
    return l


def ndvi8():
    l = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').map(lambda x: x.select().addBands(
        x.normalizedDifference(['B5', 'B4'])))
    return l


def ls5_edge_removal(lsImage):
    inner_buffer = lsImage.geometry().buffer(-3000)
    buffer = lsImage.clip(inner_buffer)
    return buffer


def period_stat(collection, start, end):
    c = collection.filterDate(start, end)
    return c.reduce(
        ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(),
                                  sharedInputs=True))


def temporal_collection(collection, start, count, interval, units):

    sequence = ee.List.sequence(0, ee.Number(count).subtract(1))
    originalStartDate = ee.Date(start)
    def filt(i):

        startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units)

        endDate = originalStartDate.advance(
            ee.Number(interval).multiply(ee.Number(i).add(1)), units)
        return collection.filterDate(startDate, endDate).reduce(ee.Reducer.mean())
    return ee.ImageCollection(sequence.map(filt))


def extract_data_over_polygons(polygon_list, data_stack, out_folder, file_basename, features,
                               n_shards=10, every_n=1):
    geomSample = ee.ImageCollection([])
    len_geom_sample = 0
    for i, g in enumerate(range(polygon_list.size().getInfo())):
        if i % every_n != 0:
            continue
        sample = data_stack.sample(
            region = ee.Feature(polygon_list.get(g)).geometry(),
            scale = 30,
            numPixels = 1, # Size of the shard.
            seed = i,
            tileScale = 8
        )
        geomSample = geomSample.merge(sample)
        len_geom_sample += 1
        if len_geom_sample == n_shards:
            desc = file_basename + '_g' + str(g)
            print('saving to {}'.format(out_folder))
            task = ee.batch.Export.table.toDrive(
                collection=geomSample,
                description=desc,
                fileFormat='TFRecord',
                folder=out_folder,
                fileNamePrefix = file_basename + str(time.time()),
                selectors=features
            )
            task.start()
            geomSample = ee.ImageCollection([])
            len_geom_sample = 0


def assign_class_code(shapefile_path):
    shapefile_path = os.path.basename(shapefile_path)
    if 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
        return 0
    if 'unirrigated' in shapefile_path:
        return 1
    if 'fallow' in shapefile_path:
        return 2
    if 'wetlands' in shapefile_path:
        return 3
    if 'uncultivated' in shapefile_path:
        return 4
    if 'points' in shapefile_path:
        # annoying workaround for earthengine
        return 10
    else:
        raise NameError('shapefile path {} isn\'t named in assign_class_code'.format(shapefile_path))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

