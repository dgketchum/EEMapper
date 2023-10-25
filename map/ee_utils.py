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
from datetime import datetime

import ee


def landsat_c2_sr(input_img):
    # credit: cgmorton; https://github.com/Open-ET/openet-core-beta/blob/master/openet/core/common.py

    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                      'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                      'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                    'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))

    prep_image = input_img \
        .select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                   0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _cloud_mask(i):
        qa_img = i.select(['QA_PIXEL'])
        cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
        cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))
        sat_mask = i.select(['QA_RADSAT']).gt(0)
        cloud_mask = cloud_mask.Or(sat_mask)

        cloud_mask = cloud_mask.Not().rename(['cloud_mask'])

        return cloud_mask

    mask = _cloud_mask(input_img)

    image = prep_image.updateMask(mask).copyProperties(input_img, ['system:time_start'])

    return image


def landsat_masked(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l4_coll = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(
        roi).filterDate(start, end_date).map(landsat_c2_sr)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll).merge(l4_coll))
    return lsSR_masked


def landsat_composites(year, start, end, roi, append_name, composites_only=False):
    start_year = datetime.strptime(start, '%Y-%m-%d').year
    if start_year != year:
        year = start_year
    lsSR_masked = landsat_masked(year, roi)

    if append_name in ['m2', 'm1', 'cy']:
        ndvi_mx = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_max_{}'.format(append_name))
    else:
        raise NotImplementedError

    return ndvi_mx


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
    l = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').map(lambda x: x.select().addBands(
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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

