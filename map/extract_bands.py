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

import ee

from map.distribute_points import get_years

YEARS = get_years()
TABLE = 'users/dgketchum/irrigation_projects/MT_Statewide_Irr'
ROI = 'users/dgketchum/boundaries/western_states_polygon'
PLOTS = 'users/dgketchum/classify/sample_4c_45k'


def request_band_extract():
    for yr in YEARS:
        start = '{}-01-01'.format(yr)
        end_date = '{}-01-01'.format(yr + 1)
        spring_s = '{}-03-01'.format(yr)
        summer_s = '{}-06-01'.format(yr)
        fall_s = '{}-09-01'.format(yr)

        l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
            ROI).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)

        l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
            ROI).filterDate(start, end_date).map(ls57mask)

        l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
            ROI).filterDate(start, end_date).map(ls8mask)

        lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))
        lsSR_spr_mn = ee.Image(lsSR_masked.filterDate(spring_s, summer_s).mean())
        lsSR_sum_mn = ee.Image(lsSR_masked.filterDate(summer_s, fall_s).mean())
        lsSR_fal_mn = ee.Image(lsSR_masked.filterDate(fall_s, end_date).mean())

        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
            ROI).filterDate(start, end_date).select('pr', 'eto', 'tmmn', 'tmmx').map(lambda x: x)

        temp_reducer = ee.Reducer.percentile([10, 50, 90])
        t_names = ['tmmn_p10_cy', 'tmmn_p50_cy', 'tmmn_p90_cy', 'tmmx_p10_cy', 'tmmx_p50_cy', 'tmmx_p90_cy']
        temp_perc = gridmet.select('tmmn', 'tmmx').reduce(temp_reducer).rename(t_names);
        precip_reducer = ee.Reducer.sum()
        precip_sum = gridmet.select('pr', 'eto').reduce(precip_reducer).rename('precip_total_cy', 'pet_total_cy')
        wd_estimate = precip_sum.select('precip_total_cy').subtract(precip_sum.select('pet_total_cy')).rename(
            'wd_est_cy')
        input_bands = lsSR_spr_mn.addBands([lsSR_sum_mn, lsSR_fal_mn, temp_perc, precip_sum, wd_estimate])

        dem1 = ee.Image('USGS/NED')
        dem2 = ee.Terrain.products(dem1).select('elevation', 'slope', 'aspect')
        static_input_bands = dem2
        coords = ee.Image.pixelLonLat().rename(['Lon_GCS', 'LAT_GCS'])
        static_input_bands = static_input_bands.addBands(coords)
        input_bands = input_bands.addBands(static_input_bands)

        plot_sample_regions = input_bands.sampleRegions(
            collection=PLOTS.filter(ee.Filter.eq('YEAR', ee.Date(start).get('YEAR'))),
            properties=['POINT_TYPE', 'YEAR'],
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(PLOTS.getInfo()['id'].split('/')[-1], yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(PLOTS.getInfo()['id'].split('/')[-1], yr),
            fileFormat='CSV')

        task.start()
        break


def get_qa_bits(image, start, end, mascara):
    pattern = 0
    for i in range(start, end - 1):
        pattern += 2 ** i
    return image.select([0], [mascara]).bitwiseAnd(pattern).rightShift(start)


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
    mask_p = (mask1 & mask2)
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
    mask_p = (mask1 & mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_mult = img_masked.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def ls5_edge_removal(lsImage):
    inner_buffer = lsImage.geometry().buffer(-3000)
    buffer = lsImage.clip(inner_buffer)
    return buffer


def is_authorized():
    try:
        ee.Initialize()
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


if __name__ == '__main__':
    is_authorized()
    request_band_extract()
# ========================= EOF ====================================================================
