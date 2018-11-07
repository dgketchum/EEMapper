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

## UCRB is going with the pure state input data for now.
# 'UCRB_WY': ('ft:1M0GDErc0dgoYajU_HStZBkp-hBL4kUiZufFdtWHG', [1989, 1996, 2010, 2013, 2016], 0.5),  # a.k.a. 2000
# 'UCRB_UT_CO': ('ft:1Av2WlcPRBd7JZqYOU73VCLOJ-b5q6H5u6Bboebdv', [1998, 2003, 2006, 2013, 2016], 0.5),  # a.k.a. 2005
# 'UCRB_UT': ('ft:144ymxhlcv8lj1u_BYQFEC1ITmiISW52q5JvxSVyk', [1998, 2003, 2006, 2013, 2016], 0.5),  # a.k.a. 2006
# 'UCRB_NM': ('ft:1pBSJDPdFDHARbdc5vpT5FzRek-3KXLKjNBeVyGdR', [1987, 2001, 2004, 2007, 2016], 0.4),  # a.k.a. 2009
# ===============================================================================

from datetime import datetime
from pprint import pprint

import ee

ROI = 'users/dgketchum/boundaries/western_states_expanded_union'
ROI_MT = 'users/dgketchum/boundaries/MT'

PLOTS = 'ft:1nNO0AfiHcZk5a_aPSr2Oo2gHqu7tvEcYn-w0RgvL'  # 100k
# PLOTS = 'ft:1oSEFGaE6wc08c_xMunUxfTqLpwR-cCL8vopDtqLi'  # 2k

TABLE = 'ft:1AEkGeGUjaVoE4ct-zR30o7BtujvdLjIGanRAhuO7'  # 100k extract eF
# TABLE = 'ft:1AEkGeGUjaVoE4ct-zR30o7BtujvdLjIGanRAhuO7'  # 2k extract eF

# YEARS = [1986, 1987, 1991, 1996, 1997, 1998, 1999] + list(range(2000, 2018))
YEARS = list(range(2008, 2014))

IRR = {
    # 'Acequias': ('ft:1emF9Imjj8GPxpRmPU2Oze2hPeojPS4O6udIQNTgX', [1987, 2001, 2004, 2007, 2016], 0.5),
    # 'CO_DIV1': ('ft:1wRNUsKChMUb9rUWDbxOeGeTaNWNZUA0YHXSLXPv2', [1998, 2003, 2006, 2013, 2016], 0.5),
    # 'CO_SanLuis': ('ft:1mcBXyFw1PoVOoAGibDpZjCgb001jA_Mj_hyd-h92', [1998, 2003, 2006, 2013, 2016], 0.5),
    # 'CA': ('ft:1oadWhheDKaonOPhIJ9lJVCwnOt5g0G644p3FC9oy', [2014], 0.5),
    # 'EastStates': ('ft:1AZUak3iuAtah1SHpkLfw0IRk_U5ei23VsPzBWxpD', [1987, 2001, 2004, 2007, 2016], 0.5),
    # 'ID': ('ft:1jDB3C181w1PGVamr64-ewpJVDQkzJc4Bvd1IPAFg', [1988, 1998, 2001, 2006, 2009, 2017], 0.5),
    # 'NV': ('ft:1DUcSDaruwvXMIyBEYd2_rCYo8w6D6v4nHTs5nsTR', [x for x in range(2001, 2011)], 0.5),
    # 'OR': ('ft:1FJMi4VXUe4BrhU6u0OF2l0uFU_rGUe3rFrSSSBVD', [1994, 1997, 2011], 0.5),
    # 'UT': ('ft:1oA0v3UUBQj3qn9sa_pDJ8bwsAphfRZUlwwPWpFrT', [1998, 2003, 2006, 2013, 2016], 0.5),
    # 'WA': ('ft:1tGN7UdKijI7gZgna19wJ-cKMumSKRwsfEQQZNQjl', [1997, 1996], 0.5),
}

ID = {

    'ID_1986': ('ft:1rAO90xDcSyR1GTKjAN4Z12vf2mVy8iKRr7vorhLr', [1986], 0.5),
    'ID_1996': ('ft:1Injcz-Q3HgQ_gip9ZEMqUAmihqbjTOEYHThuoGmL', [1996], 0.5),
    'ID_2002': ('ft:14pePO5Wr7Hcbz_VHuUYd_f5ReblXtvUFS3BrCpI2', [2002], 0.5),
    'ID_2006': ('ft:1NM9NsQJfdNAEwmZXyo5o78ha4PhgsErWzfttfZM1', [2006], 0.5),
    'ID_2008': ('ft:1VK5sWEgD35fz4pNNbg9sy6nlBD_lIl_t-ELyWbC9', [2008], 0.5),
    'ID_2009': ('ft:1RtW_lu3hFcpzZ_UUT_xPUsHarAZoenV4AibeFeBz', [2009], 0.5),
    'ID_2010': ('ft:1BSxEsy_oDUnWsWsQYJFRPaEvKsF-H_bDNE_gpBS7', [2010], 0.5),
    'ID_2011': ('ft:1NxN6aOViiJBklaUEEeGJJo6Kpy-QB10f_yGWOUyC', [2011], 0.5),
}


def export_classification(file_prefix):
    fc = ee.FeatureCollection(TABLE)

    roi = ee.FeatureCollection(ROI_MT)

    classifier = ee.Classifier.randomForest(
        numberOfTrees=50,
        variablesPerSplit=0,
        minLeafPopulation=1,
        bagFraction=0.5,
        outOfBagMode=False,
        seed=0).setOutputMode('CLASSIFICATION')

    input_props = fc.first().propertyNames().remove('YEAR').remove('POINT_TYPE').remove('system:index')
    trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

    for yr in YEARS:
        input_bands = stack_bands(yr, roi)
        annual_stack = input_bands.select(input_props)
        classified_img = annual_stack.classify(trained_model)

        task = ee.batch.Export.image.toAsset(
            image=classified_img,
            description='{}_{}'.format(file_prefix, yr),
            assetId='users/dgketchum/classy/{}_{}'.format(file_prefix, yr),
            scale=30,
            maxPixels=1e12)

        task.start()
        print(yr)
        break


def filter_irrigated():
    for k, v in ID.items():
        plots = ee.FeatureCollection(v[0])

        for year in v[1]:
            print(k, year, plots.first().getInfo())
            start = '{}-01-01'.format(year)

            early_summer_s = ee.Date(start).advance(5, 'month')
            early_summer_e = ee.Date(start).advance(7, 'month')
            late_summer_s = ee.Date(start).advance(7, 'month')
            late_summer_e = ee.Date(start).advance(10, 'month')

            if year < 2013:
                collection = ndvi5()
            else:
                collection = ndvi8()

            early_collection = period_stat(collection, early_summer_s, early_summer_e)
            late_collection = period_stat(collection, late_summer_s, late_summer_e)

            early_nd_max = early_collection.select('nd_mean').reduce(ee.Reducer.intervalMean(0.0, 15.0))
            early_int_mean = early_nd_max.reduceRegions(collection=plots,
                                                        reducer=ee.Reducer.mean(),
                                                        scale=30.0)

            s_nd_max = late_collection.select('nd_mean').reduce(ee.Reducer.intervalMean(0.0, 15.0))
            combo_mean = s_nd_max.reduceRegions(collection=early_int_mean,
                                                reducer=ee.Reducer.mean(),
                                                scale=30.0)

            filt_fc = combo_mean.filter(ee.Filter.Or(ee.Filter.gt('mean', v[2]), ee.Filter.gt('mean', v[2])))

            task = ee.batch.Export.table.toCloudStorage(filt_fc,
                                                        folder='Irrigation',
                                                        description='{}_{}'.format(k, year),
                                                        bucket='wudr',
                                                        fileNamePrefix='{}_{}'.format(k, year),
                                                        fileFormat='KML')

            task.start()


def request_band_extract(file_prefix):
    roi = ee.FeatureCollection(ROI)
    plots = ee.FeatureCollection(PLOTS)

    for yr in YEARS:
        input_bands = stack_bands(yr, roi)
        start = '{}-01-01'.format(yr)
        d = datetime.strptime(start, '%Y-%m-%d')
        epoch = datetime.utcfromtimestamp(0)
        start_millisec = (d - epoch).total_seconds() * 1000

        filtered = plots.filter(ee.Filter.eq('YEAR', ee.Number(start_millisec)))

        plot_sample_regions = input_bands.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR'],
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print(yr)


def stack_bands(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)
    spring_s = '{}-03-01'.format(yr)
    summer_s = '{}-06-01'.format(yr)
    fall_s = '{}-09-01'.format(yr)

    l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls57mask)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls8mask)
    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))
    lsSR_spr_mn = ee.Image(lsSR_masked.filterDate(spring_s, summer_s).mean())
    lsSR_sum_mn = ee.Image(lsSR_masked.filterDate(summer_s, fall_s).mean())
    lsSR_fal_mn = ee.Image(lsSR_masked.filterDate(fall_s, end_date).mean())

    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
        roi).filterDate(start, end_date).select('pr', 'eto', 'tmmn', 'tmmx')
    temp_reducer = ee.Reducer.percentile([10, 50, 90])
    t_names = ['tmmn_p10_cy', 'tmmn_p50_cy', 'tmmn_p90_cy', 'tmmx_p10_cy', 'tmmx_p50_cy', 'tmmx_p90_cy']
    temp_perc = gridmet.select('tmmn', 'tmmx').reduce(temp_reducer).rename(t_names)
    precip_reducer = ee.Reducer.sum()
    precip_sum = gridmet.select('pr', 'eto').reduce(precip_reducer).rename('precip_total_cy', 'pet_total_cy')
    wd_estimate = precip_sum.select('precip_total_cy').subtract(precip_sum.select(
        'pet_total_cy')).rename('wd_est_cy')
    input_bands = lsSR_spr_mn.addBands([lsSR_sum_mn, lsSR_fal_mn, temp_perc, precip_sum, wd_estimate])

    coords = ee.Image.pixelLonLat().rename(['Lon_GCS', 'LAT_GCS'])
    static_input_bands = coords
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect')
    static_input_bands = static_input_bands.addBands(terrain)

    # Extended Fetures/ eF
    world_climate = get_world_climate()
    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    static_input_bands = static_input_bands.addBands([tpi_1250, tpi_250, tpi_150, world_climate])

    input_bands = input_bands.addBands(static_input_bands).clip(roi)
    return input_bands


def get_world_climate():
    n = list(range(1, 13))
    months = [str(x).zfill(2) for x in n]
    parameters = ['tavg', 'tmin', 'tmax', 'prec']
    combinations = [(m, p) for m in months for p in parameters]
    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).select(p) for m, p in combinations]
    i = ee.Image(l[0])
    i = i.addBands(l)
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
    # request_band_extract('eF_2k_MT')
    filter_irrigated()
# ========================= EOF ====================================================================
