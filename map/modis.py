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
from matplotlib import pyplot as plt
from pandas import read_csv, concat, DataFrame, date_range

from map.call_ee import is_authorized

ROI = 'users/dgketchum/boundaries/lolo_huc8'
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]


def get_modis_et(start, end):
    fc = ee.FeatureCollection(ROI)
    modis = ee.ImageCollection('MODIS/006/MOD16A2').filterDate(start, end)
    _list = modis.toList(modis.size()).getInfo()
    et_image = ee.Image(_list[0]['id']).select('ET').rename('{}_{}'.format(_list[0]['id'], 'ET'))
    for i in _list[1:]:
        et_image = et_image.addBands(ee.Image(i['id']).select('ET').rename('{}_{}'.format(i['id'], 'ET')))

    reduce = et_image.reduceRegions(collection=fc, reducer=ee.Reducer.mean())

    task = ee.batch.Export.table.toCloudStorage(
        reduce,
        description='lolo_mod16_et',
        bucket='wudr',
        fileNamePrefix='lolo_mod16_et',
        fileFormat='CSV')
    task.start()

    pet_image = ee.Image(_list[0]['id']).select('PET').rename('{}_{}'.format(_list[0]['id'], 'PET'))
    for i in _list[1:]:
        pet_image = pet_image.addBands(ee.Image(i['id']).select('PET').rename('{}_{}'.format(i['id'], 'PET')))

    reduce = pet_image.reduceRegions(collection=fc, reducer=ee.Reducer.mean())

    task = ee.batch.Export.table.toCloudStorage(
        reduce,
        description='lolo_mod16_pet',
        bucket='wudr',
        fileNamePrefix='lolo_mod16_pet',
        fileFormat='CSV')
    task.start()


def get_gridmet():
    fc = ee.FeatureCollection(ROI)
    for yr in TEST_YEARS:
        for param in ['pr', 'etr', 'tmmn', 'tmmx']:
            start = '{}-01-01'.format(yr)
            end = '{}-12-31'.format(yr)
            gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(start, end)
            _list = gridmet.toList(gridmet.size()).getInfo()

            image = ee.Image(_list[0]['id']).select(param).rename('{}_{}'.format(_list[0]['id'], param))

            track = []
            for i in _list[1:]:
                track.append('{}_{}'.format(i['id'], param))
                image = image.addBands(ee.Image(i['id']).select(param).rename('{}_{}'.format(i['id'], param)))

            reduce = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean())

            task = ee.batch.Export.table.toCloudStorage(
                reduce,
                description='lolo_gridmet_{}_{}'.format(param, yr),
                bucket='wudr',
                fileNamePrefix='lolo_gridmet_{}_{}'.format(param, yr),
                fileFormat='CSV')
            task.start()


def concatenate_lolo_tables(_dir, out_file):
    _list = [os.path.join(_dir, x) for x in os.listdir(_dir) if 'gridmet' in x]
    _list.sort()
    modis = [os.path.join(_dir, x) for x in os.listdir(_dir) if 'gridmet' not in x]
    mod_data = {}
    for m in modis:
        mod = read_csv(m).drop(columns=['.geo', 'system:index', 'Id'])
        dates = [datetime.strptime(x.split('/')[-1][0:10], '%Y_%m_%d') for x in mod.columns]
        param = mod.columns[0].split('/')[-1].split('_')[-1]
        vals = [x * 0.1 for x in list(mod.loc[0, :])]
        s = DataFrame(data=vals, index=dates)
        s.fillna(method='ffill', inplace=True)
        s = s.resample('D').asfreq()
        s = s / 8.
        s.interpolate(method='polynomial', order=3, inplace=True)
        s = s.reindex(date_range(dates[0], '{}-12-31'.format(TEST_YEARS[-1])))
        s.fillna(method='ffill', inplace=True)
        mod_data[param] = s
    df = concat(mod_data, sort=False, axis=1)
    df.columns = ['MOD16A2_{}'.format(x[0]) for x in list(df.columns)]

    for csv in _list:
        c = read_csv(csv).drop(columns=['.geo', 'system:index', 'Id'])
        param = 'gridmet_{}'.format(c.columns[0].split('/')[-1].split('_')[-1]).upper()
        dates = [datetime.strptime(x.split('/')[-1].split('_')[0], '%Y%m%d') for x in c.columns]
        vals = [x for x in list(c.loc[0, :])]
        c = DataFrame(data=vals, index=dates, columns=[param])
        c.fillna(method='ffill', inplace=True)
        if param not in df.columns:
            df[param] = c.reindex(index=df.index)
        else:
            df[param].loc[c.index] = c.values.reshape(c.values.shape[0], )

    df.to_csv(out_file)
    cols = [df['MOD16A2_ET'].resample('M').sum(), df['MOD16A2_PET'].resample('M').sum(),
            df['GRIDMET_ETR'].resample('M').sum(), df['GRIDMET_PR'].resample('M').sum(),
            df['GRIDMET_TMMN'].resample('M').mean(), df['GRIDMET_TMMX'].resample('M').mean()]
    s = [(x.name, x.values) for x in cols]
    _dct = {k: v for (k, v) in s}
    df = DataFrame(data=_dct, index=cols[0].index)
    df.to_csv(out_file.replace('daily', 'monthly'))


def time_series_modis(csv):
    df = read_csv(csv)
    df.plot()
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    is_authorized()
    # get_gridmet()
    # get_modis_et('{}-01-01'.format(TEST_YEARS[0]),
    #              '{}-12-31'.format(TEST_YEARS[-1]))
    lolo = os.path.join(home, 'IrrigationGIS', 'lolo')
    extracts = os.path.join(lolo, 'lolo_ee_extracts')
    out = os.path.join(lolo, 'tables', 'lolo_daily.csv')
    concatenate_lolo_tables(extracts, out)
    # table = os.path.join(home, 'IrrigationGIS', 'lolo', 'modis_loloee_export.csv')
    # time_series_modis(table)

# ========================= EOF ====================================================================
