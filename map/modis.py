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
from pprint import pprint
from pandas import read_csv, Series
from datetime import datetime
from matplotlib import pyplot as plt
import ee

from map.call_ee import is_authorized

ROI = 'users/dgketchum/boundaries/lolo_huc8'
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]


def get_modis_et(start, end):
    fc = ee.FeatureCollection(ROI)
    coll = ee.ImageCollection('MODIS/006/MOD16A2').filterDate(start, end)
    _list = coll.toList(coll.size()).getInfo()
    image = ee.Image(_list[0]['id']).select('ET').rename('{}_{}'.format(_list[0]['id'], 'ET'))
    for i in _list[1:]:
        image = image.addBands(ee.Image(i['id']).select('ET').rename('{}_{}'.format(i['id'], 'ET')))

    reduce = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean())

    task = ee.batch.Export.table.toCloudStorage(
        reduce,
        description='modis_lolo',
        bucket='wudr',
        fileNamePrefix='modis_lolo',
        fileFormat='KML')
    task.start()


def time_series_modis(csv):
    df = read_csv(csv).drop(columns=['.geo', 'system:index', 'Id'])
    dates = [datetime.strptime(x[-13:-3], '%Y_%m_%d') for x in df.columns]
    vals = [x * 0.1 for x in list(df.loc[0, :])]
    s = Series(data=vals, index=dates)
    s.fillna(method='ffill', inplace=True)
    ns = s.resample('D').asfreq()
    ns = ns / 8.
    ns.interpolate(method='polynomial', order=3, inplace=True)
    ns.plot()
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    is_authorized()
    # get_modis_et('{}-01-01'.format(TEST_YEARS[0]),
    #              '{}-12-31'.format(TEST_YEARS[-1]))
    table = os.path.join(home, 'IrrigationGIS', 'lolo', 'modis_loloee_export.csv')
    time_series_modis(table)

# ========================= EOF ====================================================================
