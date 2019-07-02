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

import ee

ROI = 'users/dgketchum/boundaries/lolo_huc8'
TEST_YEARS = [2017, 2018]


def get_modis_et(start, end, annual=True):
    fc = ee.FeatureCollection(ROI)
    tot = ee.ImageCollection('MODIS/006/MOD16A2').\
        filter(ee.Filter.date(start, end)).select('ET').sum().multiply(0.1)
    tot = tot.multiply(ee.Image.pixelArea())
    reduce = tot.reduceRegions(collection=fc,
                               reducer=ee.Reducer.sum(),
                               scale=30)
    task = ee.batch.Export.table.toCloudStorage(
        reduce,
        description='modis_lolo_{}_'.format(year),
        bucket='wudr',
        fileNamePrefix='modis_lolo_{}_'.format(year),
        fileFormat='CSV')
    task.start()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    for year in [str(x) for x in range(2014, 2019)]:
        get_modis_et('{}-01-01'.format(year), '{}-12-31'.format(year))

    pass
# ========================= EOF ====================================================================
