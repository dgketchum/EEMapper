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

import ee

from map.assets import list_assets
from map.call_ee import is_authorized

ASSET_ROOT = 'users/dgketchum/classy'
STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
TARGET_STATES = ['OR']
BOUNDARIES = 'users/dgketchum/boundaries'
ASSET_ROOT = 'users/dgketchum/first_detected'


def first_detection():
    # this doesn't work, but it works in Code Editor
    for state in TARGET_STATES:
        bounds = os.path.join(BOUNDARIES, state)
        roi = ee.FeatureCollection(bounds)
        mask = roi.geometry().bounds().getInfo()['coordinates']
        image_list = list_assets('users/dgketchum/classy')
        out_images = []
        for yr in range(1986, 2017):
            yr_img = [x for x in image_list if x.endswith(str(yr))]
            coll = ee.ImageCollection(yr_img)
            classed = coll.mosaic().select('classification').remap([0, 1, 2, 3],
                                                                   [yr, 0, 0, 0]).rename('{}_min'.format(yr))
            out_images.append(classed)

        coll = ee.ImageCollection(out_images)
        img = coll.reduce(ee.Reducer.minMax()).rename('min', 'max')
        pprint(img.getInfo())
        task = ee.batch.Export.image.toAsset(
            image=img,
            description='{}'.format(state),
            assetId=os.path.join(ASSET_ROOT, '{}'.format(state)),
            fileNamePrefix='{}'.format(state),
            region=mask,
            scale=30,
            maxPixels=1e10)

        print(state)
        task.start()
        break


if __name__ == '__main__':
    home = os.path.expanduser('~')
    is_authorized()
    first_detection()
# ========================= EOF ====================================================================
