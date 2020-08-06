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
from collections import OrderedDict
from random import shuffle

import fiona
from shapely.geometry import shape, mapping


def fiona_merge_attribute(out_shp, file_list):
    """ Use to merge and keep the year attribute """
    years = []
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('YEAR', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        for s in file_list:
            print(s)
            year, source = int(s.split('.')[0][-4:]), os.path.basename(s.split('.')[0][:-5])
            if year not in years:
                years.append(year)
            for feat in fiona.open(s):
                feat = {'type': 'Feature', 'properties': {'SOURCE': source, 'YEAR': year},
                        'geometry': feat['geometry']}
                output.write(feat)
                ct += 1
        print(sorted(years))


def fiona_merge_no_attribute(out_shp, file_list):
    """ Use to merge shapefiles with no attributes """
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        []), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        none_geo = 0
        inval_geo = 0
        for s in file_list:
            mgrs = s.split('/')[-1].strip('.shp')
            print(mgrs)
            for feat in fiona.open(s):

                if not feat['geometry']:
                    none_geo += 1
                    continue
                geo = shape(feat['geometry'])
                if not geo.is_valid:
                    inval_geo += 1
                    continue
                if geo.area == 0.0:
                    continue
                if geo.bounds[0] < -180.0:
                    inval_geo += 1
                    print(s)
                    continue
                ct += 1
                feat = {'type': 'Feature', 'properties': {},
                        'geometry': feat['geometry']}
                output.write(feat)

    print('wrote {}, {} none, {} invalid'.format(ct, none_geo, inval_geo))


def test_train_val_split(grid, train=0.6, test=0.2, valid=0.2):
    """Split grids into test, train, split.
        Interior-buffer training grids.
        """
    with fiona.open(grid, 'r') as input:
        meta = input.meta
        features = [f for f in input]
    shuffle(features)

    len_ = len(features)
    train_, test_, valid_ = features[:int(len_ * train)], \
                            features[int(len_ * train): -int(len_ * valid)], \
                            features[-int(len_ * valid):]

    ct = 0
    for out_suffix, feature_list in zip(['_train.', '_test.', '_valid.'], [train_, test_, valid_]):
        out_name = grid.replace('.', out_suffix)
        with fiona.open(out_name, 'w', **meta) as output:
            for f in feature_list:
                if out_suffix == '_train.':
                    geo = shape(f['geometry'])
                    mod_geo = geo.buffer(-128 * 30., resolution=1, cap_style=3,)
                    f['geometry'] = mapping(mod_geo)
                output.write(f)
                ct += 1

    print('{} in, {} out'.format(len_, ct))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    grid = os.path.join(home, 'IrrigationGIS', 'EE_sample', 'grid', 'grid_training_aea.shp')
    test_train_val_split(grid)
# ========================= EOF ====================================================================
