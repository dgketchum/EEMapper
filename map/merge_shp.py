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

import fiona


def fiona_merge_MT(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'properties': OrderedDict(
        [('Irr_2009', 'int:5'), ('Irr_2010', 'int:5'), ('Irr_2011', 'int:5'),
         ('Irr_2012', 'int:5'), ('Irr_2013', 'int:5')]),
        'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for features in fiona.open(s):
                try:
                    feat = {'properties': OrderedDict([('Irr_2009', int(features['properties']['Irr_2009'])),
                                                       ('Irr_2010', int(features['properties']['Irr_2010'])),
                                                       ('Irr_2011', int(features['properties']['Irr_2011'])),
                                                       ('Irr_2012', int(features['properties']['Irr_2012'])),
                                                       ('Irr_2013', int(features['properties']['Irr_2013']))]),
                            'geometry': features['geometry']}
                    output.write(feat)
                except Exception as e:
                    pass

    return None


def fiona_merge(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for features in fiona.open(s):
                output.write(features)

    return None


def fiona_merge_attribute(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('YEAR', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            year, source = int(s.split('.')[0][-4:]), os.path.basename(s.split('.')[0][:-5])
            for feat in fiona.open(s):
                feat = {'type': 'Feature', 'properties': {'SOURCE': source, 'YEAR': year},
                        'geometry': feat['geometry']}
                output.write(feat)


def fiona_merge_no_attribute(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        []), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for feat in fiona.open(s):
                feat = {'type': 'Feature', 'properties': {},
                        'geometry': feat['geometry']}
                output.write(feat)


def get_list(_dir):
    l = []
    for path, subdirs, files in os.walk(_dir):
        for name in files:
            p = os.path.join(path, name)
            if p not in l and p.endswith('.shp'):
                l.append(p)
    return l


if __name__ == '__main__':
    home = os.path.expanduser('~')
    s_dir = os.path.join(home, 'IrrigationGIS', 'training_raw', 'uncultivated')
    l = get_list(s_dir)
    fiona_merge_no_attribute(os.path.join(s_dir, 'uncultivated.shp'), l)
    # s_dir = os.path.join(home, 'IrrigationGIS', 'training_raw', 'irrigated', 'filtered_shapefiles')
    # _dir = [os.path.join(s_dir, x) for x in os.listdir(s_dir)if x.endswith('.shp')]
    # o_dir = os.path.join(home, 'IrrigationGIS', 'training_raw', 'merged_attributed')
    # fiona_merge_attribute(os.path.join(o_dir, 'irr_merge.shp'), _dir)

# ========================= EOF ====================================================================
