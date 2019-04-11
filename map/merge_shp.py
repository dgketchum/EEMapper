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
from shapely.geometry import Polygon


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
    years = []
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('YEAR', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        for s in file_list:
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


def count_acres(shp):
    acres = 0.0
    with fiona.open(shp, 'r') as s:
        for feat in s:
            a = feat['properties']['ACRES']
            acres += a
        print(acres)


def get_area(shp):
    """use AEA conical for sq km result"""

    print(shp)
    area = 0.0
    geos = []
    dupes = 0
    unq = 0
    with fiona.open(shp, 'r') as s:
        for feat in s:

            if feat['geometry']['type'] == 'Polygon':
                coords = feat['geometry']['coordinates'][0]

                if 'irrigated' in shp and coords not in geos or 'irrigated' not in shp:
                    geos.append(coords)
                    a = Polygon(coords).area / 1e6
                    area += a
                    unq += 1
                elif 'irrigated' in shp and coords in geos:
                    dupes += 1
                else:
                    raise TypeError

            elif feat['geometry']['type'] == 'MultiPolygon':
                for l_ring in feat['geometry']['coordinates']:
                    coords = l_ring[0]
                    if 'irrigated' in shp and coords not in geos or 'irrigated' not in shp:
                        geos.append(coords)
                        a = Polygon(coords).area / 1e6
                        area += a
                        unq += 1
                    elif 'irrigated' in shp and coords in geos:
                        dupes += 1
                    else:
                        raise TypeError
            else:
                raise TypeError
        print(area)
        print(area * 247.105)


def clean_geometry(in_shp, out_shp):
    out_list = []
    ct = 0
    meta = fiona.open(in_shp).meta
    for feat in fiona.open(in_shp, 'r'):
        if feat['geometry']['type'] == 'Polygon':
            ct += 1
            out_list.append(feat)
        if feat['geometry']['type'] == 'Multipoint':
            print(feat)
    with fiona.open(out_shp, 'w', **meta) as output:
        for feat in out_list:
            output.write(feat)
    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')

    s_shp = os.path.join(home, 'IrrigationGIS', 'training_raw', 'irrigated', 'CA', 'CA_v2.shp')
    o_shp = os.path.join(home, 'IrrigationGIS', 'training_raw', 'irrigated', 'CA', 'CA_v2_gt50ac_crops_clean.shp')
    clean_geometry(s_shp, o_shp)

    s_shp = os.path.join(home, 'IrrigationGIS', 'training_raw', 'CO', 'Div1_Irrig_2015.shp')
    o_shp = os.path.join(home, 'IrrigationGIS', 'training_raw', 'irrigated', 'CA', 'CA_v2_gt50ac_crops_clean.shp')
    clean_geometry(s_shp, o_shp)
# ========================= EOF ====================================================================
