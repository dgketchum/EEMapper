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
from collections import OrderedDict
from random import shuffle
from copy import deepcopy

import numpy as np

import fiona
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
from geopandas import read_file


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
                    mod_geo = geo.buffer(-128 * 30., resolution=1, cap_style=3, )
                    f['geometry'] = mapping(mod_geo)
                output.write(f)
                ct += 1

    print('{} in, {} out'.format(len_, ct))


def select_wetlands(_file, out_file):
    """ Buffer out linear wetlands features.

    Wetlands features are from National Wetlands Data Inventorty, from
    https://www.fws.gov/wetlands/data/State-Downloads.html

    """
    out_feats = 0

    with fiona.open(_file, 'r') as input:
        meta = input.meta
        meta['schema']['properties'] = OrderedDict([('OBJECTID', 'int:10'),
                                                    ('WETLAND_TY', 'str:50'),
                                                    ('AREA_SQM', 'float:19.11')])
        print('{} input features'.format(input.__len__()))

    with fiona.open(out_file, 'w', **meta) as output:

        with fiona.open(_file, 'r') as input:
            s = datetime.now()
            for f in input:
                geo = shape(f['geometry'])

                if isinstance(geo, Polygon):
                    a = geo.area
                    b = geo.length
                    if a == 0. or b == 0.:
                        continue

                    if a / b > 12.0:
                        out_feats += 1
                        out_feature = {'type': 'Feature', 'geometry': mapping(geo), 'id': out_feats,
                                       'properties': OrderedDict([('OBJECTID', out_feats),
                                                                  ('WETLAND_TY', f['properties']['WETLAND_TY']),
                                                                  ('AREA_SQM', a / 1e6)])}
                        output.write(out_feature)

                elif isinstance(geo, MultiPolygon):
                    for p in list(geo):
                        a = p.area
                        b = p.length
                        if a == 0. or b == 0.:
                            continue

                        if a / b > 12.0:
                            out_feats += 1
                            out_feature = {'type': 'Feature', 'geometry': mapping(p), 'id': out_feats,
                                           'properties': OrderedDict([('OBJECTID', out_feats),
                                                                      ('WETLAND_TY', f['properties']['WETLAND_TY']),
                                                                      ('AREA_SQM', a / 1e6)])}
                            output.write(out_feature)

    print('{} out in {} sec \n {} {}'.format(out_feats, (datetime.now() - s).seconds, _file, out_file))
    return out_feats


def rm_dupe_geometry():
    in_shp = '/home/dgketchum/IrrigationGIS/EE_sample/centroids/irrigated_27JUL2020.shp'
    out_shp = '/home/dgketchum/IrrigationGIS/EE_sample/centroids/irrigated_13AUG2020.shp'
    df = read_file(in_shp)
    print(df.shape)
    # df = df[['SOURCE', 'geometry']]
    df.drop_duplicates(['YEAR', 'geometry'], keep='first', inplace=True)
    print(df.shape)
    out = os.path.join(out_shp, out_shp)
    df.to_file(out)


def get_fid(in_shp):
    with fiona.open(in_shp, 'r') as input_:
        meta = input_.meta
        features = [f['properties']['FID'] for f in input_]
    print(features)


def regular_grid(shp, out_shp):
    out_feats = 0
    offset = 7680.
    places = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
    offset = [(offset * p[0], offset * p[1]) for p in places]
    with fiona.open(shp, 'r') as src:
        meta = src.meta
        meta['schema']['geometry'] = 'Point'
        meta['schema']['properties'] = OrderedDict([('POLYFID', 'int:10'),
                                                    ('PTFID', 'int:10')])
        print('{} input features'.format(src.__len__()))
        with fiona.open(out_shp, 'w', **meta) as dst:
            for feat in src:
                cent = shape(feat['geometry']).centroid
                points = [(cent.x + p[0], cent.y + p[1]) for p in offset]
                fid = feat['properties']['FID']
                for l in points:
                    out_feats += 1
                    out_feat = {'type': 'Feature', 'properties': OrderedDict([('PTFID', out_feats), ('POLYFID', fid)]),
                                'geometry': mapping(Point(l))}
                    dst.write(out_feat)
    print(out_feats)


def get_years(shp):
    d = {}
    with fiona.open(shp) as src:
        for f in src:
            s = f['properties']['STUSPS']
            y = f['properties']['YEAR']
            try:
                if y in d[s]:
                    continue
                d[s].append(y)
            except KeyError:
                d[s] = [y]
    print(d)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    shpp = os.path.join(home, 'Downloads', 'irrigation_states.shp')
    get_years(shpp)
# ========================= EOF ====================================================================
