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

import fiona
from shapely.geometry import shape, mapping, Polygon, MultiPolygon


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
    counter = [x for x in range(10000, int(3e6), 10000)]

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
                                                                  ('AREA_SQM', a)])}
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
                                                                      ('AREA_SQM', a)])}
                            output.write(out_feature)

                if out_feats in counter:
                    print(out_feats, (datetime.now() - s).seconds)

    print('{} out in {} sec \n {} {}'.format(out_feats, (datetime.now() - s).seconds, _file, out_file))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    remote = '/media/research'
    wet = os.path.join(remote, 'IrrigationGIS', 'wetlands', 'raw_shp')
    files_ = [os.path.join(wet, x) for x in os.listdir(wet) if '.shp' in x]
    wet_sel = os.path.join(remote, 'IrrigationGIS', 'wetlands', 'feature_buf')
    for f in files_:
        out = f.replace('raw_shp', 'feature_buf')
        select_wetlands(f, out)
# ========================= EOF ====================================================================
