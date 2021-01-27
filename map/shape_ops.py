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
import json
from datetime import datetime
from collections import OrderedDict
from random import shuffle
from copy import deepcopy

import numpy as np

import fiona
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
from geopandas import read_file


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


def get_irr_years(shp, out_file=None):
    d = {}
    ct = 0
    with fiona.open(shp) as src:
        d = {int(f['properties']['POLYFID']): {} for f in src}
        feats = [x for x in src]

    for f in feats:
        ct += 1
        y = f['properties']['YEAR']
        fid = int(f['properties']['FID'])
        pfid = int(f['properties']['POLYFID'])
        if fid not in d[pfid].keys():
            d[pfid][fid] = [y]
        elif y not in d[pfid][fid]:
            d[pfid][fid].append(y)

    if out_file:
        with open(out_file, 'w') as f:
            json.dump(d, f)


def build_master_json(shp, out_file=None):
    d = {}
    ct = 0
    with open(os.path.join(os.getcwd(), 'data', 'irr_shards.json')) as j:
        irr = json.loads(j.read())
    with fiona.open(shp) as src:
        for f in src:
            if not f['properties']['IRR'] == 1:
                continue
            pfid = f['properties']['POLYFID']
            if pfid not in d.keys():
                d[f['properties']['POLYFID']] = {}

    with fiona.open(shp) as src:
        for f in src:
            ct += 1
            fid = f['properties']['FID']
            pfid = f['properties']['POLYFID']
            years = f['properties']['YEAR']
            split = f['properties']['SPLIT']
            state = f['properties']['STUSPS']
            if fid not in d[pfid].keys():
                d[pfid][fid] = (split, state, fid)
            elif y not in d[pfid][fid]:
                d[pfid][fid][-1].append(y)
    if out_file:
        with open(out_file, 'w') as f:
            json.dump(d, f)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # shpp = '/media/research/IrrigationGIS/EE_sample/grids_aea/grids_5070/irr_shard.shp'
    # out_json = 'data/irr_shards.json'
    # get_irr_years(shpp, out_json)

    shpp = '/media/research/IrrigationGIS/EE_sample/grids_aea/grids_5070/relevant_shard.shp'
    out_json = 'data/irr_shards.json'
    build_master_json(shpp, out_json)
# ========================= EOF ====================================================================
