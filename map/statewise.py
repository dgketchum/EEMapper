import os
from subprocess import check_call
import json
from collections import Counter
from pprint import pprint

import fiona
from call_ee import TARGET_STATES, E_STATES
from call_ee import is_authorized, request_band_extract, export_classification
from tables import concatenate_band_extract
from models import find_rf_variable_importance

ALL_STATES = TARGET_STATES + E_STATES

home = os.path.expanduser('~')
EE = os.path.join(home, 'miniconda3', 'envs', 'gcs', 'bin', 'earthengine')
GS = os.path.join(home, 'miniconda3', 'envs', 'gcs', 'bin', 'gsutil')
OGR = '/usr/bin/ogr2ogr'
AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

os.environ['GDAL_DATA'] = 'miniconda3/envs/gcs/share/gdal/'


def to_geographic(in_dir, out_dir, glob):
    in_shp = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp') and glob in x]
    for s in in_shp:
        out_shp = os.path.join(out_dir, os.path.basename(s))
        cmd = [OGR, '-f', 'ESRI Shapefile', '-t_srs', WGS, '-s_srs', AEA, out_shp, s]
        check_call(cmd)
        print(out_shp)


def push_points_to_asset(glob, state=None):

    shapes = []
    for s in ALL_STATES:
        if state and s != state:
            continue
        _file = os.path.join('gs://wudr/state_points', 'points_{}_{}.shp'.format(s, glob))
        shapes.append(_file)

    asset_ids = [os.path.basename(s).split('.')[0] for s in shapes]
    ee_root = 'users/dgketchum/points/state/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


def get_bands(pts_dir, glob, state=None):
    for s in ALL_STATES:
        if state and s != state:
            continue
        pts = os.path.join(pts_dir, 'points_{}_{}.shp'.format(s, glob))
        with fiona.open(pts, 'r') as src:
            years = list(set([x['properties']['YEAR'] for x in src]))
        print('get bands', s)
        pts = 'users/dgketchum/points/state/points_{}_{}'.format(s, glob)
        geo = 'users/dgketchum/boundaries/{}'.format(s)
        file_ = 'bands_{}_{}'.format(s, glob)
        request_band_extract(file_, pts, region=geo, years=years, filter_bounds=True, buffer=1e5)


def concatenate_bands(in_dir, out_dir, glob, state=None):
    for s in ALL_STATES:
        if state and s != state:
            continue
        print('\n{}\n'.format(s.upper()))
        glob_ = '{}_{}'.format(s, glob)
        concatenate_band_extract(in_dir, out_dir, glob=glob_)


def push_bands_to_asset(glob, state=None):
    shapes = []
    for s in ALL_STATES:
        if state and s != state:
            continue
        _file = os.path.join('gs://wudr/state_bands', '{}_{}.csv'.format(s, glob))
        shapes.append(_file)

    asset_ids = [os.path.basename(s).split('.')[0] for s in shapes]
    ee_root = 'users/dgketchum/bands/state/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


def variable_importance(in_dir, glob, importance_json=None):
    d = {}
    for s in ALL_STATES:
        try:
            print('\n{}\n'.format(s.upper()))
            csv = os.path.join(in_dir, '{}_{}.csv'.format(s, glob))
            variables = find_rf_variable_importance(csv)
            variables = [x for x in variables if x[1] > 0.05]
            d[s] = variables
            print(len(variables))
            print([x[0] for x in variables])
        except Exception as e:
            print(s, e)
            continue
    if importance_json:
        jsn = os.path.join(importance_json, 'variables_{}.json'.format(glob))
        with open(jsn, 'w') as fp:
            fp.write(json.dumps(d, indent=4, sort_keys=True))


def classify(out_coll, variable_dir, tables, years, glob, state=None):
    vars = os.path.join(variable_dir, 'variables_{}.json'.format(glob))
    with open(vars, 'r') as fp:
        d = json.load(fp)
    all_feat = []
    for k, v in d.items():
        if state and k != state:
            continue
        features = [f[0] for f in v]

        [all_feat.append(f) for f in features]
        table = os.path.join(tables, '{}_{}'.format(k, glob))
        geo = 'users/dgketchum/boundaries/{}'.format(k)
        # export_classification(out_name=k, table=table, asset_root=out_coll, region=geo,
        #                       years=years, input_props=features)
    hist = sorted(Counter(all_feat).items(), key=lambda x: x[1], reverse=True)
    pprint(hist)


if __name__ == '__main__':
    is_authorized()
    _glob = '7NOV2021'
    pt = '/media/research/IrrigationGIS/EE_extracts/point_shp'
    pt_wgs = os.path.join(pt, 'state_wgs')
    pt_aea = os.path.join(pt, 'state_aea')
    # to_geographic(pt_aea, pt_wgs, glob=_glob)
    # push_points_to_asset(glob=_glob)
    # get_bands(pt_aea, _glob)
    to_concat = '/media/research/IrrigationGIS/EE_extracts/to_concatenate/state'
    conctenated = '/media/research/IrrigationGIS/EE_extracts/concatenated/state'
    # concatenate_bands(to_concat, conctenated, glob=_glob, state='OR')
    imp_json = '/media/research/IrrigationGIS/EE_extracts/variable_importance'
    # variable_importance(conctenated, importance_json=imp_json, glob=_glob)
    # push_bands_to_asset(glob=_glob, state='OR')
    coll = 'users/dgketchum/IrrMapper/IrrMapper_sw'
    tables = 'users/dgketchum/bands/state'
    classify(coll, imp_json, tables, [2017], glob=_glob)
# ========================= EOF ====================================================================
