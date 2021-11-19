import os
from subprocess import check_call
import json
import time
from collections import Counter
from pprint import pprint

import fiona
from call_ee import TARGET_STATES, E_STATES
from call_ee import is_authorized, request_band_extract, export_classification
from tables import concatenate_band_extract
from models import find_rf_variable_importance
from variable_importance import dec_2020_variables
from assets import list_assets

ALL_STATES = TARGET_STATES + E_STATES

home = os.path.expanduser('~')
EE = os.path.join(home, 'miniconda3', 'envs', 'gcs', 'bin', 'earthengine')
GS = os.path.join(home, 'miniconda3', 'envs', 'gcs', 'bin', 'gsutil')
OGR = '/usr/bin/ogr2ogr'
AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

os.environ['GDAL_DATA'] = 'miniconda3/envs/gcs/share/gdal/'

DRYLAND_STATES = ['CO', 'ID', 'MT', 'OR', 'WA']


def to_geographic(in_dir, out_dir, glob, state):
    in_shp = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp') and glob in x and state in x]
    for s in in_shp:
        out_shp = os.path.join(out_dir, os.path.basename(s))
        cmd = [OGR, '-f', 'ESRI Shapefile', '-t_srs', WGS, '-s_srs', AEA, out_shp, s]
        check_call(cmd)
        print(out_shp)


def push_points_to_asset(_dir, glob, state, bucket):
    local_f = os.path.join(_dir, 'points_{}_{}.shp'.format(state, glob))
    bucket = os.path.join(bucket, 'state_points')
    _file = os.path.join(bucket, 'points_{}_{}.shp'.format(state, glob))
    cmd = [GS, 'cp', local_f, _file]
    check_call(cmd)

    asset_id = os.path.basename(local_f).split('.')[0]
    ee_root = 'users/dgketchum/points/state/'
    cmd = [EE, 'upload', 'table', '-f', '--asset_id={}{}'.format(ee_root, asset_id), _file]
    check_call(cmd)
    print(asset_id, _file)


def get_bands(pts_dir, glob, state):
    pts = os.path.join(pts_dir, 'points_{}_{}.shp'.format(state, glob))
    with fiona.open(pts, 'r') as src:
        years = list(set([x['properties']['YEAR'] for x in src]))
    print('get bands', state)
    pts = 'users/dgketchum/points/state/points_{}_{}'.format(state, glob)
    geo = 'users/dgketchum/boundaries/{}'.format(s)
    file_ = 'bands_{}_{}'.format(s, glob)
    request_band_extract(file_, pts, region=geo, years=years, filter_bounds=True, buffer=1e5)


def concatenate_bands(in_dir, out_dir, glob, state):
    print('\n{}\n'.format(state.upper()))
    glob_ = '{}_{}'.format(state, glob)
    concatenate_band_extract(in_dir, out_dir, glob=glob_)


def push_bands_to_asset(_dir, glob, state, bucket):
    shapes = []
    local_f = os.path.join(_dir, '{}_{}.csv'.format(state, glob))
    bucket = os.path.join(bucket, 'state_bands')
    _file = os.path.join(bucket, '{}_{}.csv'.format(state, glob))
    cmd = [GS, 'cp', local_f, _file]
    check_call(cmd)
    shapes.append(_file)
    asset_ids = [os.path.basename(shp).split('.')[0] for shp in shapes]
    ee_root = 'users/dgketchum/bands/state/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '-f', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


def variable_importance(in_dir, glob, importance_json=None, state=None):
    d = {}
    for s in ALL_STATES:
        if state and s != state:
            continue
        try:
            print('\n{}\n'.format(s.upper()))
            csv = os.path.join(in_dir, '{}_{}.csv'.format(s, glob))
            variables = find_rf_variable_importance(csv)
            # variables = [x for x in variables[:50]]
            d[s] = variables
            print(len(variables))
            pprint([x[0] for x in variables])
        except Exception as e:
            print(s, e)
            continue
    if importance_json:
        jsn = os.path.join(importance_json, 'variables_{}_{}.json'.format(state, glob))
        with open(jsn, 'w') as fp:
            fp.write(json.dumps(d, indent=4, sort_keys=True))


def remove_image(image):
    cmd = [EE, 'rm', image]
    check_call(cmd)
    print('remove ', image)


def classify(out_coll, variable_dir, tables, years, glob, state):
    vars = os.path.join(variable_dir, 'variables_{}_{}.json'.format(state, glob))
    with open(vars, 'r') as fp:
        d = json.load(fp)
    features = [f[0] for f in d[state]]
    for f in ['elevation', 'slope', 'tpi_150', 'tpi_250', 'tpi_1250']:
        if f not in features:
            features.append(f)
    for f in dec_2020_variables():
        if f not in features:
            features.append(f)
    var_txt = os.path.join(variable_dir, '{}_vars.txt'.format(state))
    with open(var_txt, 'w') as fp:
        for f in features:
            fp.write('{}\n'.format(f))
    table = os.path.join(tables, '{}_{}'.format(state, glob))
    geo = 'users/dgketchum/boundaries/{}'.format(state)
    export_classification(out_name=state, table=table, asset_root=out_coll, region=geo,
                          years=years, input_props=features, bag_fraction=0.5)
    hist = sorted(Counter(features).items(), key=lambda x: x[1], reverse=True)
    pprint(hist)


def clean_deprecated_data(coll, pt_geo, pt_proj, bucket, check_all=False):
    l = list_assets(coll)
    for s in ALL_STATES:
        pass
    for i in l:
        pass


if __name__ == '__main__':
    is_authorized()
    _glob = '18NOV2021'
    _bucket = 'gs://wudr'

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/IrrigationGIS'

    pt = '/media/research/IrrigationGIS/EE_extracts/point_shp'
    pt_wgs = os.path.join(pt, 'state_wgs')
    pt_aea = os.path.join(pt, 'state_aea')

    extracts = os.path.join(root, 'EE_extracts')
    to_concat = os.path.join(extracts, 'to_concatenate/state')
    conctenated = os.path.join(extracts, 'concatenated/state')
    imp_json = os.path.join(extracts, 'variable_importance')

    # coll = 'users/dgketchum/IrrMapper/IrrMapper_sw'
    coll = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp_'
    tables = 'users/dgketchum/bands/state'

    # clean_deprecated_data(coll, pt_wgs, pt_aea, _bucket)
    for s in ['OR']:
        to_geographic(pt_aea, pt_wgs, glob=_glob, state=s)
        push_points_to_asset(pt_wgs, glob=_glob, state=s, bucket=_bucket)
        # get_bands(pt_aea, _glob, state=s)

        # concatenate_bands(to_concat, conctenated, glob=_glob, state=s)
        # variable_importance(conctenated, importance_json=imp_json, glob=_glob, state=s)
        # push_bands_to_asset(conctenated, glob=_glob, state=s, bucket=_bucket)

        # classify(coll, imp_json, tables, [x for x in range(2015, 2022)], glob=_glob, state=s)
# ========================= EOF ====================================================================
