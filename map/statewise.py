import os
from subprocess import check_call
import json

from call_ee import TARGET_STATES, E_STATES, YEARS
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


def to_geographic(in_dir, out_dir):
    in_shp = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp')]
    for s in in_shp:
        out_shp = os.path.join(out_dir, os.path.basename(s))
        cmd = [OGR, '-f', 'ESRI Shapefile', '-t_srs', 'EPSG:4326', '-s_srs', AEA, out_shp, s]
        check_call(cmd)
        print(out_shp)


def push_points_to_asset(glob='5NOV2021'):
    shapes = [os.path.join('gs://wudr/state_points', 'points_{}_{}.shp'.format(s, glob)) for s in ALL_STATES]
    asset_ids = [os.path.basename(s).split('.')[0] for s in shapes]
    ee_root = 'users/dgketchum/points/state/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


def get_bands():
    for s in ALL_STATES:
        print('get bands', s)
        pts = 'users/dgketchum/points/state/points_{}_5NOV2021'.format(s)
        geo = 'users/dgketchum/boundaries/{}'.format(s)
        file_ = 'bands_{}_5NOV2021'.format(s)
        request_band_extract(file_, pts, region=geo, years=YEARS, filter_bounds=True)


def concatenate_bands(in_dir, out_dir, glob='5NOV2021'):
    for s in ALL_STATES:
        print('\n{}\n'.format(s.upper()))
        glob_ = '{}_{}'.format(s, glob)
        concatenate_band_extract(in_dir, out_dir, glob=glob_)


def push_bands_to_asset(glob='5NOV2021'):
    shapes = [os.path.join('gs://wudr/state_bands', '{}_{}.csv'.format(s, glob)) for s in ALL_STATES[:3]]
    asset_ids = [os.path.basename(s).split('.')[0] for s in shapes]
    ee_root = 'users/dgketchum/bands/state/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


def variable_importance(in_dir, glob='5NOV2021', importance_json=None):
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
            break
    if importance_json:
        jsn = os.path.join(importance_json, 'variables_{}.json'.format(glob))
        with open(jsn, 'w') as fp:
            fp.write(json.dumps(d, indent=4, sort_keys=True))


def classify(out_coll, variable_dir, tables, years, glob='5NOV2021'):
    vars = os.path.join(variable_dir, 'variables_{}.json'.format(glob))
    with open(vars, 'r') as fp:
        d = json.load(fp)
    for k, v in d.items():
        features = [f[0] for f in v]
        table = os.path.join(tables, '{}_{}'.format(k, glob))
        geo = 'users/dgketchum/boundaries/{}'.format(k)
        export_classification(out_name=k, table=table, asset_root=out_coll, region=geo,
                              years=years, input_props=features)


if __name__ == '__main__':
    is_authorized()
    pt = '/media/research/IrrigationGIS/EE_extracts/point_shp'
    pt_wgs = os.path.join(pt, 'state_wgs')
    pt_aea = os.path.join(pt, 'state_aea')
    # to_geographic(pt_aea, pt_wgs)
    # push_points_to_asset()
    # get_bands()
    to_concat = '/media/research/IrrigationGIS/EE_extracts/to_concatenate/state'
    conctenated = '/media/research/IrrigationGIS/EE_extracts/concatenated/state'
    concatenate_bands(to_concat, conctenated)
    imp_json = '/media/research/IrrigationGIS/EE_extracts/variable_importance'
    # variable_importance(conctenated, importance_json=None)
    # push_bands_to_asset()
    coll = 'users/dgketchum/IrrMapper/IrrMapper_sw'
    tables = 'users/dgketchum/bands/state'
    # classify(coll, imp_json, tables, [2017])
# ========================= EOF ====================================================================
