import os
import sys
import time
import json

import ee

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from subprocess import check_call
import random

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from map.call_ee import is_authorized, stack_bands, BOUNDARIES

sys.setrecursionlimit(5000)

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = '/home/dgketchum/miniconda/envs/irmp/bin/earthengine'
GS = '/home/dgketchum/miniconda/envs/irmp/bin/gsutil'

OGR = '/usr/bin/ogr2ogr'

AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

FEATURE_COLS_DROP = ['system:index', '.geo']

TRAINING_DATA = {'AZ': 'AZ_24NOV2021', 'CA': 'CA_14NOV2021',
                 'CO': 'CO_10NOV2021', 'ID': 'ID_10NOV2021',
                 'KS': 'KS_7NOV2021', 'MT': 'MT_15NOV2021',
                 'ND': 'ND_7NOV2021', 'NE': 'NE_7NOV2021',
                 'NM': 'NM_7NOV2021', 'NV': 'NV_7NOV2021',
                 'OK': 'OK_7NOV2021', 'OR': 'OR_22NOV2021',
                 'SD': 'SD_7NOV2021', 'TX': 'TX_7NOV2021',
                 'UT': 'UT_10NOV2021', 'WA': 'WA_10NOV2021',
                 'WY': 'WY_7NOV2021'}


def to_geographic(in_dir, out_dir, states, mgrs_path, n_samples=None):
    """Creates a shapefile for each state with MGRS tile info."""
    out_shapefiles = []
    for state in states:
        print(f"Processing points for {state}...")
        in_shp = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if
                  x.endswith('.shp') and TRAINING_DATA[state] in x][0]
        mgrs = gpd.read_file(mgrs_path)

        points = gpd.read_file(in_shp)
        points = points.sjoin(mgrs, how="inner")
        points = points.to_crs(epsg=4326)
        points.drop(columns=['index_right'], inplace=True)
        points['STUSPS'] = state
        points['NEW_YEAR'] = [random.randint(2017, 2024) for _ in range(len(points))]

        if n_samples:
            points = points.groupby('POINT_TYPE', group_keys=False).apply(lambda x: x.sample(n=n_samples))

        points.index = range(len(points))
        points['FID'] = [f'{row['STUSPS']}_{i:06d}' for i, row in points.iterrows()]

        out_shp = os.path.join(out_dir, f'{TRAINING_DATA[state]}.shp')
        points.to_file(out_shp)
        out_shapefiles.append(out_shp)
        print(f"Wrote {out_shp}")

    return out_shapefiles


def push_points_to_asset(_dir, shapefile, bucket):
    """Uploads a single state's shapefile to a unique Earth Engine asset."""
    shp_name = os.path.basename(shapefile).replace('.shp', '')
    asset_id = f"users/dgketchum/points/state_mgrs/{shp_name}"

    print(f"Uploading {shp_name} to GCS...")
    gcs_path = os.path.join(bucket, 'redevelopment_points', os.path.basename(shapefile))
    cmd = [GS, 'cp', shapefile, gcs_path]
    check_call(cmd)

    for ext in ['prj', 'shx', 'dbf', 'cpg']:
        sidecar_file = shapefile.replace('.shp', f'.{ext}')
        if os.path.exists(sidecar_file):
            cmd = [GS, 'cp', sidecar_file, os.path.join(bucket, 'redevelopment_points')]
            check_call(cmd)

    print(f"Uploading {gcs_path} to Earth Engine asset {asset_id}")
    cmd = [EE, 'upload', 'table', '-f', f'--asset_id={asset_id}', gcs_path]
    check_call(cmd)
    print(f"Upload complete for {asset_id}")
    return asset_id


def get_bands(shp_dir, extract_modern=False, check_dir=None, diagnose=False):
    shapefiles = [os.path.join(shp_dir, f) for f in os.listdir(shp_dir) if f.endswith('.shp')]
    points_dfs = [gpd.read_file(shp) for shp in shapefiles]
    points_df = pd.concat(points_dfs)

    if extract_modern:
        file_prefix = 'irrmapper_redev/bands_modern'

    else:
        file_prefix = 'irrmapper_redev/bands'

    mgrs_ee_tiles = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    mgrs_tiles = points_df['MGRS_TILE'].unique()

    states = points_df['STUSPS'].unique()

    for state in states:
        state_df = points_df[points_df['STUSPS'] == state]

        for tile in mgrs_tiles:
            tile_df = state_df[state_df['MGRS_TILE'] == tile]

            if extract_modern:
                target_year_col = 'NEW_YEAR'
                years = sorted(list(tile_df[target_year_col].unique()))
            else:
                target_year_col = 'YEAR'
                years = sorted(list(tile_df[target_year_col].unique()))

            for year in years:
                year_df = tile_df[tile_df[target_year_col] == year]

                if year_df.empty:
                    continue

                desc = f'bands_{tile}_{state}_{year}'

                # densest extract test
                # if extract_modern and not desc.startswith(f'bands_12TUL_UT_'):
                #     continue
                #
                # elif not extract_modern and desc != f'bands_12TUL_UT_2009':
                #     continue

                if check_dir:
                    if os.path.exists(os.path.join(check_dir, f'{desc}.csv')):
                        print(f"Skipping {desc}, already exists.")
                        continue

                feature_coll = ee.FeatureCollection(year_df.__geo_interface__)

                roi = mgrs_ee_tiles.filterMetadata('MGRS_TILE', 'equals', tile)
                region = ee.FeatureCollection(roi)

                stack = stack_bands(year, region, southern=False)

                # if tables are coming out empty, use this to find missing bands
                if diagnose:
                    filtered = ee.FeatureCollection([feature_coll.first()])
                    bad_ = []
                    bands = stack.bandNames().getInfo()
                    for b in bands:
                        stack_ = stack.select([b])

                        def sample_regions(i, points):
                            red = ee.Reducer.toCollection(i.bandNames())
                            reduced = i.reduceRegions(points, red, 30, stack_.select(b).projection())
                            fc = reduced.map(lambda f: ee.FeatureCollection(f.get('features'))
                                             .map(lambda q: q.copyProperties(f, None, ['features'])))
                            return fc.flatten()

                        data = sample_regions(stack_, filtered)
                        try:
                            print(b, data.getInfo()['features'][0]['properties'][b])
                        except Exception as e:
                            print(b, 'not there', e)
                            bad_.append(b)
                    print(bad_)
                    return None

                selectors = ['FID', 'POINT_TYPE', 'YEAR', 'NEW_YEAR', 'MGRS_TILE', 'STUSPS']

                plot_sample_regions = stack.sampleRegions(
                    collection=feature_coll,
                    properties=selectors,
                    scale=30,
                    tileScale=16)

                task = ee.batch.Export.table.toCloudStorage(
                    plot_sample_regions,
                    description=desc,
                    bucket='wudr',
                    fileNamePrefix=f'{file_prefix}/{desc}',
                    fileFormat='CSV')

                try:
                    task.start()
                except ee.ee_exception.EEException as e:
                    print('{}, waiting on '.format(e), desc, '......')
                    time.sleep(600)
                    task.start()

                print(f'{file_prefix}/{desc}')


def process_bands_to_parquet(in_dir, out_dir, category_counters=None, overwrite=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_list = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]

    if category_counters:
        categories = {c: set() for c in category_counters}
    else:
        categories = None

    ct = 0
    for f in tqdm(file_list):
        if not f.endswith('.csv'):
            continue

        try:
            df = pd.read_csv(f)
            df.set_index('FID', inplace=True)
        except pd.errors.EmptyDataError:
            os.remove(f)
            continue

        if categories:
            [categories[c].add(v) for c in categories.keys() for v in df[c].unique()]

        for fid, row in df.iterrows():
            try:
                year = int(row['YEAR'])
                pt_type = int(row['POINT_TYPE'])
                mgrs = row['MGRS_TILE']
                state = row['STUSPS']

                out_file = os.path.join(out_dir, f'{fid}_{year}_{state}_{mgrs}_{pt_type}.parquet')
                if os.path.exists(out_file) and not overwrite:
                    continue

                features = row.drop(['YEAR', 'POINT_TYPE', 'MGRS_TILE', 'STUSPS'])

                features_df = pd.DataFrame(features).T
                features_df = features_df.drop(columns=FEATURE_COLS_DROP)

                features_df.to_parquet(out_file)

            except Exception as e:
                print(f"Error processing row {fid} in {f}: {e}")

    if categories:
        out_json = os.path.join(os.path.dirname(out_dir), 'categorical.json')
        categories = {k: [int(i) for i in v] for k, v in categories.items()}
        with open(out_json, 'w') as fp:
            fp.write(json.dumps(categories, indent=4, sort_keys=True))
        print(f'Wrote {out_json}')


if __name__ == '__main__':
    is_authorized()
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    pt_aea = os.path.join(root, 'irrmapper', 'EE_extracts', 'point_shp', 'state_aea')
    pt_wgs = os.path.join(root, 'irrmapper', 'EE_extracts', 'point_shp', 'state_wgs_mgrs')
    mgrs = os.path.join(root, 'boundaries', 'mgrs', 'mgrs_aea.shp')
    bucket = 'gs://wudr'

    states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

    # state_shapefiles = to_geographic(pt_aea, pt_wgs, states=states, mgrs_path=mgrs)

    # for shp in state_shapefiles:
    # push_points_to_asset(pt_wgs, shp, bucket)

    chk = '/data/ssd2/irrmapper/bands/states/bands_past/'
    get_bands(pt_wgs, extract_modern=False, check_dir=chk, diagnose=False)

    # chk = '/data/ssd2/irrmapper/bands/states/bands_modern/'
    # get_bands(pt_wgs, extract_modern=True, check_dir=chk, diagnose=False)

# ========================= EOF ====================================================================
