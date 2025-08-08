import os
import sys
import time
import json

import ee

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from subprocess import check_call
from datetime import datetime

import fiona
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from map.call_ee import is_authorized, stack_bands, BOUNDARIES

sys.setrecursionlimit(5000)

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = '/home/dgketchum/miniconda3/envs/met/bin/earthengine'
GS = '/home/dgketchum/google-cloud-sdk/bin/gsutil'

OGR = '/usr/bin/ogr2ogr'

AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

FEATURE_COLS_DROP = ['system:index', '.geo']


def to_geographic(in_dir, out_dir, states, mgrs_path, n_samples=None):
    df_list = []
    for state in states:
        in_shp = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp') and state in x]
        dates = []
        for s in in_shp:
            try:
                d = datetime.strptime(s.split('_')[-1].split('.')[0], '%d%b%Y')
                dates.append(d)
            except ValueError:
                continue
        latest = in_shp[dates.index(max(dates))]
        mgrs = gpd.read_file(mgrs_path)

        points = gpd.read_file(latest)
        points = points.sjoin(mgrs, how="inner")
        points = points.to_crs(epsg=4326)
        points.drop(columns=['index_right'], inplace=True)
        points['STUSPS'] = state

        if n_samples:
            points = points.groupby('POINT_TYPE', group_keys=False).apply(lambda x: x.sample(n=n_samples))
            points.index = list(range(points.shape[0]))

        df_list.append(points)

    df = pd.concat(df_list, ignore_index=True)
    df['FID'] = ['%s_%s' % (row['STUSPS'], str(i).zfill(6)) for i, row in df.iterrows()]

    out_shp = os.path.join(out_dir, 'master_training_points.shp')
    df.to_file(out_shp)
    print(out_shp)
    return out_shp


def push_points_to_asset(_dir, shapefile, bucket):
    shp_name = os.path.basename(shapefile).replace('.shp', '')
    local_files = [os.path.join(_dir, '{}.{}'.format(shp_name, ext)) for ext in
                   ['shp', 'prj', 'shx', 'dbf']]
    bucket = os.path.join(bucket, 'state_points')
    bucket_files = [os.path.join(bucket, '{}.{}'.format(shp_name, ext)) for ext in
                    ['shp', 'prj', 'shx', 'dbf']]
    for lf, bf in zip(local_files, bucket_files):
        cmd = [GS, 'cp', lf, bf]
        check_call(cmd)

    asset_id = os.path.basename(bucket_files[0]).split('.')[0]
    ee_dst = 'users/dgketchum/points/state_redev/{}'.format(asset_id)
    cmd = [EE, 'upload', 'table', '-f', '--asset_id={}'.format(ee_dst), bucket_files[0]]
    check_call(cmd)
    print(asset_id, bucket_files[0])


def get_bands(shp, southern_states, extract_modern=False, modern_ndvi_dir=None, check_dir=None,
              diagnose=False):
    points_df = gpd.read_file(shp)

    if extract_modern:
        if not modern_ndvi_dir:
            raise ValueError("modern_ndvi_dir must be provided when extract_modern is True")

        file_prefix = 'irrmapper_redev/bands_modern'

        print("Extracting modern years from processed NDVI files...")
        modern_files = [f for f in os.listdir(modern_ndvi_dir) if f.endswith('.parquet')]
        modern_info = []
        for f in modern_files:
            parts = os.path.basename(f).split('_')
            fid = '_'.join(parts[:2])
            year = int(parts[2])
            modern_info.append({'FID': fid, 'YEAR': year})

        modern_years_df = pd.DataFrame(modern_info)
        points_df = points_df.drop(columns=['YEAR']).merge(modern_years_df, on='FID')

    else:
        file_prefix = 'irrmapper_redev/bands'

    mgrs_ee_tiles = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    mgrs_tiles = points_df['MGRS_TILE'].unique()

    for tile in mgrs_tiles:
        tile_df = points_df[points_df['MGRS_TILE'] == tile]
        states = tile_df['STUSPS'].unique()

        for state in states:
            is_southern = state in southern_states
            state_df = tile_df[tile_df['STUSPS'] == state]
            years = sorted(list(state_df['YEAR'].unique()))

            for year in years:
                year_df = state_df[state_df['YEAR'] == year]

                if year_df.empty:
                    continue

                desc = f'bands_{tile}_{state}_{year}'

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

                selectors = ['FID', 'POINT_TYPE', 'YEAR', 'MGRS_TILE', 'STUSPS']

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
            [categories[c].add(v)  for  c in categories.keys() for v in df[c].unique()]

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
    root = '/media/research/IrrigationGIS/irrmapper'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/irrmapper'

    pt_wgs = os.path.join(root, 'EE_extracts/point_shp', 'state_wgs_mgrs')
    master_shp = os.path.join(pt_wgs, 'master_training_points.shp')

    southern = ['AZ', 'CA', 'NM', 'TX', 'OK']
    #
    # check_d = '/data/ssd2/irrmapper/bands/past_training/'
    # get_bands(master_shp, southern, extract_modern=False, check_dir=check_d, diagnose=False)
    #
    # check_d = '/data/ssd2/irrmapper/bands/modern_update/'
    # modern_ndvi_dir_ = '/data/ssd2/irrmapper/ndvi/modern_processed/'
    # get_bands(master_shp, southern, extract_modern=True, modern_ndvi_dir=modern_ndvi_dir_, check_dir=check_d)

    in_dir_past = '/data/ssd2/irrmapper/bands/past_bands/'
    out_dir_past = '/data/ssd2/irrmapper/bands/past_processed/'
    process_bands_to_parquet(in_dir_past, out_dir_past,
                             category_counters=['nlcd', 'cdl', 'crop5c', 'cropland', 'gsw'],
                             overwrite=True)

    # in_dir_modern = '/data/ssd2/irrmapper/bands/modern_bands/'
    # out_dir_modern = '/data/ssd2/irrmapper/bands/modern_processed/'
    # process_bands_to_parquet(in_dir_modern, out_dir_modern)

# ========================= EOF ====================================================================
