import os
import sys
import time

import ee

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from subprocess import check_call
from datetime import datetime

import fiona
import geopandas as gpd
import pandas as pd
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


def get_bands(shp, southern_states, extract_modern=False, modern_ndvi_dir=None, check_dir=None):
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

                roi_ = 'users/dgketchum/boundaries/{}'.format(state)
                region = ee.FeatureCollection(roi_)

                bands = stack_bands(year, region, southern=False)
                selectors = ['FID', 'POINT_TYPE', 'YEAR', 'MGRS_TILE', 'STUSPS']

                plot_sample_regions = bands.sampleRegions(
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


if __name__ == '__main__':
    is_authorized()
    root = '/media/research/IrrigationGIS/irrmapper'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/irrmapper'

    pt_wgs = os.path.join(root, 'EE_extracts/point_shp', 'state_wgs_mgrs')
    master_shp = os.path.join(pt_wgs, 'master_training_points.shp')

    southern = ['AZ', 'CA', 'NM', 'TX', 'OK']

    check_d = '/data/ssd2/irrmapper/bands/past_training/'
    get_bands(master_shp, southern, extract_modern=False, check_dir=check_d)

    check_d = '/data/ssd2/irrmapper/bands/modern_update/'
    modern_ndvi_dir_ = '/data/ssd2/irrmapper/ndvi/modern_processed/'
    get_bands(master_shp, southern, extract_modern=True, modern_ndvi_dir=modern_ndvi_dir_, check_dir=check_d)

# ========================= EOF ====================================================================
