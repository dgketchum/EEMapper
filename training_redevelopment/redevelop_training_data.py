import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pprint import pprint

import ee

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from subprocess import check_call
import random

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from map.call_ee import is_authorized, stack_bands

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


def get_bands(shp_dir, extract_modern=False, check_dir=None, diagnose=False, select_states=None):
    """"""
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

        if select_states and state not in select_states:
            continue

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

                if check_dir:
                    if os.path.exists(os.path.join(check_dir, f'{desc}.csv')):
                        print(f"Skipping {desc}, already exists.")
                        continue

                feature_coll = ee.FeatureCollection(year_df.__geo_interface__)

                roi = mgrs_ee_tiles.filterMetadata('MGRS_TILE', 'equals', tile)
                region = ee.FeatureCollection(roi)

                try:
                    stack = stack_bands(year, region, southern=False)
                except ee.ee_exception.EEException as exc:
                    print(f'{desc} error: {exc}')
                    continue

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

                if extract_modern:
                    desc_prepend = 'modern'
                else:
                    desc_prepend = 'past'

                task = ee.batch.Export.table.toCloudStorage(
                    plot_sample_regions,
                    description=f'{desc_prepend}_{desc}',
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


def _process_band_file(params):
    f, out_dir, category_counters, overwrite = params
    if not f.endswith('.csv'):
        return None, None

    try:
        df = pd.read_csv(f)
        df.set_index('FID', inplace=True)
    except pd.errors.EmptyDataError:
        os.remove(f)
        return None, None
    except Exception as e:
        print(f"Error reading {f}: {e}")
        return None, None

    if category_counters:
        uniques = {c: set(df[c].unique()) for c in category_counters}
        counts = {c: df[c].value_counts() for c in category_counters}
    else:
        uniques = None
        counts = None

    for fid, row in df.iterrows():
        try:
            year = int(row['YEAR'])
            new_year = int(row['NEW_YEAR'])
            pt_type = int(row['POINT_TYPE'])
            mgrs = row['MGRS_TILE']
            state = row['STUSPS']

            out_file = os.path.join(out_dir, f'{fid}_{year}_{new_year}_{state}_{mgrs}_{pt_type}.parquet')
            if os.path.exists(out_file) and not overwrite:
                continue

            features = row.drop(['YEAR', 'NEW_YEAR', 'POINT_TYPE', 'MGRS_TILE', 'STUSPS'])

            features_df = pd.DataFrame(features).T
            features_df = features_df.drop(columns=FEATURE_COLS_DROP)

            features_df.to_parquet(out_file)

        except Exception as e:
            print(f"Error processing row {fid} in {f}: {e}")

    return uniques, counts


def process_bands_to_parquet(in_dir, out_dir, category_counters=None, categorical_json=None,
                             overwrite=False, num_workers=1):
    """"""
    file_list = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]

    if category_counters and categorical_json is None:
        raise ValueError

    if category_counters:
        categories = {c: set() for c in category_counters}
        category_counts = {c: pd.Series(dtype=int) for c in category_counters}
    else:
        categories = None
        category_counts = None

    params = [(f, out_dir, category_counters, overwrite) for f in file_list]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(_process_band_file, params), total=len(file_list)))

    if categories:
        for res in results:
            if res[0] and res[1]:
                uniques, counts = res
                for c, values in uniques.items():
                    categories[c].update(values)
                for c, series in counts.items():
                    category_counts[c] = category_counts[c].add(series, fill_value=0)

        datestamp = datetime.now().strftime("%Y%m%d")
        out_json = os.path.join(os.path.dirname(categorical_json), f'categorical_{datestamp}.json')
        categories = {k: [int(i) for i in v if pd.notna(i)] for k, v in categories.items()}
        with open(out_json, 'w') as fp:
            fp.write(json.dumps(categories, indent=4, sort_keys=True))
        print(f'Wrote {out_json}')

        print("\n--- Categorical Metadata Summary ---")
        for category, values in categories.items():
            print(f"  - {category}: {len(values)} unique values")
        print("------------------------------------")

        print("\n--- Categorical Counts Summary ---")
        for category, counts_series in category_counts.items():
            print(f"\nCounts for {category}:")
            pprint(counts_series.astype(int).to_dict())
        print("----------------------------------")


if __name__ == '__main__':
    is_authorized()
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    pt_aea = os.path.join(root, 'irrmapper', 'EE_extracts', 'point_shp', 'state_aea')
    pt_wgs = os.path.join(root, 'irrmapper', 'EE_extracts', 'point_shp', 'state_wgs_mgrs')
    mgrs = os.path.join(root, 'boundaries', 'mgrs', 'mgrs_aea.shp')
    bucket = 'gs://wudr'

    # states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
    east_states = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

    # state_shapefiles = to_geographic(pt_aea, pt_wgs, states=east_stats, mgrs_path=mgrs)

    # for shp in state_shapefiles:
        # push_points_to_asset(pt_wgs, shp, bucket)

    past_extract = '/data/ssd2/irrmapper/states/bands/bands_past/'
    # get_bands(pt_wgs, extract_modern=False, check_dir=past_extract, diagnose=False, select_states=east_states)

    modrern_extract = '/data/ssd2/irrmapper/states/bands/bands_modern/'
    # get_bands(pt_wgs, extract_modern=True, check_dir=modrern_extract, diagnose=False, select_states=east_states)

    past_bands_processed = '/data/ssd2/irrmapper/states/bands/past_processed/'
    categorical_json_ = '/data/ssd2/irrmapper/states/combined_classifier/'
    process_bands_to_parquet(past_extract, past_bands_processed,
                             category_counters=['nlcd', 'cdl', 'crop5c', 'cropland', 'gsw'],
                             categorical_json = categorical_json_,
                             num_workers=24,
                             overwrite=False)

    modern_bands_processed = '/data/ssd2/irrmapper/states/bands/modern_processed/'
    process_bands_to_parquet(modrern_extract, modern_bands_processed, category_counters=None,
                             num_workers=24, overwrite=False)

# ========================= EOF ====================================================================
