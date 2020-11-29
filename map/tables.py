import json
import os
from datetime import datetime

from geopandas import GeoDataFrame, read_file
from numpy import where, sum, nan, std, array, min, max, mean, int16
from pandas import read_csv, concat, errors, Series, merge, DataFrame
from pandas import to_datetime
from pandas.io.json import json_normalize
from shapely.geometry import Polygon
from map.variable_importance import original_names

INT_COLS = ['POINT_TYPE', 'YEAR']

KML_DROP = ['system:index', 'altitudeMo', 'descriptio',
            'extrude', 'gnis_id', 'icon', 'loaddate',
            'metasource', 'name_1', 'shape_area', 'shape_leng', 'sourcedata',
            'sourcefeat', 'sourceorig', 'system_ind', 'tessellate',
            'tnmid', 'visibility', ]

DUPLICATE_HUC8_NAMES = ['Beaver', 'Big Sandy', 'Blackfoot', 'Carrizo', 'Cedar', 'Clear', 'Colorado Headwaters', 'Crow',
                        'Frenchman', 'Horse', 'Jordan', 'Lower Beaver', 'Lower White', 'Medicine', 'Muddy', 'Palo Duro',
                        'Pawnee', 'Redwater', 'Rock', 'Salt', 'San Francisco', 'Santa Maria', 'Silver', 'Smith',
                        'Stillwater', 'Teton', 'Upper Bear', 'Upper White', 'White', 'Willow']

COLS = ['SCENE_ID',
        'PRODUCT_ID',
        'SPACECRAFT_ID',
        'SENSOR_ID',
        'DATE_ACQUIRED',
        'COLLECTION_NUMBER',
        'COLLECTION_CATEGORY',
        'SENSING_TIME',
        'DATA_TYPE',
        'WRS_PATH',
        'WRS_ROW',
        'CLOUD_COVER',
        'NORTH_LAT',
        'SOUTH_LAT',
        'WEST_LON',
        'EAST_LON',
        'TOTAL_SIZE',
        'BASE_URL']

DROP_COUNTY = ['system:index', 'AFFGEOID', 'COUNTYFP', 'COUNTYNS', 'GEOID', 'LSAD', 'STATEFP', '.geo']

SELECT = ['cdl',
          'nlcd',
          'nd_max_cy',
          'nd_3',
          'B5_3',
          'slope',
          'B7_3',
          'LAT_GCS',
          'nd_max_m1',
          'B4_3',
          'Lon_GCS',
          'nd_max_m2',
          'B5_4',
          'nd_4',
          'elevation',
          'B7_4',
          'B6_3',
          'B2_3',
          'prec_5',
          'B3_3',
          'prec_3',
          'B6_4',
          'tmin_5_1',
          'B4_4',
          'prec_4',
          'tmin_4_1',
          'prec_7',
          'prec_6',
          'tpi_250',
          'B5_2',
          'B3_4',
          'B2_4',
          'tpi_150',
          'prec_8',
          'tmin_6_1',
          'nd_2',
          'gsw',
          'prec_2',
          'tmin_7_1',
          'tpi_1250']


def concatenate_county_data(folder, out_file, glob='counties', acres=False):
    df = None
    base_names = [x for x in os.listdir(folder)]
    _files = [os.path.join(folder, x) for x in base_names if x.startswith(glob) and not 'total_area' in x]
    totals_file = [os.path.join(folder, x) for x in base_names if 'total_area' in x][0]
    first = True

    for csv in _files:

        print(csv)
        if first:
            df = read_csv(totals_file).sort_values('COUNTYNS')
            cty_str = df['COUNTYFP'].map(lambda x: str(int(x)).zfill(3))
            idx_str = df['STATEFP'].map(lambda x: str(int(x))) + cty_str
            idx = idx_str.map(int)
            df['FIPS'] = idx
            df.index = idx
            if acres:
                df['total_area'] = df['sum'] / 4046.86
            else:
                df['total_area'] = df['sum']
            df.drop(columns=['sum'], inplace=True)
            first = False

        prefix, year = os.path.basename(csv).split('_')[1], os.path.basename(csv).split('_')[4]

        c = read_csv(csv).sort_values('COUNTYNS')
        name = '{}_{}'.format(prefix, year)
        cty_str = c['COUNTYFP'].map(lambda x: str(int(x)).zfill(3))
        idx_str = c['STATEFP'].map(lambda x: str(int(x))) + cty_str
        idx = idx_str.map(int)
        c.index = idx
        if acres:
            c[name] = c['sum'] / 4046.86
        else:
            c[name] = c['sum']

        df = concat([df, c[name]], axis=1)
        print(c.shape, csv)

    # print('size: {}'.format(df.shape))
    # df = df.reindex(sorted(df.columns), axis=1)
    df.drop(columns=DROP_COUNTY, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.to_csv(out_file, index=False)
    print('saved {}'.format(out_file))


def concatenate_band_extract(root, out_dir, glob='None', sample=None, select=None):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
    # TODO: remove this
    l += [os.path.join(root, 'bands_27NOV2020_1_2013.csv')]
    l.sort()
    first = True
    for csv in l:
        try:
            if first:
                df = read_csv(csv)
                cols = list(df.columns)
                df.columns = cols
                print(df.shape, csv)
                first = False
            else:
                c = read_csv(csv)
                cols = list(c.columns)
                c.columns = cols
                df = concat([df, c], sort=False)
                print(c.shape, csv)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df.drop(columns=['system:index', '.geo'], inplace=True)
    df.drop(df[(df['gsw'] == 1) & (df['POINT_TYPE'] == 0)].index, inplace=True)

    if sample:
        _len = int(df.shape[0] / 1e3 * sample)
        out_file = os.path.join(out_dir, '{}_{}.csv'.format(glob, _len))
    else:
        out_file = os.path.join(out_dir, '{}.csv'.format(glob))

    for c in df.columns:
        if c in INT_COLS:
            df[c] = df[c].astype(int, copy=True)
        else:
            df[c] = df[c].astype(float, copy=True)
    if sample:
        df = df.sample(frac=sample).reset_index(drop=True)

    if select:
        print(df['POINT_TYPE'].value_counts())
        df = df[select + ['POINT_TYPE', 'YEAR']]
        out_file = os.path.join(out_dir, '{}_sub.csv'.format(glob))
        sub_df = df[df['POINT_TYPE'] == 0].sample(n=55000)
        for x in range(1, 4):
            sub = df[df['POINT_TYPE'] == x].sample(n=35000)
            sub_df = concat([sub_df, sub], sort=False)
        df = sub_df

    print('size: {}'.format(df.shape))
    print(df['POINT_TYPE'].value_counts())
    print('file: {}'.format(out_file))
    df.to_csv(out_file, index=False)


def rm_dupe_geometry():
    in_dir = '/home/dgketchum/IrrigationGIS/east_training/dupes'
    out_dir = '/home/dgketchum/IrrigationGIS/east_training/'
    shapes = [(x, os.path.join(in_dir, x)) for x in os.listdir(in_dir) if '.shp' in x]
    for bn, fn in shapes:
        df = read_file(fn)
        print(df.shape)
        df = df[['SOURCE', 'geometry']]
        df.drop_duplicates(subset=['geometry'], keep='first', inplace=True)
        print(df.shape)
        out = os.path.join(out_dir, bn)
        df.to_file(out)


def concatenate_irrigation_attrs(_dir, out_filename, glob, find_boolean=False, template_geometry=None):
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if glob in x]
    _files.sort()
    first_year = True
    for year in range(2011, 2021):
        yr_files = [f for f in _files if str(year) in f]
        first_state = True
        for f in yr_files:
            if first_state:
                df = read_csv(f, index_col=0)
                df.dropna(subset=['sum'], inplace=True)
                df.rename(columns={'sum': 'ISQMT_{}'.format(year)}, inplace=True)
                df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)
                df['IPCT_{}'.format(year)] = df['ISQMT_{}'.format(year)] / df['AREA_SQMT'.format(year)] * 100.
                first_state = False
            else:
                c = read_csv(f, index_col=0)
                c.dropna(subset=['sum'], inplace=True)
                c.rename(columns={'sum': 'ISQMT_{}'.format(year)}, inplace=True)
                c['IPCT_{}'.format(year)] = c['ISQMT_{}'.format(year)] / c['AREA_SQMT'.format(year)] * 100.
                df = concat([df, c], sort=False)
                df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)

        print(year, df.shape)
        if first_year:
            master = df
            first_year = False
        else:
            master['IPCT_{}'.format(year)] = df['IPCT_{}'.format(year)]
            master['ISQMT_{}'.format(year)] = df['ISQMT_{}'.format(year)]

    if find_boolean:
        bool_cols = array([master[x].values for x in master.columns if 'Irr_' in x])
        bool_sum = sum(bool_cols, axis=0)
        master['IYears'] = bool_sum

    if template_geometry:
        t_gdf = GeoDataFrame.from_file(template_geometry).to_crs('epsg:4326')
        geo = t_gdf['geometry']
        master.drop(['.geo'], axis=1, inplace=True)
    else:
        master.dropna(subset=['.geo'], inplace=True)
        coords = Series(json_normalize(master['.geo'].apply(json.loads))['coordinates'].values,
                        index=master.index)
        geo = coords.apply(to_polygon)
        master.dropna(subset=['geometry'], inplace=True)
        master.drop(['.geo'], axis=1, inplace=True)

    master.to_csv(out_filename.replace('.shp', '.csv'))
    # gpd = GeoDataFrame(master, crs='epsg:4326', geometry=geo)
    # gpd.to_file(out_filename)


def concatenate_attrs_huc(_dir, out_csv_filename, out_shp_filename, template_geometry):
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.csv')]
    _files.sort()
    first = True
    df_geo = []
    template_names = []
    count_arr = None
    names = None

    for year in range(1986, 2017):
        print(year)
        yr_files = [f for f in _files if str(year) in f]
        _mean = [f for f in yr_files if 'mean' in f][0]

        if first:
            df = read_csv(_mean, index_col=['huc8']).sort_values('huc8', axis=0)
            names = df['Name']
            units = df.index
            template_gpd = read_file(template_geometry).sort_values('huc8', axis=0)
            for i, r in template_gpd.iterrows():

                if r['Name'] in DUPLICATE_HUC8_NAMES:
                    df_geo.append(r['geometry'])
                    template_names.append('{}_{}'.format(r['Name'], str(r['states']).replace(',', '_')))
                elif r['Name'] in names.values and r['Name'] not in template_names:
                    df_geo.append(r['geometry'])
                    template_names.append(r['Name'])
                else:
                    print('{} is in the list'.format(r['Name']))

            mean_arr = df['mean']
            df.drop(columns=KML_DROP, inplace=True)
            df.drop(columns=['.geo', 'mean'], inplace=True)
            _count_csv = [f for f in yr_files if 'count' in f][0]
            count_arr = read_csv(_count_csv, index_col=0)['count'].values
            df['TotalPix'.format(year)] = count_arr
            df['Ct_{}'.format(year)] = mean_arr * count_arr
            first = False

        else:
            mean_arr = read_csv(_mean, index_col=['huc8']).sort_values('huc8', axis=0)['mean'].values
            df['Ct_{}'.format(year)] = mean_arr * count_arr

    year_cts = [x for x in df.columns if 'Ct_' in x]
    cts = df.drop(columns=[x for x in df.columns if x not in year_cts])

    arr = cts.values
    max_pct = (max(arr, axis=1) / df['TotalPix'].values).reshape(arr.shape[0], 1)
    df['max_pct'] = max_pct

    min_pct = (min(arr, axis=1) / df['TotalPix'].values).reshape(arr.shape[0], 1)
    df['min_pct'] = min_pct

    mean_pct = (mean(arr, axis=1) / df['TotalPix'].values).reshape(arr.shape[0], 1)
    df['mean_pct'] = mean_pct

    diff = (max_pct - min_pct) / mean_pct
    df['diff_pct'] = diff

    std_ = std(arr, axis=1).reshape(arr.shape[0], 1)
    df['std_dev'] = std_

    df.drop(columns=['Name'], inplace=True)
    df['Name'] = names
    df['geometry'] = df_geo
    df['huc8'] = units
    df.to_csv(out_csv_filename)
    gpd = GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=df_geo)
    gpd.to_file(out_shp_filename)


def to_polygon(j):
    if not isinstance(j, list):
        return nan
    try:
        return Polygon(j[0])
    except ValueError:
        return nan
    except TypeError:
        return nan
    except AssertionError:
        return nan


def count_landsat_scenes(index, shp):
    pr_list = [str(x) for x in read_file(shp)['PR'].values]
    first = True
    l8, l7, l5 = 0, 0, 0
    with open(index) as f:
        for line in f:
            if first:
                # skip header line
                first = False
            else:
                l = line.split(',')
                sat = l[2]
                pr = l[9].zfill(2) + l[10].zfill(3)
                dt = to_datetime(l[7]).to_pydatetime()
                doy = int(dt.strftime('%j'))
                start, end = datetime(1986, 1, 1), datetime(2016, 12, 31)
                if pr in pr_list and start < dt < end and 59 < doy < 365:
                    if sat == 'LANDSAT_8':
                        l8 += 1
                    elif sat == 'LANDSAT_7':
                        l7 += 1
                    elif sat == 'LANDSAT_5':
                        l5 += 1
        print('{} l8, {} l7, {} l5'.format(l8, l7, l5))


def concatenate_validation(_dir, out_file, glob='validation'):
    _list = [os.path.join(_dir, x) for x in os.listdir(_dir) if glob in x]
    _list.sort()
    first = True
    for csv in _list:
        try:
            if first:
                df = read_csv(csv)
                first = False
            else:
                c = read_csv(csv)
                df = concat([df, c], sort=False)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass
    # df = df.sample(n=32000)
    df.drop(columns=['system:index', '.geo'], inplace=True)
    df.to_csv(out_file)


def concatenate_attrs_county(_dir, out_csv_filename, out_shp_filename, template_geometry):
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.csv') and 'total' not in x]
    total = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.csv') and 'total' in x][0]
    _files.sort()
    first = True
    df_geo = []
    template_names = []

    for year in range(1986, 2019):
        print(year)
        _file = [f for f in _files if str(year) in f][0]

        if first:
            df = read_csv(_file, index_col=['GEOID']).sort_values('GEOID', axis=0)
            template_gpd = read_file(template_geometry).sort_values('GEOID', axis=0)
            for i, r in template_gpd.iterrows():
                df_geo.append(r['geometry'])
                template_names.append(r['GEOID'])

            mean_arr = df['sum'].values
            df['Ct_{}'.format(year)] = mean_arr
            df.drop(columns=['.geo', 'sum'], inplace=True)
            count_arr = read_csv(total)['sum'].values
            df['total_area'.format(year)] = count_arr
            first = False

        else:
            mean_arr = read_csv(_file, index_col=['GEOID']).sort_values('GEOID', axis=0)['sum'].values
            df['Ct_{}'.format(year)] = mean_arr

    year_cts = [x for x in df.columns if 'Ct_' in x]
    cts = df.drop(columns=[x for x in df.columns if x not in year_cts])

    arr = cts.values
    max_pct = (max(arr, axis=1) / df['total_area'].values).reshape(arr.shape[0], 1)
    df['max_pct'] = max_pct

    min_pct = (min(arr, axis=1) / df['total_area'].values).reshape(arr.shape[0], 1)
    df['min_pct'] = min_pct

    mean_pct = (mean(arr, axis=1) / df['total_area'].values).reshape(arr.shape[0], 1)
    df['mean_pct'] = mean_pct

    diff = (max_pct - min_pct) / mean_pct
    df['diff_pct'] = diff

    early_col = ['Ct_{}'.format(x) for x in range(1986, 1991)]
    late_col = ['Ct_{}'.format(x) for x in range(2014, 2019)]
    df['early_mean'] = df[early_col].mean(axis=1) / df['total_area']
    df['late_mean'] = df[late_col].mean(axis=1) / df['total_area']
    df['delta'] = (df['late_mean'] - df['early_mean']) / df[year_cts].mean(axis=1)

    std_ = std(arr, axis=1).reshape(arr.shape[0], 1)
    df['std_dev'] = std_

    # df.drop(columns=['Name'], inplace=True)
    # df['Name'] = names
    df['geometry'] = df_geo
    df.to_csv(out_csv_filename)
    gpd = GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=df_geo)
    gpd.to_file(out_shp_filename)


def get_project_totals(csv, out_file):
    df = read_csv(csv)
    df.drop(['COUNTYFP', 'COUNTYNS', 'LSAD', 'GEOID'], inplace=True, axis=1)
    df = df.groupby(['STATEFP']).sum()
    df.to_csv(out_file.replace('.csv', '_state.csv'))
    s = df.sum()
    s.to_csv(out_file.replace('.csv', '_totals.csv'))
    print('project totals, acreas: {}'.format(s))


def join_comparison_to_shapefile(csv, shp, out_shape):
    df = read_csv(csv, engine='python')
    nass_col = [x for x in list(df.columns) if 'NASS' in x]
    irr_col = [x for x in list(df.columns) if 'IM' in x]

    df['nass_mean'] = (mean(df[nass_col], axis=1))
    df['irr_mean'] = (mean(df[irr_col], axis=1))
    df['n_diff'] = ((df['irr_mean'] - df['nass_mean']) / (df['irr_mean'] + df['nass_mean']))

    gdf = read_file(shp)
    df['GEOID'] = [str(x).zfill(5) for x in df['STCT']]
    gdf = merge(df, gdf, left_on='GEOID', right_on='GEOID', how='left')
    out = GeoDataFrame(gdf, geometry='geometry', crs={'init': 'epsg:4326'})
    out.to_file(out_shape)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    data_dir = '/media/research'
    d = os.path.join(data_dir, 'IrrigationGIS', 'EE_extracts', 'to_concatenate')
    glob = 'bands_27NOV2020'
    o = os.path.join(data_dir, 'IrrigationGIS', 'EE_extracts', 'concatenated')
    concatenate_band_extract(d, o, glob, select=SELECT)
# ========================= EOF ====================================================================
