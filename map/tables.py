import json
import os
from datetime import datetime

from geopandas import GeoDataFrame, read_file
from numpy import where, nan, std, min, max, mean, ones_like, rint
from pandas import read_csv, concat, errors, Series, merge
from pandas import to_datetime
from pandas.io.json import json_normalize
from shapely.geometry import Polygon
from map.variable_importance import dec_2020_variables

INT_COLS = ['POINT_TYPE', 'YEAR']

KML_DROP = ['system:index', 'altitudeMo', 'descriptio',
            'extrude', 'gnis_id', 'icon', 'loaddate',
            'metasource', 'name_1', 'shape_area', 'shape_leng', 'sourcedata',
            'sourcefeat', 'sourceorig', 'system_ind', 'tessellate',
            'tnmid', 'visibility', ]

DUPLICATE_HUC8_NAMES = \
    ['Beaver', 'Big Sandy', 'Blackfoot', 'Carrizo', 'Cedar', 'Clear', 'Colorado Headwaters', 'Crow',
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

SELECT = dec_2020_variables()


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


def concatenate_band_extract(root, out_dir, glob='None', sample=None, test_correlations=False,
                             select=None, binary=False, fallow=False, nd_only=False, southern=False):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
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

    df = df.drop(columns=['system:index', '.geo'])

    if nd_only:
        drop_cols = []
        for c in list(df.columns):
            if 'evi' in c:
                drop_cols.append(c)
            if 'gi' in c:
                drop_cols.append(c)
            if 'nw' in c:
                drop_cols.append(c)
        df = df.drop(columns=drop_cols)

    print(df['POINT_TYPE'].value_counts())

    points = df['POINT_TYPE'].values
    nines = ones_like(points) * 9

    if 'nd_max_gs' in df.columns:
        points = where((df.POINT_TYPE == 0) & (df.nd_max_gs < 0.68), nines, points)
        # points = where((df.POINT_TYPE == 1) & (df.nd_max_gs > 0.68), nines, points)
    else:
        points = where((df.POINT_TYPE == 0) & (df.nd_max_cy < 0.68), nines, points)
        # points = where((df.POINT_TYPE == 1) & (df.nd_max_cy > 0.68), nines, points)

    # previous year's data is not needed in areas without dryland agriculture
    if southern:
        cols = list(df.columns)
        cols_ = []
        for c in cols:
            if '_m1' in c or '_m2' in c:
                continue
            cols_.append(c)
        df = df[cols_]

    df['POINT_TYPE'] = points
    print(df['POINT_TYPE'].value_counts())
    try:
        points = where((df['POINT_TYPE'] == 4) & (df['nd_max_gs'] > 0.6), nines, points)
    except KeyError:
        points = where((df['POINT_TYPE'] == 4) & (df['nd_max_cy'] > 0.6), nines, points)

    df['POINT_TYPE'] = points
    df = df[df['POINT_TYPE'] != 9]

    df['POINT_TYPE'][df['POINT_TYPE'] == 4] = 1

    print(df['POINT_TYPE'].value_counts())

    if test_correlations:
        l = []
        cols = [c for c in df.columns if c not in ['crop5c', 'nlcd', 'cropland', 'POINT_TYPE', 'YEAR']]
        for c in cols:
            for cc in cols:
                if cc == c:
                    continue
                corr = df[c].corr(df[cc])
                l.append((c, cc, corr))
        l = sorted(l, key=lambda x: x[2], reverse=True)

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

        df = df[SELECT + ['POINT_TYPE', 'YEAR']]

        out_file = os.path.join(out_dir, '{}_{}.csv'.format(glob, len(SELECT)))
        sub_df = df[df['POINT_TYPE'] == 0]
        shape = sub_df.shape[0]
        target = int(shape / 3.)
        target_f = int(shape / 6.)
        for i, x in zip([1, 2, 3], [target, target, target]):
            if target_f == 0:
                sub_df = df
                break
            try:
                sub = df[df['POINT_TYPE'] == i].sample(n=x)
                sub_df = concat([sub_df, sub], sort=False)
            except ValueError:
                print('not enough {} class to sample {}'.format(i, x))
        df = sub_df

    print('size: {}'.format(df.shape))
    print('file: {}'.format(out_file))
    if binary:
        df['POINT_TYPE'][df['POINT_TYPE'] > 0] = 1
        out_file = out_file.replace('.csv', '_binary.csv')
    print(df['POINT_TYPE'].value_counts())
    df.to_csv(out_file, index=False)


def balance_band_extract(csv_in, csv_out, binary=False):
    df = read_csv(csv_in)
    counts = df['POINT_TYPE'].value_counts()
    print(counts)
    sub_df = df.loc[0:3, :]
    target = counts[0]
    sub_target = int(rint(target / 3))
    for i, x in zip([0, 1, 2, 3], [target, sub_target, sub_target, sub_target]):
        try:
            sub = df[df['POINT_TYPE'] == i].sample(n=x)
            sub_df = concat([sub_df, sub])
        except ValueError:
            print('not enough {} class to sample {}'.format(i, x))
    df = sub_df
    if binary:
        df['POINT_TYPE'][df['POINT_TYPE'] > 0] = 1
    print(df['POINT_TYPE'].value_counts())
    df.to_csv(csv_out, index=False)


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


def concatenate_irrigation_attrs(_dir, out_filename, glob, template_geometry=None):
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if glob in x]
    _files.sort()
    first = True
    for f in _files:
        year = int(os.path.basename(f).split('.')[0][-4:])
        if first:
            df = read_csv(f, index_col=0)
            df.dropna(subset=['sum'], inplace=True)
            df.rename(columns={'sum': 'irr_{}'.format(year)}, inplace=True)
            df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)
            first = False
        else:
            c = read_csv(f, index_col=0)
            df['irr_{}'.format(year)] = c['sum']

    if template_geometry:
        t_gdf = GeoDataFrame.from_file(template_geometry).to_crs('epsg:4326')
        geo = t_gdf['geometry']
        df.drop(['.geo'], axis=1, inplace=True)
    else:
        df.dropna(subset=['.geo'], inplace=True)
        coords = Series(json_normalize(df['.geo'].apply(json.loads))['coordinates'].values,
                        index=df.index)
        geo = coords.apply(to_polygon)
        df.drop(['.geo'], axis=1, inplace=True)

    df.to_csv(out_filename.replace('.shp', '.csv'))
    gpd = GeoDataFrame(df, crs='epsg:4326', geometry=geo)
    gpd.to_file(out_filename)


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


def join_tables(one, two, out_file):
    one = read_csv(one)
    two = read_csv(two)
    df = concat([one, two], ignore_index=True)
    df.to_csv(out_file)


def join_shp_csv(in_shp, csv_dir, out_shp, join_on='id', glob='.csv', drop=None):
    gdf = read_file(in_shp)
    gdf.index = [int(i) for i in gdf[join_on]]
    if drop:
        gdf.drop(columns=drop, inplace=True)

    csv_l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if x.endswith(glob)]
    first = True
    for csv in csv_l:
        y = int(csv.split('.')[0][-4:])
        try:
            if first:
                df = read_csv(csv, index_col=join_on)
                df = df.rename(columns={'mean': 'irr_{}'.format(y)})
                print(df.shape, csv)
                first = False
            else:
                c = read_csv(csv, index_col=join_on)
                c = c.rename(columns={'mean': 'irr_{}'.format(y)})
                df = concat([df, c['irr_{}'.format(y)]], axis=1)
                print(c.shape, csv)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    geo = [gdf.loc[i].geometry for i in df.index]
    df = GeoDataFrame(df, crs=gdf.crs, geometry=geo)
    df.to_file(out_shp)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    data_dir = '/media/research'
    d = os.path.join(data_dir, 'IrrigationGIS', 'EE_extracts', 'rebuild_co')
    # glob = 'bands_CO'
    # o = os.path.join(data_dir, 'IrrigationGIS', 'EE_extracts', 'rebuild_co')
    # concatenate_band_extract(d, o, glob, select=True, binary=False, fallow=False)
    one_ = os.path.join(d, 'bands_3DEC2020_50.csv')
    two_ = os.path.join(d, 'bands_CO_50.csv')
    out_ = os.path.join(d, 'bands_4DEC2020_mod_CO.csv')
    # join_tables(one_, two_, out_file=out_)

    dalby = '/media/research/IrrigationGIS/Montana/dalby'
    spatial = os.path.join(dalby, 'Dalby_WRS_mapped_by_DORFLU2019-20221209T162926Z-001/Dalby_WRS_mapped_by_DORFLU2019')
    s = os.path.join(spatial, 'WRS_1946_71_mappedby_DORFLU2019_merge_1a.shp')
    o = os.path.join(spatial, 'WRS_1946_71_mappedby_DORFLU2019_merge_1a_irr.shp')
    c_source = os.path.join(dalby, 'wrs_flu_ee_extracts/wrs_flood_not_mapped')
    join_shp_csv(s, c_source, o, join_on='OBJECTID')

# ========================= EOF ====================================================================
