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

import json
import os

from geopandas import GeoDataFrame, read_file
from numpy import where, sum, nan, std, array, min, max, mean
from pandas import read_csv, concat, errors, Series, DataFrame
from pandas.io.json import json_normalize
from shapely import geometry

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


def concatenate_band_extract(root, out_dir, glob='None', sample=None):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
    l.sort()
    first = True
    for csv in l:
        try:
            if first:
                df = read_csv(csv)
                first = False
            else:
                c = read_csv(csv)
                df = concat([df, c], sort=False)
                print(c.shape, csv)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df.drop(columns=['system:index', '.geo'], inplace=True)

    if sample:
        _len = int(df.shape[0]/1e3 * sample)
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

    print('size: {}'.format(df.shape))
    df.to_csv(out_file, index=False)


def concatenate_irrigation_attrs(_dir, out_filename):
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.csv')]
    _files.sort()
    first_year = True
    for year in range(1986, 2017):
        yr_files = [f for f in _files if str(year) in f]
        first_state = True
        for f in yr_files:
            if first_state:
                df = read_csv(f, index_col=0)
                df.dropna(subset=['mean'], inplace=True)
                df.rename(columns={'mean': 'IPct_{}'.format(year)}, inplace=True)
                df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)
                df['Irr_{}'.format(year)] = where(df['IPct_{}'.format(year)].values > 0.5, 1, 0)
                first_state = False
            else:
                c = read_csv(f, index_col=0)
                c.dropna(subset=['mean'], inplace=True)
                c.rename(columns={'mean': 'IPct_{}'.format(year)}, inplace=True)
                c['Irr_{}'.format(year)] = where(c['IPct_{}'.format(year)].values > 0.5, 1, 0)
                df = concat([df, c], sort=False)
                df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)

        print(year, df.shape)
        if first_year:
            master = df
            first_year = False
        else:
            master['IPct_{}'.format(year)] = df['IPct_{}'.format(year)]
            master['Irr_{}'.format(year)] = df['Irr_{}'.format(year)]

    bool_cols = array([master[x].values for x in master.columns if 'Irr_' in x])
    bool_sum = sum(bool_cols, axis=0)
    master['IYears'] = bool_sum
    master.dropna(subset=['.geo'], inplace=True)
    coords = Series(json_normalize(master['.geo'].apply(json.loads))['coordinates'].values,
                    index=master.index)
    master['geometry'] = coords.apply(to_polygon)
    master.dropna(subset=['geometry'], inplace=True)
    gpd = GeoDataFrame(master.drop(['.geo'], axis=1),
                       crs={'init': 'epsg:4326'})
    gpd.to_file(out_filename)


def concatenate_attrs(_dir, out_filename, template_geometry, huc_level=8):

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
            df = read_csv(_mean)
            names = df['Name']
            template_gpd = read_file(template_geometry)

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
            df.drop(columns=['.geo', 'mean'], inplace=True)
            _count_csv = [f for f in yr_files if 'count' in f][0]
            count_arr = read_csv(_count_csv, index_col=0)['count'].values
            df['TotalPix'.format(year)] = count_arr
            df['Ct_{}'.format(year)] = mean_arr * count_arr
            first = False

        else:
            mean_arr = read_csv(_mean, index_col=0)['mean'].values
            df['Ct_{}'.format(year)] = mean_arr * count_arr

    df.drop(columns=['Name'], inplace=True)
    df.drop(columns=KML_DROP, inplace=True)
    df['Name'] = names
    df['geometry'] = df_geo
    df.to_csv(out_filename)
    gpd = GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=df_geo)
    gpd.to_file(out_filename.replace(os.path.basename(out_filename),
                                     'irrigation_timeseries_huc{}.shp'.format(huc_level)))


def add_stat_attrs(shp, out_shp):

    template_gpd = read_file(shp)
    df = DataFrame(template_gpd)
    df.drop(columns=['geometry'], inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
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

    gpd = GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=template_gpd['geometry'])
    gpd.to_file(out_shp)


def to_polygon(j):
    if not isinstance(j, list):
        return nan
    try:
        return geometry.Polygon(j[0])
    except ValueError:
        return nan
    except TypeError:
        return nan
    except AssertionError:
        return nan


if __name__ == '__main__':
    home = os.path.expanduser('~')
    extracts = os.path.join(home, 'IrrigationGIS', 'attr_irr', 'csv', 'MT_agpoly')
    shape = os.path.join(home, 'IrrigationGIS', 'attr_irr', 'shp', 'MT_irr_attrs.shp')
    concatenate_irrigation_attrs(extracts, out_filename=shape)

# ========================= EOF ====================================================================
