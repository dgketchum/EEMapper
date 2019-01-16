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
from numpy import where, array, sum, nan
from pandas import read_csv, concat, errors, Series
from pandas.io.json import json_normalize
from shapely import geometry
from shapely.geometry.polygon import Polygon

INT_COLS = ['POINT_TYPE', 'YEAR']

KML_JUNK = ['Name', 'descriptio', 'timestamp', 'begin', 'end', 'altitudeMo',
            'tessellate', 'extrude', 'visibility', 'drawOrder', 'icon']

KML_TIME_SERIES_KEEP = ['areaacres', 'areasqkm', 'huc6', 'loaddate', 'Mean_1986',
                        'shape_area', 'shape_leng', 'states', 'Count_1986', 'geometry']


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


def concatenate_sum_attrs(_dir, out_filename, template_geometry=None):

    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.csv')]
    _files.sort()
    first = True

    for year in range(1986, 1987):
        print(year)
        df_geo = None
        yr_files = [f for f in _files if str(year) in f]
        _mean = [f for f in yr_files if 'mean' in f][0]
        _count = [f for f in yr_files if 'count' in f][0]

        if first:
            df = read_csv(_mean, index_col=0)
            names = df['Name']

            if template_geometry:
                template_gpd = read_file(template_geometry)
                str_geo = template_gpd['geometry']


            else:
                str_geo = df['.geo']
                coords = Series(json_normalize(str_geo.apply(json.loads))['coordinates'].values,
                                index=df.index)
                df_geo = coords.apply(to_polygon)

            df.drop(columns=['.geo'], inplace=True)
            df.rename(columns={'mean': 'Mean_{}'.format(year)}, inplace=True)
            count_arr = read_csv(_count, index_col=0)['count'].values
            df['Count_{}'.format(year)] = count_arr
            first = False

        else:
            mean_arr = read_csv(_mean, index_col=0)['mean'].values
            count_arr = read_csv(_count, index_col=0)['count'].values
            df['Count_{}'.format(year)] = count_arr
            df['Mean_{}'.format(year)] = mean_arr

    df.drop(columns=['Name'], inplace=True)
    [df.drop(columns=[x], inplace=True) for x in df.columns if x not in KML_TIME_SERIES_KEEP]
    df['Name'] = names
    # df.to_csv(out_filename)
    for i, r in str_geo.iteritems():
        if not isinstance(r, Polygon):
            print(type(r), df.iloc[i])
    gpd = GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=template_gpd['geometry'])
    gpd.to_file(out_filename.replace(os.path.basename(out_filename), '_irrigation_timeseries_huc6.shp'))


def to_polygon(j):
    if not isinstance(j, list):
        return nan
    try:
        return geometry.Polygon(j[0])
    except ValueError:
        return nan
    except TypeError:
        return nan


if __name__ == '__main__':
    home = os.path.expanduser('~')
    extracts = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_huc6')
    out_table = os.path.join(home, 'IrrigationGIS', 'time_series', 'tables', 'concatenated_huc6.csv')
    template = os.path.join(home, 'IrrigationGIS', 'hydrography', 'huc6_semiarid_clip.shp')
    concatenate_sum_attrs(extracts, out_table, template_geometry=template)
    # csv = os.path.join(extracts, 'concatenated', '')

# ========================= EOF ====================================================================
