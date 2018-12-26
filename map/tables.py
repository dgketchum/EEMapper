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

import os

from geopandas import GeoDataFrame, read_file
from numpy import where, array, sum
from pandas import read_csv, concat, errors, DataFrame

INT_COLS = ['POINT_TYPE', 'YEAR']

KML_JUNK = ['Name', 'descriptio', 'timestamp', 'begin', 'end', 'altitudeMo',
            'tessellate', 'extrude', 'visibility', 'drawOrder', 'icon']


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
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.shp')]
    _files.sort()
    first_year = True
    for year in range(1986, 2016):
        yr_files = [f for f in _files if str(year) in f]
        first_state = True
        for f in yr_files:
            if first_state:
                gdf = read_file(f).drop(columns=KML_JUNK)
                df = DataFrame(gdf).set_index('system_ind')
                if 'mean' in df.columns:
                    print(f)
                else:
                    print('not in', f)
                # df.dropna(subset=['mean'], inplace=True)
                # df.rename(columns={'mean': 'IrrPct_{}'.format(year)}, inplace=True)
                # df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)
                # df['Irr_{}'.format(year)] = where(df['IrrPct_{}'.format(year)].values > 0.5, 1, 0)
                first_state = False
            else:
                gdc = read_file(f).drop(columns=KML_JUNK)
                c = DataFrame(gdc).set_index('system_ind')
                if 'mean' in c.columns:
                    print(f)
                else:
                    print('not in', f)
                # c.dropna(subset=['mean'], inplace=True)
                # c.rename(columns={'mean': 'IrrPct_{}'.format(year)}, inplace=True)
                # c['Irr_{}'.format(year)] = where(c['IrrPct_{}'.format(year)].values > 0.5, 1, 0)
                # df = concat([df, c], sort=False)
                # df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)

        # print(df.shape)
        # if first_year:
        #     master = df
        #     master.rename(columns={'.geo': 'geometry'}, inplace=True)
        #     first_year = False
        # else:
        #     master['IrrPct_{}'.format(year)] = df['IrrPct_{}'.format(year)]
        #     master['Irr_{}'.format(year)] = df['Irr_{}'.format(year)]

    # bool_cols = array([master[x].values for x in master.columns if 'Irr_' in x])
    # bool_sum = sum(bool_cols, axis=0)
    # master['Years_Irrigated'] = bool_sum
    # gpd = GeoDataFrame(master)
    # for i, r in gpd.iterrows():
    #     print(r['geometry'].geom_type)
    # gpd.to_file(out_filename)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # extracts = os.path.join(home, 'IrrigationGIS', 'EE_extracts')
    # rt = os.path.join(extracts, 'to_concatenate')
    # out = os.path.join(extracts, 'concatenated')
    # concatenate_band_extract(rt, out, glob='bands_11DEC')
    extracts = os.path.join(home, 'IrrigationGIS', 'attr_irr', 'shp')
    o = os.path.join(home, 'IrrigationGIS', 'attr_irr', 'shp', 'DRI_agpoly_IrrAttr.shp')
    d = os.path.join(extracts, 'DRI_agpoly')
    concatenate_irrigation_attrs(d, o)

    # csv = os.path.join(extracts, 'concatenated', '')

# ========================= EOF ====================================================================
