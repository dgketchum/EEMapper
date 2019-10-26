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
from copy import deepcopy

from geopandas import GeoDataFrame
from numpy import nan
from pandas import read_table, read_csv, DataFrame, Series, concat
from pandas.io.json import json_normalize

from map.tables import to_polygon

DROP = ['SOURCE_DESC', 'SECTOR_DESC', 'GROUP_DESC',
        'COMMODITY_DESC', 'CLASS_DESC', 'PRODN_PRACTICE_DESC',
        'UTIL_PRACTICE_DESC', 'STATISTICCAT_DESC', 'UNIT_DESC',
        'SHORT_DESC', 'DOMAIN_DESC', 'DOMAINCAT_DESC',  'STATE_FIPS_CODE',
        'ASD_CODE', 'ASD_DESC', 'COUNTY_ANSI',
        'REGION_DESC', 'ZIP_5', 'WATERSHED_CODE',
        'WATERSHED_DESC', 'CONGR_DISTRICT_CODE', 'COUNTRY_CODE',
        'COUNTRY_NAME', 'LOCATION_DESC', 'YEAR', 'FREQ_DESC',
        'BEGIN_CODE', 'END_CODE', 'REFERENCE_PERIOD_DESC',
        'WEEK_ENDING', 'LOAD_TIME', 'VALUE', 'AGG_LEVEL_DESC',
        'CV_%', 'STATE_ALPHA', 'STATE_NAME', 'COUNTY_NAME']


TSV = {1987: ('DS0041/35206-0041-Data.tsv', 'ITEM01018', 'FLAG01018'),
       1992: ('DS0042/35206-0042-Data.tsv', 'ITEM010018', 'FLAG010018'),
       1997: ('DS0043/35206-0043-Data.tsv', 'ITEM01019', 'FLAG01019')}


def get_old_nass(_dir, out_file):
    master = None
    first = True
    for k, v in TSV.items():
        print(v)
        value = 'VALUE_{}'.format(k)
        _file, item, flag = v
        csv = os.path.join(_dir, _file)
        df = read_table(csv)
        df.columns = [str(x).upper() for x in df.columns]
        df.index = df['FIPS']
        try:
            df.drop('FIPS', inplace=True)
        except KeyError:
            pass
        df = df[['LEVEL', item, flag]]
        df = df[df['LEVEL'] == 1]
        if k != 1997:
            df = df[df[flag] == 0]
        df.dropna(axis=0, subset=[item], inplace=True, how='any')
        if first:
            first = False
            master = deepcopy(df)
            master[value] = df[item].astype(float)
            master.drop([flag, item, 'LEVEL'], inplace=True, axis=1)
        else:
            master = concat([master, df], axis=1)
            master[value] = df[item].astype(float)
            master.drop([flag, item, 'LEVEL'], inplace=True, axis=1)

    master.to_csv(out_file)


def get_nass(csv, out_file, old_nass=None):
    first = True
    if old_nass:
        old_df = read_csv(old_nass)
        old_df.index = old_df['FIPS']
    for c in csv:
        print(c)
        try:
            df = read_table(c, sep='\t')
            assert len(list(df.columns)) > 2
        except AssertionError:
            df = read_csv(c)
        df.dropna(axis=0, subset=['COUNTY_CODE'], inplace=True, how='any')
        cty_str = df['COUNTY_CODE'].map(lambda x: str(int(x)).zfill(3))
        idx_str = df['STATE_FIPS_CODE'].map(lambda x: str(int(x))) + cty_str
        idx = idx_str.map(int)
        df.index = idx
        df['ST_CNTY_STR'] = df['STATE_ALPHA'] + '_' + df['COUNTY_NAME']
        df = df[(df['SOURCE_DESC'] == 'CENSUS') &
                (df['SECTOR_DESC'] == 'ECONOMICS') &
                (df['GROUP_DESC'] == 'FARMS & LAND & ASSETS') &
                (df['COMMODITY_DESC'] == 'AG LAND') &
                (df['CLASS_DESC'] == 'ALL CLASSES') &
                (df['PRODN_PRACTICE_DESC'] == 'IRRIGATED') &
                (df['UTIL_PRACTICE_DESC'] == 'ALL UTILIZATION PRACTICES') &
                (df['STATISTICCAT_DESC'] == 'AREA') &
                (df['UNIT_DESC'] == 'ACRES') &
                (df['SHORT_DESC'] == 'AG LAND, IRRIGATED - ACRES') &
                (df['DOMAIN_DESC'] == 'TOTAL')]
        df['VALUE'] = df['VALUE'].map(lambda x: nan if 'D' in x else int(x.replace(',', '')))
        if first:
            first = False
            new_nass = deepcopy(df)
            new_nass['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            new_nass.drop(columns=DROP, inplace=True)
        else:
            new_nass['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']

    new_nass.to_csv(out_file.replace('.csv', '_new.csv'))
    df = concat([old_df, new_nass], axis=1)
    df.to_csv(out_file)


def strip_null(row):
    if isinstance(row, str):
        if ',' in row:
            val = float(row.replace(',', ''))
        elif 'D' in row:
            val = nan
        else:
            val = float(row)
    elif isinstance(row, float):
        val = row
    elif isinstance(row, int):
        val = float(row)
    else:
        print(row)

    return val


def merge_nass_irrmapper(nass, irrmapper, out_name):
    years = [1987, 1992, 1997, 2002, 2007, 2012, 2017]
    year_str = [str(x) for x in years]

    idf = read_csv(irrmapper, index_col=[2])
    cols = [x for x in idf.columns if x[-4:] in year_str]
    idf = idf[cols]
    idf.sort_index(axis=1, inplace=True)
    idf.columns = ['IM_{}'.format(x) for x in years]

    ndf = read_csv(nass, index_col=[0])
    ndf.drop(columns=['FIPS'], inplace=True)
    ndf.sort_index(axis=1, inplace=True)
    cols = [x for x in ndf.columns if 'VALUE' in x]
    ndf = ndf[cols]
    cols = ['NASS_{}'.format(y) for y in years]
    ndf.columns = cols

    df = concat([ndf, idf], axis=1)
    df.dropna(axis=0, thresh=8, inplace=True)
    df.to_csv(out_name)


if __name__ == '__main__':
    home = os.path.expanduser('~')

    nass_tables = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')
    irr_tables = os.path.join(nass_tables, 'counties_v2', 'noCdlMask_minYr5')

    old_data = os.path.join(nass_tables, 'old_nass.csv')
    old_data_dir = os.path.join(nass_tables, 'ICPSR_35206')

    # get_old_nass(old_data_dir, out_file=old_data)
    _files = [os.path.join(nass_tables, x) for x in ['qs.census2002.txt',
                                                     'qs.census2007.txt',
                                                     'qs.census2012.txt',
                                                     'qs.census2017.txt']]
    merged = os.path.join(nass_tables, 'nass_merged.csv')
    get_nass(_files, merged, old_nass=old_data)

    # irr = os.path.join(irr_tables, 'irr_merged.csv')
    # nass = os.path.join(nass_tables, 'nass_merged.csv')
    #
    # o = os.path.join(irr_tables, 'nass_irrMap.csv')
    #
    # merge_nass_irrmapper(nass, irr, o)

# ========================= EOF ====================================================================
