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
        'WEEK_ENDING', 'LOAD_TIME', 'VALUE']


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
    for c in csv:
        print(c)
        try:
            df = read_table(c, sep='\t')
            assert len(list(df.columns)) > 2
        except AssertionError:
            df = read_csv(c)
        df.dropna(axis=0, subset=['COUNTY_CODE'], inplace=True, how='any')
        df.index = df['STATE_ALPHA'] + '_' + df['COUNTY_NAME']
        cols = [x for x in df.columns.values]
        cols[-1] = 'cv_pct'
        df.columns = cols
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
        if first:
            first = False
            new_nass = deepcopy(df)
            new_nass['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            new_nass.drop(columns=DROP)
        else:
            new_nass['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            new_nass.drop(columns=DROP)
    if old_nass:
        old_df = read_csv(old_nass)

    new_nass.to_csv(out_file)


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
    idf = read_csv(irrmapper)
    ndf = read_csv(nass)
    cols = ['VALUE_2002', 'VALUE_2007', 'VALUE_2012']
    ndf[cols] = ndf[cols].applymap(lambda x: strip_null(x))
    df = DataFrame(columns=['State', 'State_Code', 'Cty_Name', 'Cty_Code', 'IM2002_ac', 'NASS_2002_ac',
                            'IM2007_ac', 'NASS_2007_ac', 'IM2012_ac', 'NASS_2012_ac', 'IM2017_ac', 'NASS_2017_ac'])

    gdf = GeoDataFrame(columns=['State', 'State_Code', 'Cty_Name', 'Cty_Code', 'IM2002_ac', 'NASS_2002_ac',
                                'IM2007_ac', 'NASS_2007_ac', 'IM2012_ac', 'NASS_2012_ac', 'IM2017_ac', 'NASS_2017_ac',
                                'geometry'], crs={'init': 'epsg:4326'})
    idx = 0
    for i, r in idf.iterrows():
        for j, e in ndf.iterrows():
            if r['STATEFP'] == e['STATE_FIPS_CODE'] and r['COUNTYFP'] == int(e['COUNTY_ANSI']):
                irr_area = (r['noCdlMask_2002'] / 4046.86,
                            r['noCdlMask_2007'] / 4046.86,
                            r['noCdlMask_2012'] / 4046.86,
                            r['noCdlMask_2017'] / 4046.86)

                nass_area = (e['VALUE_2002'], e['VALUE_2007'], e['VALUE_2012'], e['VALUE_2017'])

                data = {'State': e['STATE_ALPHA'], 'State_Code': r['STATEFP'], 'Cty_Code': r['COUNTYFP'],
                        'Cty_Name': r['NAME'],
                        'IM2002_ac': irr_area[0], 'NASS_2002_ac': nass_area[0],
                        'IM2007_ac': irr_area[1], 'NASS_2007_ac': nass_area[1],
                        'IM2012_ac': irr_area[2], 'NASS_2012_ac': nass_area[2],
                        'IM2017_ac': irr_area[3], 'NASS_2017_ac': nass_area[3]}

                s = Series(name=idx, data=data)
                df.loc[idx] = s
                data['geometry'] = idf.loc[i]['.geo']
                gdf.loc[idx] = data
                idx += 1

    df.to_csv(out_name)

    gdf['dif_2002'] = (gdf['NASS_2002_ac'] - gdf['IM2002_ac']) / gdf['NASS_2002_ac']
    gdf['dif_2007'] = (gdf['NASS_2007_ac'] - gdf['IM2007_ac']) / gdf['NASS_2007_ac']
    gdf['dif_2012'] = (gdf['NASS_2012_ac'] - gdf['IM2012_ac']) / gdf['NASS_2012_ac']
    gdf['dif_2012'] = (gdf['NASS_2017_ac'] - gdf['IM2017_ac']) / gdf['NASS_2017_ac']

    coords = Series(json_normalize(gdf['geometry'].apply(json.loads))['coordinates'].values,
                    index=df.index)
    gdf['geometry'] = coords.apply(to_polygon)
    gdf.dropna(subset=['geometry'], inplace=True)
    out_shp = out_name.replace('.csv', '.shp')
    gdf.to_file(out_shp)


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
