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

import matplotlib.pyplot as plt
from numpy import nan
from pandas import read_table, read_csv, DataFrame, Series, concat
from pandas.io.json import json_normalize
from geopandas import GeoDataFrame
from map.tables import to_polygon
import json
from copy import deepcopy

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


def get_nass(csv, out_file):
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
            master = deepcopy(df)
            master['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            master.drop(columns=DROP)
        else:

            master['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']

    master.to_csv(out_file)


def merge_nass_irrmapper(nass, irrmapper, out_name):
    idf = read_csv(irrmapper)
    ndf = read_csv(nass)
    df = DataFrame(columns=['State', 'County_Code', 'County_Name', 'IrrMap_2012_ac',
                            'NASS_2012_ac'])
    gdf = GeoDataFrame(columns=['State', 'County_Code', 'County_Name', 'IrrMap_2012_ac',
                                'NASS_2012_ac', 'geometry'])
    idx = 0
    for i, r in idf.iterrows():
        for j, e in ndf.iterrows():
            if r['STATEFP'] == e['STATE_FIPS_CODE'] and r['COUNTYFP'] == int(e['COUNTY_ANSI']):
                irr_area = r['count'] * r['mean_2012'] * 900.0 / 4046.86

                try:
                    nass_area = int(e['VALUE'].replace(',', ''))
                except ValueError:
                    nass_area = nan

                data = {'State': r['STATEFP'], 'County_Code': r['COUNTYFP'],
                        'County_Name': r['NAME'], 'IrrMap_2012_ac': irr_area,
                        'NASS_2012_ac': nass_area}

                s = Series(name=idx, data=data)
                df.loc[idx] = s
                data['geometry'] = idf.loc[i]['.geo']
                gdf.loc[idx] = data
                idx += 1

    coords = Series(json_normalize(gdf['geometry'].apply(json.loads))['coordinates'].values,
                    index=df.index)
    gdf['geometry'] = coords.apply(to_polygon)
    df.dropna(axis=1)
    df.to_csv(out_name)
    gdf.dropna(axis=1)
    gdf.to_file(out_name.replace('.csv', '.shp'))


def compare_nass_irrmapper(csv):
    df = read_csv(csv)
    fig, ax = plt.subplots(1, 1)
    s = Series(index=df.index)
    s.loc[0], s.loc[df.shape[0]] = 0, 1e6
    s.interpolate(axis=0, inplace=True)
    s.index = s.values
    s.plot(x=s.values, ax=ax, kind='line', loglog=True)
    df.plot(x='NASS_2012_ac', y='IrrMap_2012_ac', kind='scatter',
            xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True)
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tables = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')
    _files = [os.path.join(tables, x) for x in ['qs.census2002.txt',
                                                'qs.census2007.txt',
                                                'qs.census2012.txt']]
    merged = os.path.join(tables, 'nass_merged.csv')
    get_nass(_files, merged)

    # irr = os.path.join(tables, 'county_irrmap.csv')
    # nass = os.path.join(tables, 'nass_2012.csv')
    # o = os.path.join(tables, 'nass_irrrmap_comp.csv')
    # merge_nass_irrmapper(nass, irr, o)

# ========================= EOF ====================================================================
