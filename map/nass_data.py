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

from pandas import read_table, read_csv, DataFrame

API_KEY = '93135C42-1E18-3C3C-81FA-5E064F59B423'


def get_nass(csv):
    df = read_table(csv, sep='\t')
    df.dropna(axis=0, subset=['COUNTY_CODE'], inplace=True, how='any')
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
    df.to_csv(csv.replace('qs.sample.txt', 'nass_2012.csv'))


def compare_county_stats(nass, irrmapper):
    idf = read_csv(irrmapper)
    ndf = read_csv(nass)
    df = DataFrame()
    for i, r in idf.iterrows():
        for j, e in ndf.iterrows():
            if r['STATEFP'] == e['STATE_FIPS_CODE'] and r['COUNTYFP'] == int(e['COUNTY_ANSI']):
                df.update({'State': r['STATEFP'], 'County_Code': r['COUNTYFP'],
                           })


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tables = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')
    # compare_county_stats(os.path.join(tables, 'counties.csv'))
    # get_nass(os.path.join(tables, 'qs.census2012.txt'))
    n = os.path.join(tables, 'nass_2012.csv')
    i = os.path.join(tables, 'county_irrmap.csv')
    compare_county_stats(n, i)
# ========================= EOF ====================================================================
