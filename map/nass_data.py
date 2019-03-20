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
from pandas import read_table, read_csv, DataFrame, Series

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


def merge_nass_irrmapper(nass, irrmapper):
    idf = read_csv(irrmapper)
    ndf = read_csv(nass)
    df = DataFrame(columns=['State', 'County_Code', 'County_Name', 'IrrMap_2012_ac', 'NASS_2012_ac'])
    idx = 0
    for i, r in idf.iterrows():
        for j, e in ndf.iterrows():
            if r['STATEFP'] == e['STATE_FIPS_CODE'] and r['COUNTYFP'] == int(e['COUNTY_ANSI']):
                irr_area = r['count'] * r['mean_2012'] * 900.0 / 4046.86

                try:
                    nass_area = int(e['VALUE'].replace(',', ''))
                except ValueError:
                    nass_area = nan

                s = Series(name=idx, data={'State': r['STATEFP'], 'County_Code': r['COUNTYFP'],
                                           'County_Name': r['NAME'], 'IrrMap_2012_ac': irr_area,
                                           'NASS_2012_ac': nass_area})
                df.loc[idx] = s
                idx += 1
    df.dropna(axis=1)
    df.to_csv('/home/dgketchum/IrrigationGIS/time_series/exports_county/nass_irrrmap_comp_2012.csv')


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
    # compare_county_stats(os.path.join(tables, 'counties.csv'))
    # get_nass(os.path.join(tables, 'qs.census2012.txt'))
    # n = os.path.join(tables, 'nass_2012.csv')
    # i = os.path.join(tables, 'county_irrmap.csv')
    # merge_nass_irrmapper(n, i)
    comp = os.path.join(tables, 'nass_irrrmap_comp_2012.csv')
    compare_nass_irrmapper(comp)
# ========================= EOF ====================================================================
