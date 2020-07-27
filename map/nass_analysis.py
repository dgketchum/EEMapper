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
from copy import deepcopy
from pandas import read_csv


STATEFP_MAP = {'AZ': 4,
               'CA': 6,
               'CO': 8,
               'ID': 16,
               'MT': 30,
               'NV': 32,
               'NM': 35,
               'OR': 41,
               'UT': 49,
               'WA': 53,
               'WY': 56}

STATEFP_INV = {v: k for k, v in STATEFP_MAP.items()}


def irr_time_series_totals(irr, nass):

    years = [2007, 2017]

    df = read_csv(irr)
    df.drop(['COUNTYFP', 'COUNTYNS', 'LSAD', 'GEOID'], inplace=True, axis=1)
    totals = df.groupby(['STATEFP']).sum()
    names = [STATEFP_INV[x] for x in list(totals.index)]
    totals['NAME'] = names
    labels = ['noCdlMask_{}'.format(x) for x in years]
    df = totals[labels]

    nass = read_csv(nass, index_col=[0])
    nass.dropna(axis=0, subset=['STATE_ANSI'], inplace=True)
    nass['STATE_ANSI'] = nass['STATE_ANSI'].astype(int)
    nass = nass.loc[nass['STATE_ANSI'].isin(list(df.index))]
    state_nass = deepcopy(nass)
    series_cols = ['VALUE_{}'.format(x) for x in years]
    cols = series_cols + ['STATE_ANSI']
    state_nass = state_nass[cols].groupby(['STATE_ANSI']).sum()
    state_nass = state_nass.loc[list(STATEFP_INV.keys())]
    df[series_cols] = state_nass[series_cols]
    names = [STATEFP_INV[x] for x in list(totals.index)]
    df['NAME'] = names
    df.to_csv('/home/dgketchum/Downloads/nass_irrmapper_2007_2017.csv')


if __name__ == '__main__':
    home = '/media/research'
    county = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')
    nass_merged = os.path.join(county, 'nass_merged.csv')
    irr_tables = os.path.join(county, 'counties_v2', 'noCdlMask_minYr5')

    irrmapper_all = os.path.join(irr_tables, 'irr_merged_ac.csv')
    totals_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
                                 'figures', 'totals_time_series.pdf')
    irr_time_series_totals(irrmapper_all, nass_merged)

# ========================= EOF ====================================================================
