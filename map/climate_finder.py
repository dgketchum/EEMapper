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


import io
import os

import requests
from pandas import read_csv

CAG_URL = 'https://www.ncdc.noaa.gov/cag/county/time-series/{}-{}-pcp-{}-{}-1989-2019.csv'


def get_climate_counties(counties, no_months=5, wy_end=10, out_file=None):
    co = read_csv(counties, header=0)
    co['MEAN_SUMMER_PPT'] = ''
    co['MEAN_ANNUAL_PPT'] = ''
    co['DROUGHTS'] = ''
    for i, r in co.iterrows():
        st, co_fip, co_name = r['STATE_ABV'], str(r['CNTY_FIPS']).zfill(3), r['NAME']
        try:
            url = CAG_URL.format(st, co_fip, 12, wy_end)
            req = requests.get(url).content
            raw_df = read_csv(io.StringIO(req.decode('utf-8')), skiprows=3)
            r['MEAN_ANNUAL_PPT'] = raw_df['Value'].mean()

            url = CAG_URL.format(st, co_fip, no_months, wy_end)
            req = requests.get(url).content
            raw_df = read_csv(io.StringIO(req.decode('utf-8')), skiprows=3)

            r['MEAN_SUMMER_PPT'] = raw_df['Value'].mean()
            low_ppt = raw_df.sort_values(by=['Value']).head(n=5)
            r['DROUGHTS'] = [int(str(x)[:4]) for x in list(low_ppt['Date'])]
            co.iloc[i] = r
            print(st, co_name)
        except Exception as e:
            print(e, st, co_name)
    co.to_csv(out_file)
    return None


def get_county_variability(out_file):
    df = read_csv(out_file)
    pass


if __name__ == '__main__':
    home = os.path.expanduser('~')
    county_data = os.path.join(home, 'IrrigationGIS', 'climate_data')
    county_csv = os.path.join(county_data, 'conus_counties.csv')
    get_county_droughts(county_csv)
# ========================= EOF ====================================================================
