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

from pandas import read_csv, concat, errors, merge

INT_COLS = ['POINT_TYPE', 'YEAR']


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


def concatenate_irrigation_attrs(_dir):
    _files = [os.path.join(_dir, x) for x in os.listdir(_dir)]

    first_year = True
    for year in range(1986, 2017):
        yr_files = [f for f in _files if str(year) in f]
        first_state = True
        for f in yr_files:
            if first_state:
                df = read_csv(f, index_col=0)
                df.dropna(subset=['mean'], inplace=True)
                df.rename(columns={'mean': 'IrrPct_{}'.format(year)}, inplace=True)
                first_state = False
            else:
                c = read_csv(f, index_col=0)
                c.dropna(subset=['mean'], inplace=True)
                c.rename(columns={'mean': 'IrrPct_{}'.format(year)}, inplace=True)
                df = concat([df, c], sort=False)
                df.drop_duplicates(subset=['.geo'], keep='first', inplace=True)
        if first_year:
            master = df
            first_year = False
        else:
            master = concat([master, df], sort=False)
        pass


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # extracts = os.path.join(home, 'IrrigationGIS', 'EE_extracts')
    # rt = os.path.join(extracts, 'to_concatenate')
    # out = os.path.join(extracts, 'concatenated')
    # concatenate_band_extract(rt, out, glob='bands_11DEC')
    extracts = os.path.join(home, 'IrrigationGIS', 'attr_irr')
    d = os.path.join(extracts, 'DRI_agpoly')
    concatenate_irrigation_attrs(d)

    # csv = os.path.join(extracts, 'concatenated', '')

# ========================= EOF ====================================================================
