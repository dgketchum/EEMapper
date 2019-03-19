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

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np


def time_series_normalized(csv):
    df = read_csv(csv).sort_values('huc8', axis=0).drop(columns=['geometry'])
    # df.dropna(axis=0, how='any', inplace=True)
    index = df['huc8']
    drop = [x for x in df.columns if not x.startswith('Ct_')]
    df.drop(columns=drop, inplace=True)
    df.index = index
    df = df.div(df.mean(axis=1), axis=0)
    dft = df.transpose()
    dft.index = [int(x.replace('Ct_', '')) for x in df.columns.values]
    dft = dft.reindex(sorted(dft.columns), axis=1)
    labels = [x for x in dft.columns]
    range = dft.index.values
    vals = [dft[x].values for x in labels]
    dft.plot()
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tables = os.path.join(home, 'IrrigationGIS', 'time_series')
    huc_8 = os.path.join(tables, 'tables', 'concatenated_huc8.csv')
    time_series_normalized(huc_8)
# ========================= EOF ====================================================================
