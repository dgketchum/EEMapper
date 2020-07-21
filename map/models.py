# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2 (the "License");
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
# =============================================================================================

import os
import sys
from pprint import pprint

from numpy import int64, float32
from pandas import read_csv, concat
from dask import dataframe as dd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn


abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']


def consumer(arr):
    c = [(arr[x, x] / sum(arr[x, :])) for x in range(0, arr.shape[1])]
    print('consumer accuracy: {}'.format(c))


def producer(arr):
    c = [(arr[x, x] / sum(arr[:, x])) for x in range(0, arr.shape[0])]
    print('producer accuracy: {}'.format(c))


def random_forest(csv, binary=False):
    df = read_csv(csv, engine='python')
    # df[df['POINT_TYPE'] == 3] = 2
    # df = df.sample(frac=0.2).reset_index(drop=True)
    labels = df['POINT_TYPE'].values
    print(df['POINT_TYPE'].value_counts())
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    data = df.values

    print(csv)
    names = df.columns
    print(list(names))
    print(df.shape)
    if binary:
        labels = labels.reshape((labels.shape[0],))
        labels[labels > 1] = 1
    else:
        labels = labels.reshape((labels.shape[0],))

    x, x_test, y, y_test = train_test_split(data, labels, test_size=0.33,
                                            random_state=None)

    rf = RandomForestClassifier(n_estimators=100,
                                min_samples_split=11,
                                n_jobs=-1,
                                bootstrap=False)

    rf.fit(x, y)
    _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
    important = sorted(_list, key=lambda x: x[1], reverse=True)
    pprint(rf.score(x_test, y_test))
    y_pred = rf.predict(x_test)
    cf = confusion_matrix(y_test, y_pred)
    pprint(cf)
    producer(cf)
    consumer(cf)
    return important


def get_confusion_matrix(csv, spec=None):
    df = read_csv(csv, engine='python')

    if spec:
        for c in df.columns:
            if c in INT_COLS:
                df[c] = df[c].astype(int, copy=True)
            else:
                df[c] = df[c].astype(float, copy=True)

        counts = df['POINT_TYPE'].value_counts()
        _min = min(counts.values)
        for i, j in spec:
            if i == 0:
                ndf = df[df['POINT_TYPE'] == i].sample(n=j)
            else:
                ndf = concat([ndf, df[df['POINT_TYPE'] == i].sample(n=j)], sort=False)
        sample_counts = ndf['POINT_TYPE'].value_counts()
        print('original set: {}\n sampled set: {}'.format(counts, sample_counts))
        df = ndf

    y_true, y_pred = df['POINT_TYPE'].values, df['classification'].values

    print('\nclassifcation...')
    cf = confusion_matrix(y_true, y_pred)
    pprint(cf)
    producer(cf)
    consumer(cf)

    print('\nbinary classification ...')
    pt = [1 if x in [1, 2, 3] else 0 for x in df['POINT_TYPE'].values]
    cls = [1 if x in [1, 2, 3] else 0 for x in df['classification'].values]
    cf = confusion_matrix(pt, cls)
    pprint(cf)
    producer(cf)
    consumer(cf)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    bands = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'concatenated', 'sr_series.csv')
# ========================= EOF ====================================================================
