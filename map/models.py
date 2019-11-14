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
from time import time

import tensorflow as tf
from numpy import dot, mean, flatnonzero
from pandas import read_csv, concat
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']


def consumer(arr):
    c = [(arr[x, x] / sum(arr[x, :])) for x in range(0, arr.shape[1])]
    print('consumer accuracy: {}'.format(c))


def producer(arr):
    c = [(arr[x, x] / sum(arr[:, x])) for x in range(0, arr.shape[0])]
    print('producer accuracy: {}'.format(c))


def pca(csv):
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    data = df.values
    names = df.columns

    x = data
    y = labels.reshape((labels.shape[0],))
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                            random_state=None)

    pca = PCA()
    _ = pca.fit_transform(x)
    x_centered = x - mean(x, axis=0)
    cov_matrix = dot(x_centered.T, x_centered) / len(names)
    eigenvalues = pca.explained_variance_
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
        print(dot(eigenvector.T, dot(cov_matrix, eigenvector)))
        print(eigenvalue)


def random_forest(csv, binary=False):
    df = read_csv(csv, engine='python')
    # df = df.sample(frac=0.5).reset_index(drop=True)
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


def find_rf_variable_importance(csv):
    first = True
    master = {}
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    data = df.values
    names = df.columns

    for x in range(10):
        print('model iteration {}'.format(x))
        rf = RandomForestClassifier(n_estimators=100,
                                    min_samples_split=11,
                                    n_jobs=-1,
                                    bootstrap=False)

        rf.fit(data, labels)
        _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
        imp = sorted(_list, key=lambda x: x[1], reverse=True)

        if first:
            for (k, v) in imp:
                master[k] = v
            first = False
        else:
            for (k, v) in imp:
                master[k] += v

    master = list(master.items())
    master = sorted(master, key=lambda x: x[1], reverse=True)
    pprint(master)


def random_forest_k_fold(csv):
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    data = df.values
    names = df.columns
    labels = labels.reshape((labels.shape[0],))
    kf = KFold(n_splits=2, shuffle=True)

    for train_idx, test_idx in kf.split(data[:-1, :], y=labels[:-1]):
        x, x_test = data[train_idx], data[test_idx]
        y, y_test = labels[train_idx], labels[test_idx]

        rf = RandomForestClassifier(n_estimators=100,
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


def multilayer_perceptron(x, weights, biases):
    out_layer = tf.add(tf.matmul(x, weights), biases)
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def normalize_feature_array(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return data


def get_size(start_path='.'):
    """ Size of data directory in GB.
    :param start_path:
    :return:
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    total_size = total_size * 1e-9
    return total_size


def random_hyperparameter_search(csv):

    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    x = df.values
    y = labels.reshape((labels.shape[0],))
    # x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
    #                                         random_state=None)
    clf = RandomForestClassifier(n_estimators=100)

    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)

    start = time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    start = time()
    grid_search.fit(x, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)


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
    vals = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'validation_tables', 'validation_12AUG2019.csv')
    # get_confusion_matrix(vals, ((0, 185), (1, 38), (2, 8740), (3, 1037)))
    bands = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'concatenated', 'bands_15JUL_v2_kw_USEDINPAPER.csv')
    find_rf_variable_importance(bands)
# ========================= EOF ====================================================================
