import os
import sys
from pprint import pprint
from time import time
from subprocess import call
from copy import deepcopy

from numpy import dot, mean, flatnonzero, ones_like, where, zeros_like
from pandas import read_csv, concat
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold

from geopandas import GeoDataFrame
from shapely.geometry import Point

from map import FEATURE_NAMES
from map.variable_importance import dec4_names

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']
CLASS_NAMES = ['IRR', 'DRYL', 'WETl', 'UNCULT']


def consumer(arr):
    c = [(arr[x, x] / sum(arr[x, :])) for x in range(0, arr.shape[1])]
    return c


def producer(arr):
    c = [(arr[x, x] / sum(arr[:, x])) for x in range(0, arr.shape[0])]
    return c


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


def random_forest(csv, n_estimators=150, out_shape=None):
    print('\n', csv)
    c = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    # c = c[c['POINT_TYPE'] != 1]
    # c = c[c['POINT_TYPE'] != 2]
    # c = c[c['POINT_TYPE'] != 3]

    split = int(c.shape[0] * 0.7)

    df = deepcopy(c.loc[:split, :])
    y = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    x = df.values

    val = deepcopy(c.loc[split:, :])
    y_test = val['POINT_TYPE'].values
    geo = val.apply(lambda x: Point(x['Lon_GCS'], x['LAT_GCS']), axis=1)
    val.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    val.dropna(axis=1, inplace=True)
    x_test = val.values

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                n_jobs=-1,
                                bootstrap=True)

    rf.fit(x, y)
    y_pred = rf.predict(x_test)
    if out_shape:
        val['pred'] = y_pred
        val['label'] = y_test

        ones = ones_like(y_test)
        zeros = zeros_like(y_test)
        val['corr'] = where(y_pred == y_test, ones, zeros)

        gdf = GeoDataFrame(val, geometry=geo, crs="EPSG:4326")
        gdf.to_file(out_shape)
        gdf = gdf[gdf['corr'] == 0]
        incor = os.path.join(os.path.dirname(out_shape),
                             '{}_{}'.format('incor', os.path.basename(out_shape)))
        gdf.to_file(incor)

    cf = confusion_matrix(y_test, y_pred)
    pprint(cf)
    pprint(producer(cf))
    pprint(consumer(cf))
    return


def random_forest_feature_select(csv, n_estimators=100):
    df = read_csv(csv, engine='python').sample(frac=0.1)
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    labels = labels.reshape((labels.shape[0],))
    features = dec4_names()
    precision = []
    for i, c in enumerate(features, start=1):
        cols = [f for f in features[:i]]
        sub_df = df[cols]
        data = sub_df.values
        x, x_test, y, y_test = train_test_split(data, labels, test_size=0.33,
                                                random_state=None)

        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    n_jobs=-1,
                                    bootstrap=False)

        rf.fit(x, y)
        y_pred = rf.predict(x_test)
        cf = confusion_matrix(y_test, y_pred)
        prec = consumer(cf)
        print(i, cols[-1:], prec[0])
        precision.append(prec[0])
    print(precision)


def export_tree(rf, tree_idx, out_file=None):
    tree = rf.estimators_[tree_idx]
    export_graphviz(tree, out_file=out_file,
                    feature_names=FEATURE_NAMES,
                    class_names=CLASS_NAMES,
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    png = out_file.replace('.dot', '.png')
    call(['dot', '-Tpng', out_file, '-o', png, '-Gdpi=600'])
    return None


def find_rf_variable_importance(csv):
    first = True
    master = {}
    df = read_csv(csv, engine='python')
    # df = df[(df['POINT_TYPE'] == 0) | (df['POINT_TYPE'] == 1)]

    labels = list(df['POINT_TYPE'].values)
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    data = df.values
    names = df.columns

    for x in range(10):
        d, _, l, _ = train_test_split(data, labels, train_size=0.67)
        print('model iteration {}'.format(x))
        rf = RandomForestClassifier(n_estimators=150,
                                    n_jobs=-1,
                                    bootstrap=True)

        rf.fit(d, l)
        _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
        imp = sorted(_list, key=lambda x: x[1], reverse=True)
        print([f[0] for f in imp[:10]])

        if first:
            for (k, v) in imp:
                master[k] = v
            first = False
        else:
            for (k, v) in imp:
                master[k] += v

    master = list(master.items())
    master = sorted(master, key=lambda x: x[1], reverse=True)
    return master


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
        pprint(important)
        pprint(rf.score(x_test, y_test))
        y_pred = rf.predict(x_test)
        cf = confusion_matrix(y_test, y_pred)
        pprint(cf)
        producer(cf)
        consumer(cf)

    return important


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
    out_ = os.path.join('/media/research', 'IrrigationGIS', 'EE_extracts', 'concatenated')
    # shapefile = '/media/research/IrrigationGIS/EE_extracts/evaluated_points/eval_18JAN2021.shp'
    extracts = os.path.join(out_, 'bands_3DEC2020_fallow.csv')
    # find_rf_variable_importance(extracts)
    # random_forest(extracts)
    wa = '/media/research/IrrigationGIS/EE_extracts/concatenated/state/MT_10NOV2021.csv'
    find_rf_variable_importance(wa)
# ========================= EOF ====================================================================
