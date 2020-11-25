import os
import sys
from pprint import pprint
from time import time
from subprocess import call

# import tensorflow as tf
from numpy import dot, mean, flatnonzero, unique
from numpy.random import randint
from pandas import read_csv, concat, get_dummies, DataFrame
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from map import FEATURE_NAMES


abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']
CLASS_NAMES = ['IRR', 'DRYL', 'WETl', 'UNCULT']


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


def random_forest(csv, binary=False, n_estimators=100):
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    data = df.values
    if binary:
        labels = labels.reshape((labels.shape[0],))
        labels[labels > 1] = 1
    else:
        labels = labels.reshape((labels.shape[0],))

    x, x_test, y, y_test = train_test_split(data, labels, test_size=0.33,
                                            random_state=None)

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_features=11,
                                max_depth=4,
                                min_samples_split=11,
                                n_jobs=-1,
                                bootstrap=False)

    rf.fit(x, y)
    return rf


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
    out_ = os.path.join(home, 'Downloads')
    extracts = '/media/research/IrrigationGIS/EE_extracts'
    vals = os.path.join(extracts, 'validation_tables', 'validation_12AUG2019.csv')
    bands = os.path.join(extracts, 'concatenated', 'bands_15JUL_v2_kw_USEDINPAPER.csv')
    rf_ = random_forest(bands, binary=False, n_estimators=10)
    for x in range(10):
        out_file = os.path.join(out_, 'tree_{}.dot'.format(x))
        export_tree(rf_, 0, out_file=out_file)
# ========================= EOF ====================================================================
