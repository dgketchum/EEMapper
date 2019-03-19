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
from numpy import unique, dot, mean
from numpy.random import randint
import tensorflow as tf
from pandas import get_dummies, read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)


def mlp(csv):
    start = datetime.now()
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    data = df.values

    x = normalize_feature_array(data)
    y = get_dummies(labels.reshape((labels.shape[0],))).values
    N = len(unique(labels))
    n = data.data.shape[1]
    print('')
    print('train on {}'.format(data.shape))

    nodes = 500
    eta = 0.01
    epochs = 10000
    seed = 128
    batch_size = 1000

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                            random_state=None)

    X = tf.placeholder("float", [None, n])
    Y = tf.placeholder("float", [None, N])

    weights = {
        'hidden': tf.Variable(tf.random_normal([n, nodes], seed=seed), name='Wh'),
        'output': tf.Variable(tf.random_normal([nodes, N], seed=seed), name='Wo')}
    biases = {
        'hidden': tf.Variable(tf.random_normal([nodes], seed=seed), name='Bh'),
        'output': tf.Variable(tf.random_normal([N], seed=seed), name='Bo')}

    y_pred = tf.add(tf.matmul(multilayer_perceptron(X, weights['hidden'], biases['hidden']),
                              weights['output']), biases['output'])

    loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(epochs):

            offset = randint(0, y.shape[0] - batch_size - 1)

            batch_data = x[offset:(offset + batch_size), :]
            batch_labels = y[offset:(offset + batch_size), :]

            feed_dict = {X: batch_data, Y: batch_labels}

            _, loss = sess.run([optimizer, loss_op],
                               feed_dict=feed_dict)

            if step % 1000 == 0:
                pred = tf.nn.softmax(y_pred)
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('Test accuracy: {}, loss {}'.format(accuracy.eval({X: x_test, Y: y_test}), loss))

    print('training time: {} seconds'.format((datetime.now() - start).seconds))
    return None


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


def find_rf_variable_importance(csv):
    first = True
    master = {}
    for x in range(10):
        imp = random_forest(csv)
        if first:
            for (k, v) in imp:
                master[k] = v
            first = False
        else:
            for (k, v) in imp:
                master[k] += v

    pprint(master)


def random_forest(csv):
    df = read_csv(csv, engine='python')
    labels = df['POINT_TYPE'].values
    df.drop(columns=['YEAR', 'POINT_TYPE'], inplace=True)
    data = df.values
    names = df.columns

    x = data
    y = labels.reshape((labels.shape[0],))

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                            random_state=None)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x, y)
    _list = [(f, v) for f, v in zip(names, rf.feature_importances_)]
    important = sorted(_list, key=lambda x: x[1], reverse=True)
    print(rf.score(x_test, y_test))
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


if __name__ == '__main__':
    home = os.path.expanduser('~')
    csv_loaction = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'concatenated')
    csv = os.path.join(csv_loaction, 'bands_40k_14NOV.csv')
    pca(csv)
    # random_forest(csv)
    # mlp(csv)
# ========================= EOF ====================================================================
