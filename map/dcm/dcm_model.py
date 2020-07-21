import os
from pprint import pprint

from numpy import array, append, where, count_nonzero, zeros_like
from pandas import read_csv

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from map.dcm.dcm_helper import DCMHelper
from map.dcm.utils import PrettyLogger
DEVICE = torch.device("cuda")


logger = PrettyLogger()
helper = DCMHelper()
scaler_ = StandardScaler()


def train(net, x_train, x_test, y_train, y_test):

    train_dataloader = helper.make_data_loader(x_train, y_train, shuffle=True)
    test_dataloader = helper.make_data_loader(x_test, y_test, shuffle=False)

    loss_train_list, acc_train_list, attn_train_list = [], [], []
    loss_test_list, acc_test_list, attn_test_list = [], [], []
    helper.train_model(
        net, train_dataloader, test_dataloader, DEVICE, logger,
        loss_train_list, acc_train_list, attn_train_list,
        loss_test_list, acc_test_list, attn_test_list,
    )

    test_dataloader = helper.make_data_loader(x_test, y_test, shuffle=False)

    y_train_soft_pred, y_train_hard_pred, attn_train = helper.predict(
        net, helper.make_data_loader(x_train, y_train, shuffle=False), DEVICE)

    y_test_soft_pred, y_test_hard_pred, attn_test = helper.predict(
        net, test_dataloader, DEVICE)

    acc_train = accuracy_score(y_train, y_train_hard_pred)
    acc_test = accuracy_score(y_test, y_test_hard_pred)
    logger.info("train acc:", acc_train, "test acc:", acc_test)

    return None


def run_model(csv):
    df = read_csv(csv)
    labels = df['POINT_TYPE'].values
    coords = df[['LAT_GCS', 'Lon_GCS']].values
    years = df['YEAR'].values
    df.drop(columns=['Unnamed: 0.1', 'LAT_GCS', 'Lon_GCS', 'POINT_TYPE', 'YEAR'], inplace=True)
    df[df > 0.0] = df.div(10000.)
    df[df < 0.0] = 0.0
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    doy_features = [['{}{}'.format(str(y).rjust(3, '0'), x) for x in bands] for y in range(1, 367)]

    doy_data = [df[dv].values for dv in doy_features[135:267]]
    doy_coords = [where(x[:, :2] > 0.0, coords, zeros_like(coords)) for x in doy_data]
    data = [append(x, c, axis=1) for x, c in zip(doy_data, doy_coords)]
    data = array(data).swapaxes(0, 1)

    info_ = append(count_nonzero(data[:, :, 0], axis=1).reshape((df.shape[0], 1)),
                   years.reshape((df.shape[0], 1)), axis=1)
    # info_ = info_[info_[:, 0].argsort()]
    # x, x_test, y, y_test = train_test_split(data, labels, test_size=0.33,
    #                                         random_state=None)

    x = helper.input_x(x)
    y = helper.input_y(y)
    x_test = helper.input_x(x_test)
    y_test = helper.input_y(y_test)

    scaler, x, x_test = helper.normalize_without_scaler(x, x_test)

    net = helper.build_model()
    helper.init_parameters(net)
    net = nn.DataParallel(net)
    net.to(DEVICE)

    train(net, x, x_test, y, y_test)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    bands = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'concatenated', 'sr_series.csv')
    run_model(bands)
# ========================= EOF ====================================================================
