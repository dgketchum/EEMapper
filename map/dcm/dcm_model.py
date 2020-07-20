import os
from pprint import pprint

from numpy import array
from pandas import read_csv, unique

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from map.dcm.dcm_helper import DCMHelper
from map.dcm.utils import PrettyLogger
DEVICE = torch.device("cuda")


logger = PrettyLogger()
helper = DCMHelper()


def train(net, x_train, x_test, y_train, y_test):

    scaler, x_train, x_test = helper.normalize_without_scaler(x_train, x_test)

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
    num_classes = df['POINT_TYPE'].unique()
    df.drop(columns=['Unnamed: 0.1', 'LAT_GCS', 'Lon_GCS', 'POINT_TYPE', 'YEAR'], inplace=True)
    df[df > 0.0] = df.div(10000.)
    df[df == -99] = 0.0
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    doy_features = [['{}{}'.format(str(y).rjust(3, '0'), x) for x in bands] for y in range(1, 367)]
    doy_data = [df[dv].values for dv in doy_features]
    data = array(doy_data).swapaxes(0, 1)
    x, x_test, y, y_test = train_test_split(data, labels, test_size=0.33,
                                            random_state=None)

    x = helper.input_x(x)
    y = helper.input_y(y)
    x_test = helper.input_x(x_test)
    y_test = helper.input_y(y_test)

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
