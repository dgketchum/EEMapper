import os
import numpy as np
import tensorflow as tf
try:
    from map.trainer.training_utils import make_test_dataset
except ModuleNotFoundError:
    from trainer.training_utils import make_test_dataset

# dates are generic, dates of each year as below, but data is from many years
# the year of the data is not used in training, just date position
DATES = {0: '19860101',
         1: '19860131',
         2: '19860302',
         3: '19860401',
         4: '19860501',
         5: '19860531',
         6: '19860630',
         7: '19860730',
         8: '19860829',
         9: '19860928',
         10: '19861028',
         11: '19861127',
         12: '19861227'}

# see feature_spec.py for dict of bands, lat , lon, elev, label
CHANNELS = 7
BANDS = 91

structure = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def write_npy(out, recs):
    dataset = make_test_dataset(recs).batch(1)
    count = 0
    obj_ct = np.array([0, 0, 0, 0])
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(4)])
        obj_ct += classes
        features = features.numpy().squeeze()
        a = np.append(features, labels, axis=2)
        np.save(os.path.join(out, '{}.npy'.format(count)), a)
        count += 1
        if count % 100 == 0:
            print(count)
    print(obj_ct)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    np_images = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'data', 'npy')
    tf_recs = os.path.join(home, 'IrrigationGIS', 'tfrecords')
    if not os.path.isdir(np_images):
        np_images = 'gs://ts_data/cmask/npy'
        tf_recs = 'gs://ts_data/cmask/points'
    write_npy(np_images, tf_recs)

# ========================= EOF ====================================================================
