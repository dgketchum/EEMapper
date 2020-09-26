import os
import numpy as np
import tempfile
import shutil
from google.cloud import storage

try:
    from map.trainer.training_utils import make_test_dataset
    from map.data.bucket import get_bucket_contents
except ModuleNotFoundError:
    from trainer.training_utils import make_test_dataset
    from data.bucket import get_bucket_contents
    print('used alternate import')

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


def write_npy_gcs(recs, bucket=None, bucket_dst=None):
    storage_client = storage.Client()
    try:
        bucket_content = get_bucket_contents(bucket)[bucket_dst]
        count = sorted([int(x[0].split('.')[0]) for x in bucket_content])[-1] + 1
    except:
        count = 0
    dataset = make_test_dataset(recs).batch(1)
    obj_ct = np.array([0, 0, 0, 0])
    tmpdirname = tempfile.mkdtemp()
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(4)])
        obj_ct += classes
        features = features.numpy().squeeze()
        a = np.append(features, labels, axis=2)
        tmp_name = os.path.join(tmpdirname, '{}.npy'.format(count))

        try:
            np.save(tmp_name, a)
        except OSError as e:
            print('{} exceeded memory, flushing'.format(e))
            shutil.rmtree(tmpdirname)
            tmpdirname = tempfile.mkdtemp()

        bucket = storage_client.get_bucket(bucket)
        blob = bucket.blob(os.path.join(bucket_dst, '{}.npy'.format(count)))
        blob.upload_from_filename(tmp_name)
        count += 1
        if count % 100 == 0:
            print(count)
        print(obj_ct)


def write_npy_local(out, recs):
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
    to_cloud = True
    np_images = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'data', 'npy')
    tf_recs = os.path.join(home, 'IrrigationGIS', 'tfrecords', 'test')
    out_bucket = 'ts_data'
    bucket_dir = 'cmask/npy/train'
    # tf_recs = 'gs://ts_data/cmask/points'
    write_npy_gcs(tf_recs, bucket=out_bucket, bucket_dst=bucket_dir)

# ========================= EOF ====================================================================
