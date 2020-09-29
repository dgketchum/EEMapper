import os
import numpy as np
import tempfile
import shutil
import tarfile
from google.cloud import storage

try:
    from map.trainer.training_utils import make_test_dataset
    from map.data.bucket import get_bucket_contents
except ModuleNotFoundError:
    from trainer.training_utils import make_test_dataset
    from data.bucket import get_bucket_contents

    print('used alternate import')


def write_and_push_tar():

def write_npy_gcs(recs, bucket=None, bucket_dst=None):
    storage_client = storage.Client()

    def push_tar(t_dir, bckt, items):
        si, ei = os.path.basename(items[0].split('.')[0]), os.path.basename(items[-1].split('.')[0])
        tar_filename = '{}_{}_{}.tar'.format(os.path.basename(recs), si, ei)
        tar_archive = os.path.join(t_dir, tar_filename)
        with tarfile.open(tar_archive, 'w') as tar:
            print(count, si, ei, )
            for i in items:
                tar.add(i)
        bucket = storage_client.get_bucket(bckt)
        blob_name = os.path.join(bucket_dst, tar_filename)
        blob = bucket.blob(blob_name)
        print('push {}'.format(blob_name))
        blob.upload_from_filename(tar_archive)
        shutil.rmtree(t_dir)

    try:
        bucket_content = get_bucket_contents(bucket)
        sub_content = bucket_content[bucket_dst]
        count = sorted([int(x[0].split('.')[0]) for x in sub_content])[-1] + 1
    except:
        count = 0

    dataset = make_test_dataset(recs).batch(1)
    obj_ct = np.array([0, 0, 0, 0])
    tmpdirname = tempfile.mkdtemp()
    items = []
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(4)])
        obj_ct += classes
        features = features.numpy().squeeze()
        a = np.append(features, labels, axis=2)
        tmp_name = os.path.join(tmpdirname, '{}.npy'.format(count))
        np.save(tmp_name, a)
        items.append(tmp_name)
        count += 1

        if len(items) == 20:
            push_tar(tmpdirname, bucket, items)
            tmpdirname = tempfile.mkdtemp()
            items = []

    push_tar(tmpdirname, bucket, items)
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
    # np_images = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'data', 'npy')
    # tf_recs = os.path.join(home, 'IrrigationGIS', 'tfrecords', 'test')
    out_bucket = 'ts_data'
    bucket_dir = 'cmask/tar/train/train_patches'
    tf_recs = 'gs://ts_data/cmask/train'
    write_npy_gcs(tf_recs, bucket=out_bucket, bucket_dst=bucket_dir)

# ========================= EOF ====================================================================
