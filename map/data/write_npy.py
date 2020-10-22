import os
import numpy as np
import tempfile
import shutil
import tarfile
import torch
from google.cloud import storage

try:
    from map.trainer.training_utils import make_test_dataset
    from map.data.bucket import get_bucket_contents
except ModuleNotFoundError:
    from trainer.training_utils import make_test_dataset
    from data.bucket import get_bucket_contents


def write_tfr_to_gcs(recs, bucket=None, bucket_dst=None, pattern='*gz', category='irrigated', start_count=0):
    """ Write tfrecord.gz to numpy, push .tar of torch tensor.pth to GCS bucket"""
    storage_client = storage.Client()

    def push_tar(t_dir, bckt, items, ind):
        tar_filename = '{}_{}.tar'.format(category, str(ind).zfill(6))
        tar_archive = os.path.join(t_dir, tar_filename)
        with tarfile.open(tar_archive, 'w') as tar:
            for i in items:
                tar.add(i, arcname=os.path.basename(i))
        bucket = storage_client.get_bucket(bckt)
        blob_name = os.path.join(bucket_dst, tar_filename)
        blob = bucket.blob(blob_name)
        print('push {}'.format(blob_name))
        blob.upload_from_filename(tar_archive)
        shutil.rmtree(t_dir)

    count = start_count

    dataset = make_test_dataset(recs, pattern).batch(1)
    obj_ct = np.array([0, 0, 0, 0])
    tmpdirname = tempfile.mkdtemp()
    items = []
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(4)])
        obj_ct += classes
        features = features.numpy().squeeze()
        a = np.append(features, labels, axis=2)
        a = torch.from_numpy(a)
        tmp_name = os.path.join(tmpdirname, '{}.pth'.format(str(j).zfill(7)))
        torch.save(a, tmp_name)
        items.append(tmp_name)

        if len(items) == 20:
            push_tar(tmpdirname, bucket, items, count)
            tmpdirname = tempfile.mkdtemp()
            items = []
            count += 1

    if len(items) > 0:
        push_tar(tmpdirname, bucket, items, count)
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
    out_bucket = 'ts_data'
    _type = 'dryland'
    bucket_dir = 'cmask/tar/train/train_points/{}'.format(_type)
    tf_recs = 'gs://ts_data/cmask/points/unproc'
    glob_pattern = '*{}*gz'.format(_type)
    write_tfr_to_gcs(tf_recs, bucket=out_bucket, bucket_dst=bucket_dir, category=_type, start_count=308)

# ========================= EOF ====================================================================
