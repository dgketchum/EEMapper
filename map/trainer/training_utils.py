import numpy as np
import time
import os
import json
import io
from pprint import pprint
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from numpy import median, dstack, sum, count_nonzero, unique, vectorize

from collections import defaultdict
from sklearn.metrics import confusion_matrix

from map.trainer.config import BUFFER_SIZE
from map.trainer import feature_spec

MODE = 'irr'
FEATURES_DICT = feature_spec.features_dict()
FEATURES = feature_spec.features()
step_, length_ = 7, len(FEATURES)
NDVI_INDICES = [(x, y) for x, y in zip(range(2, length_, step_), range(3, length_, step_))]


def mask_unlabeled_values(y_true, y_pred):
    '''
    y_pred: softmaxed tensor
    y_true: one-hot tensor of labels
    Returns two vectors of labels. Assumes input
    tensors are 4-dimensional (batchxrowxcolxdepth)
    '''
    mask = tf.not_equal(tf.reduce_sum(y_true, axis=-1), 0)
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return y_true, y_pred


def confusion_matrix_from_generator(datasets, batch_size, model, n_classes=4):
    '''
    inputs: list of tf.data.Datasets, not batched, without repeat.
    '''
    out_cmat = np.zeros((n_classes, n_classes))
    labels = range(n_classes)
    instance_count = 0
    uniq = defaultdict(int)
    for dataset in datasets:
        dataset = dataset.batch(batch_size)
        for batch in dataset:
            features, y_true = batch[0], batch[1]
            y_pred = model(features)['logits']
            instance_count += y_pred.shape[0]
            y_true, y_pred = mask_unlabeled_values(y_true, y_pred)
            unique, counts = np.unique(y_true, return_counts=True)
            for u, c in zip(unique, counts):
                uniq[u] += c
            cmat = confusion_matrix(y_true, y_pred, labels=labels)
            out_cmat += cmat
            # print(instance_count)
    precision_dict = {}
    recall_dict = {}
    for i in range(n_classes):
        precision_dict[i] = 0
        recall_dict[i] = 0
    for i in range(n_classes):
        precision_dict[i] = out_cmat[i, i] / np.sum(out_cmat[i, :])  # row i
        recall_dict[i] = out_cmat[i, i] / np.sum(out_cmat[:, i])  # column i
    return out_cmat.astype(np.int), recall_dict, precision_dict, instance_count, uniq


def m_acc(y_true, y_pred):
    y_true_sum = tf.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    acc = K.mean(K.equal(y_pred_masked, y_true_masked))
    return acc


def add_ndvi_raster(image_stack):
    '''
    These indices are hardcoded, and taken from the
    sorted keys in feature_spec.
    (NIR - Red) / (NIR + Red)
        2 0_nir_mean
        3 0_red_mean
        8 1_nir_mean
        9 1_red_mean
        14 2_nir_mean
        15 2_red_mean
        20 3_nir_mean
        21 3_red_mean
        26 4_nir_mean
        27 4_red_mean
        32 5_nir_mean
        33 5_red_mean
    '''
    out = []
    for nir_idx, red_idx in NDVI_INDICES:
        # Add a small constant in the denominator to ensure
        # NaNs don't occur because of missing data. Missing
        # data (i.e. Landsat 7 scan line failure) is represented as 0
        # in TFRecord files. Adding \{epsilon} will barely
        # change the non-missing data, and will make sure missing data
        # is still 0 when it's fed into the model.
        ndvi = (image_stack[:, :, nir_idx] - image_stack[:, :, red_idx]) / \
               (image_stack[:, :, nir_idx] + image_stack[:, :, red_idx] + 1e-8)
        out.append(ndvi)
    stack = tf.concat((image_stack, tf.stack(out, axis=-1)), axis=-1)
    return stack


def parse_tfrecord(example_proto):
    """the parsing function.
    read a serialized example into the structure defined by features_dict.
    args:
      example_proto: a serialized example.
    returns:
      a dictionary of tensors, keyed by feature name.
    """
    parsed = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    return parsed


def filter_list_into_classes(lst):
    out = defaultdict(list)
    for f in lst:
        if 'irrigated' in f:
            out['irrigated'].append(f)
        elif 'fallow' in f:
            out['fallow'].append(f)
        elif 'dryland' in f:
            out['dryland'].append(f)
        elif 'uncultivated' in f:
            out['uncultivated'].append(f)

    return out


def make_training_dataset(root, batch_size=16):
    pattern = "*gz"
    datasets = []
    files = tf.io.gfile.glob(os.path.join(root, pattern))
    files = filter_list_into_classes(files)  # So I don't have to move files
    # into separate directories; just use their names.
    for class_name, file_list in files.items():
        dataset = get_dataset(file_list)
        datasets.append(dataset.repeat())
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    dataset = tf.data.experimental.choose_from_datasets(datasets,
                                                        choice_dataset).batch(batch_size).repeat().shuffle(
        buffer_size=BUFFER_SIZE)
    return dataset


def make_test_dataset(root):
    pattern = "*gz"
    training_root = os.path.join(root, pattern)
    datasets = get_dataset(training_root)
    return datasets


def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
      A tf.data.Dataset
    """
    if not isinstance(pattern, list):
        pattern = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(pattern, compression_type='GZIP',
                                      num_parallel_reads=8)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    to_tup = to_tuple(add_ndvi=False)
    dataset = dataset.map(to_tup, num_parallel_calls=5)
    return dataset


def one_hot(labels, n_classes):
    h, w = labels.shape
    labels = tf.squeeze(labels) - 1
    ls = []
    for i in range(n_classes):
        where = tf.where(labels != i + 1, tf.zeros((h, w)), 1 * tf.ones((h, w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp


def make_dataset(root, batch_size=16, training=True):
    paths = ['irrigated', 'uncultivated', 'unirrigated']
    pattern = "*gz"
    datasets = []
    for path in paths:
        if os.path.isdir(os.path.join(root, path)):
            training_root = os.path.join(root, path, pattern)
            dataset = get_dataset(training_root)
            if training:
                datasets.append(dataset.repeat())
            else:
                datasets.append(dataset)
    if not len(datasets):
        training_root = os.path.join(root, pattern)
        datasets = [get_dataset(training_root)]
    if not training:
        return datasets
    choice_dataset = tf.data.Dataset.range(len(paths)).repeat()
    dataset = tf.data.experimental.choose_from_datasets(datasets,
                                                        choice_dataset).batch(batch_size).repeat().shuffle(
        buffer_size=30)
    return dataset


def to_tuple(add_ndvi):
    """
    Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """

    def to_tup(inputs):
        features_list = [inputs.get(key) for key in FEATURES]
        stacked = tf.stack(features_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])  # TC scaled somehow: * 0.0001
        if add_ndvi:
            image_stack = add_ndvi_raster(stacked)
        else:
            image_stack = stacked
        # 'constant' is the label for label raster.
        labels = one_hot(inputs.get(MODE), n_classes=4)
        labels = tf.cast(labels, tf.int32)
        return image_stack, labels

    return to_tup


def inspect_tfrecord(rec):
    tf.executing_eagerly()

    raw_dataset = tf.data.TFRecordDataset(rec)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        m = json.loads(MessageToJson(example))
        l = m['features']['feature'].keys()
        f_keys = FEATURES_DICT.keys()
        missing = [k for k in f_keys if k not in l]
        print('{} missing {}'.format(rec, missing))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tf_rec = os.path.join(home, 'IrrigationGIS', 'tfrecords')
    recs = [os.path.join(tf_rec, x) for x in os.listdir(tf_rec) if x.endswith('2015.tfrecord')]
    for r in recs:
        inspect_tfrecord(r)
# ==========================================================================================================
