import numpy as np
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from collections import defaultdict
from sklearn.metrics import confusion_matrix
from map.trainer import feature_spec

features_dict = feature_spec.features_dict()
bands = feature_spec.bands()
features = feature_spec.features()


def one_hot(labels, n_classes):
    h, w, d = labels.shape
    labels = tf.squeeze(labels)
    ls = []
    for i in range(n_classes):
        if i == 0:
            where = tf.where(labels != i + 1, tf.zeros((h, w)), 1 * tf.ones((h, w)))
        else:
            where = tf.where(labels != i + 1, tf.zeros((h, w)), tf.ones((h, w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp


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


def confusion_matrix_from_generator(datasets, batch_size, model, n_classes=3):
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


def parse_tfrecord(example_proto):
    """the parsing function.
    read a serialized example into the structure defined by features_dict.
    args:
      example_proto: a serialized example.
    returns:
      a dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, features_dict)


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


def filter_list_into_classes(lst):
    out = defaultdict(list)
    for f in lst:
        if 'irrigated' in f and 'unirrigated' not in f:
            out['irrigated'].append(f)
        elif 'unirrigated' in f or 'fallow' in f:
            out['unirrigated'].append(f)
        elif 'uncultivated' in f or 'wetlands' in f:
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
        buffer_size=config.BUFFER_SIZE)
    return dataset


def make_test_dataset(root, batch_size=16):
    pattern = "*gz"
    datasets = []
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
    glob = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)

    return dataset


def to_tuple(inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in sorted(features)]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    inputs = stacked[:, :, :len(bands)] * 0.0001
    labels = one_hot(stacked[:, :, len(bands):], n_classes=5)
    labels = tf.cast(labels, tf.int32)
    return inputs, labels


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    home = os.path.expanduser('~')
    tf_recs = os.path.join(home, 'IrrigationGIS', 'tfrecords')
    dataset = make_test_dataset(tf_recs, True).batch(1)

    for features, labels in dataset:
        lb = np.sum(labels, axis=-1) == 0
        lab = np.argmax(labels, axis=-1).astype(np.float32)
        lab[lb] = np.nan
        f, ax = plt.subplots(ncols=2)
        ax[0].imshow(lab.squeeze())
        features = features.numpy().squeeze()
        vis = np.dstack((features[:, :, 3], features[:, :, 1], features[:, :, 0]))
        vis = vis / np.max(vis)
        ax[1].imshow(vis)
        # plt.colorbar()
        plt.show()
        break
