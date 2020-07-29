import numpy as np
import time
import os
import tensorflow as tf
import tensorflow
from collections import defaultdict

from . import feature_spec
from . import config
# import matplotlib.pyplot as plt
# import feature_spec
# import config

features_dict = feature_spec.features_dict()
BANDS = feature_spec.bands() # includes mask raster
FEATURES = feature_spec.features() # only input features
NDVI_INDICES = [(2, 3), (8, 9), (14, 15), (20, 21), (26, 27), (32, 33)]

def tf_distance_map(mask):
    im_shape = mask.shape
    mask = tf.cast(mask, tf.bool)
    mask = tf.math.logical_not(mask) # make the non-masked areas masked
    [mask,] = tf.py_function(distance_map, [mask], [tf.float32])
    mask.set_shape(im_shape)
    return mask

def distance_map(mask):
    mask = distance_transform_edt(mask) 
    mask[0, 0] = 0
    return mask

def random_rotate_image(image):
    image = rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image

def tf_random_rotate_image(image):
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image

def one_hot(labels, n_classes):
    h, w = labels.shape
    labels = tf.squeeze(labels)
    ls = []
    for i in range(n_classes):
        where = tf.where(labels != i+1, tf.zeros((h, w)), 1*tf.ones((h,w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp

def one_hot_border_labels(labels, n_classes):
    h, w, d = labels.shape
    labels = tf.squeeze(labels)
    ls = []
    border_labels = None
    for i in range(n_classes):
        if i == 0:
            # informative names, here
            where = tf.where(labels != i+1, tf.zeros((h, w)), 1*tf.ones((h,w)))
            border_labels = tf_distance_map(where)
            border_labels = tf.where(border_labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
        elif i == 2 and border_labels is not None:
            where = tf.where(labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
            where = tf.where(border_labels == 1, 10*tf.ones((h, w)), where)
        else:
            where = tf.where(labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
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


def confusion_matrix_from_generator(datasets, batch_size, model, n_classes):
    ''' 
    inputs: list of tf.data.Datasets, not batched, without repeat.
    '''   
    out_cmat = np.zeros((n_classes, n_classes))
    labels = range(n_classes)
    instance_count = 0
    for dataset in datasets:
        for batch in dataset:
            features, y_true = batch[0], batch[1]
            y_pred = model(features)['logits'].numpy()
            instance_count += y_pred.shape[0]
            y_true, y_pred = mask_unlabeled_values(y_true, y_pred)
            cmat = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes)
            out_cmat += cmat
    precision_dict = {}
    recall_dict = {}
    for i in range(n_classes):
        precision_dict[i] = 0
        recall_dict[i] = 0
    for i in range(n_classes):
        precision_dict[i] = out_cmat[i, i] / np.sum(out_cmat[i, :]) # row i
        recall_dict[i] = out_cmat[i, i] / np.sum(out_cmat[:, i]) # column i
    return out_cmat, recall_dict, precision_dict, instance_count


def get_dataset(pattern, add_ndvi):
  """Function to read, parse and format to tuple a set of input tfrecord files.
  Get all the files matching the pattern, parse and convert to tuple.
  Args:
    pattern: A file pattern to match in a Cloud Storage bucket,
             or list of GCS files
  Returns:
    A tf.data.Dataset
  """
  if not isinstance(pattern, list):
      pattern = tf.io.gfile.glob(pattern)
  dataset = tf.data.TFRecordDataset(pattern, compression_type='GZIP',
          num_parallel_reads=8)
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
  to_tup = to_tuple(add_ndvi)
  dataset = dataset.map(to_tup, num_parallel_calls=5)
  return dataset

def parse_tfrecord(example_proto):
  """the parsing function.
  read a serialized example into the structure defined by features_dict.
  args:
    example_proto: a serialized example.
  returns:
    a dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, features_dict)

def to_tuple(add_ndvi):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns:
    A tuple of (inputs, outputs).
  """
  def to_tup(inputs):
      features_list = [inputs.get(key) for key in sorted(FEATURES)]
      stacked = tf.stack(features_list, axis=0)
      # Convert from CHW to HWC
      stacked = tf.transpose(stacked, [1, 2, 0]) * 0.0001
      if add_ndvi:
          image_stack = add_ndvi_raster(stacked)
      else:
          image_stack = stacked
      # 'constant' is the label for label raster. 
      labels = one_hot(inputs.get('constant'), n_classes=5)
      labels = tf.cast(labels, tf.int32)
      return image_stack, labels

  return to_tup

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
        ndvi = (image_stack[:,:, nir_idx] - image_stack[:,:, red_idx]) /\
                (image_stack[:,:, nir_idx] + image_stack[:,:, red_idx] + 1e-8) 
        out.append(ndvi)
    return tf.concat((image_stack, tf.stack(out, axis=-1)), axis=-1)

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

def make_training_dataset(root, add_ndvi, batch_size=16):
    pattern = "*gz"
    files = tf.io.gfile.glob(os.path.join(root, pattern))
    dataset = get_dataset(files)
    dataset = dataset.batch(batch_size).repeat().shuffle(buffer_size=config.BUFFER_SIZE)
    return dataset

def make_balanced_training_dataset(root, add_ndvi, batch_size=16):
    pattern = "*gz"
    datasets = []
    files = tf.io.gfile.glob(os.path.join(root, pattern))
    files = filter_list_into_classes(files) # So I don't have to move files
    # into separate directories; just use their names.
    for class_name, file_list in files.items():
        dataset = get_dataset(file_list)
        datasets.append(dataset.repeat())
    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    dataset = tf.data.experimental.choose_from_datasets(datasets,
            choice_dataset).batch(batch_size).repeat().shuffle(buffer_size=config.BUFFER_SIZE)
    return dataset

def sort_files_into_years(files):

    year_to_files = defaultdict(list)

    for f in files:
        filename = os.path.basename(f)
        year = int(filename[:4])
        year_to_files[year].append(f)

    return year_to_files

def make_yearly_test_dataset(root, add_ndvi, batch_size=16):
    pattern = "*gz"
    test_root = os.path.join(root, pattern)
    files = tf.io.gfile.glob(test_root)
    year_to_files = sort_files_into_years(files)
    year_to_dataset = {}
    for year, files in year_to_files.items():
        year_to_dataset[year] = get_dataset(files, add_ndvi)
    return year_to_dataset

good_years = set([2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015])

def filter_unirrigated_years(files):

    out = []
    for f in files:
        filename = os.path.basename(f)
        year = int(filename[:4])
        if year in good_years:
            out.append(f)
    return out


def make_test_dataset(root, add_ndvi, batch_size=16):
    pattern = "*gz"
    training_root = os.path.join(root, pattern)
    files = tf.io.gfile.glob(training_root)
    # files = filter_unirrigated_years(files)
    datasets = get_dataset(files, add_ndvi).batch(config.BATCH_SIZE)
    return datasets

def md(root, add_ndvi, batch_size=16):
    pattern = "*gz"
    training_root = os.path.join(root, pattern)
    files = tf.io.gfile.glob(training_root)
    datasets = get_dataset(files, add_ndvi).batch(1)
    return datasets


if __name__ == '__main__':
    dataset = make_test_dataset('/tmp/', True)
    '''
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy()
        features = features.numpy()
        print(np.max(features))
        if len(features[np.isnan(features)]):
            for i in range(features.shape[-1]):
                slc = features[0, :, :, i]
                if len(slc[np.isnan(slc)]):
                    fig, ax = plt.subplots(ncols=3)
                    ax[0].imshow(features[0, :, :, 32])
                    ax[1].imshow(features[0, :, :, 33])
                    ax[2].imshow(features[0, :, :, 41])
                    plt.show()
    mask = np.sum(d, axis=-1) == 0
    d = np.argmax(d, axis=-1).astype(np.float32)
    d[mask] = np.nan
    for j in range(35, 42):
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(i[0,:, :, j])
        ax[1].imshow(d.squeeze())
        plt.suptitle(j)
        plt.show()
    '''
