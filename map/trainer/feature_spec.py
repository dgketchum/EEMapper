import tensorflow as tf

'''
Feature spec for reading/writing tf records
'''
features_dict_ = {'blue_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'blue_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'green_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'nir_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'red_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir1_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'swir2_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100101': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100131': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100302': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100401': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100501': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100531': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100630': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100730': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100829': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20100928': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20101028': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20101127': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'tir_20101227': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'elev': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'slope': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'aspect': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'lat': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'lon': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'cdl': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'irr': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  }


def features_dict():
    return features_dict_


def bands():
    bands = list(features_dict_.keys())
    return bands


def features():
    features = list(features_dict_.keys())[:-1]
    return features


if __name__ == '__main__':
    print(len(features_dict_))
    pass
