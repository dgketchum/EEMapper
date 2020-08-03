import tensorflow as tf

'''
Feature spec for reading/writing tf records
'''
features_dict_ = {'20100501_blue': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100501_green': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100501_nir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100501_red': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100501_swir1': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100501_swir2': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100501_tir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_blue': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_green': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_nir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_red': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_swir1': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_swir2': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100531_tir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_blue': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_green': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_nir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_red': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_swir1': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_swir2': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100630_tir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_blue': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_green': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_nir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_red': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_swir1': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_swir2': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  '20100730_tir': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
                  'constant': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None)}

# features_dict_ = {'0_blue_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '0_green_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '0_nir_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '0_red_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '0_swir1_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '0_swir2_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '1_blue_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '1_green_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '1_nir_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '1_red_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '1_swir1_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '1_swir2_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '2_blue_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '2_green_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '2_nir_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '2_red_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '2_swir1_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '2_swir2_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '3_blue_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '3_green_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '3_nir_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '3_red_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '3_swir1_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '3_swir2_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '4_blue_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '4_green_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '4_nir_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '4_red_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '4_swir1_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '4_swir2_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '5_blue_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '5_green_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '5_nir_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '5_red_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '5_swir1_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   '5_swir2_mean': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None),
#                   'constant': tf.io.FixedLenFeature(shape=[256, 256], dtype=tf.float32, default_value=None)}


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
