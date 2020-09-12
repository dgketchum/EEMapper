# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os

import numpy as np

import rasterio
from rasterio.transform import from_origin

from map.trainer import feature_spec

MODE = 'irr'
FEATURES_DICT = feature_spec.features_dict()
FEATURES = feature_spec.features()
step_, length_ = 7, len(FEATURES)

r_idx, g_idx, b_idx = [FEATURES.index(x) for x in FEATURES if 'red' in x], \
                      [FEATURES.index(x) for x in FEATURES if 'green' in x], \
                      [FEATURES.index(x) for x in FEATURES if 'blue' in x]

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])


def build_raster(npy_dir, out_tif_dir):
    l = [os.path.join(npy_dir, x) for x in os.listdir(npy_dir)]
    for j, f in enumerate(l):

        a = np.load(f)
        labels = a[:, :, -4:]
        features = a[:, :, :-4]
        for n in range(labels.shape[-1]):
            labels[:, :, n] *= n + 1
        labels = np.sum(labels, axis=-1)
        # features = features.numpy().squeeze()

        for i, n in enumerate(features):
            print(i, features[:, :, i].mean())

        r, g, b = features[:, :, r_idx], features[:, :, g_idx], features[:, :, b_idx]

        norm = lambda arr: ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        rgb = map(norm, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
        rgb = np.dstack(rgb)

        lat, lon = features[:, :, -3].mean(), features[:, :, -2].mean()

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(rgb)
        ax[1].imshow(labels, cmap=cmap)
        plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
        plt.show()

        transform = from_origin(472137, 5015782, 0.5, 0.5)
        new_dataset = rasterio.open('test1.tif', 'w', driver='GTiff',
                                    height=rgb.shape[0], width=rgb.shape[1],
                                    count=1, dtype=str(rgb.dtype),
                                    crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                                    transform=transform)

        new_dataset.write(rgb, 1)
        new_dataset.close()


if __name__ == '__main__':
    npy = '/home/dgketchum/PycharmProjects/EEMapper/map/data/npy'
    tif = '/home/dgketchum/PycharmProjects/EEMapper/map/data/tif'

# ========================= EOF ====================================================================
