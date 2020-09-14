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

from pyproj import Proj, transform
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

lat_idx = [FEATURES.index(x) for x in FEATURES if 'lat' in x][0]
lon_idx = [FEATURES.index(x) for x in FEATURES if 'lon' in x][0]

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])


def build_raster(npy_dir, out_tif_dir, plot=False):
    l = [os.path.join(npy_dir, x) for x in os.listdir(npy_dir) if x.endswith('.npy')]
    for j, f in enumerate(l):

        a = np.load(f)
        labels = a[:, :, -4:]
        cdl = a[:, :, -6]
        cconf = a[:, :, -5]

        for n in range(labels.shape[-1]):
            labels[:, :, n] *= n + 1
        labels = np.sum(labels, axis=-1)

        features = a[:, :, :-4]
        r, g, b = features[:, :, r_idx], features[:, :, g_idx], features[:, :, b_idx]

        norm = lambda arr: ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255))
        rgb = map(norm, [np.median(r, axis=2), np.median(g, axis=2), np.median(b, axis=2)])
        rgb = np.dstack(rgb).astype('uint8')

        lat, lon = features[:, :, lat_idx].max(), features[:, :, lon_idx].min()
        if plot:
            fig, ax = plt.subplots(ncols=4)
            ax[0].imshow(rgb)
            ax[1].imshow(labels, cmap=cmap)
            ax[2].imshow(cdl)
            ax[3].imshow(cconf)
            plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
            plt.show()

        in_proj = Proj('epsg:4326')
        out_proj = Proj('epsg:5070', preserve_units=True)
        x1, y1 = lon, lat
        lon, lat = transform(in_proj, out_proj, y1, x1)
        affine = from_origin(lon, lat, 30, 30)
        tif_name = os.path.join(out_tif_dir, '{}.tif'.format(j))

        meta = dict(driver='GTiff',
                    height=rgb.shape[0], width=rgb.shape[1],
                    count=3, dtype=str(rgb.dtype),
                    crs='epsg:5070',
                    transform=affine)

        with rasterio.open(tif_name, 'w', **meta) as dst:
            for b in range(rgb.shape[-1]):
                band = rgb[:, :, b].astype(rasterio.uint8)
                dst.write_band(b + 1, band)


if __name__ == '__main__':
    npy = '/home/dgketchum/PycharmProjects/IrrMapper/data/npy'
    tif = '/home/dgketchum/PycharmProjects/EEMapper/map/data/tif'
    build_raster(npy, tif, plot=False)
# ========================= EOF ====================================================================
