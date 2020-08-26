import json
import os
import scipy.ndimage.measurements as mnts
from numpy import stack, zeros_like, array, sort, sum, where
from map.trainer.training_utils import make_test_dataset

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

structure = array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])


def write_pixel_set(out, recs, label_names=None):
    dataset = make_test_dataset(recs, True).batch(1)
    count = 0
    sizes = {}
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        features = features.numpy().squeeze()
        pixels = []

        bbox_slices = {}
        for i in range(0, labels.shape[2]):
            B = labels[:, :, i].copy()
            print(B.max())
            if B.max():
                bbox_slices[i] = mnts.find_objects(mnts.label(B, structure=structure)[0])
        print(bbox_slices)
        pixels.append([])
        pixels = stack(pixels, axis=0)  # (C, S)
        pixels = pixels.reshape((1, *pixels.shape))  # (1, C, S)
        with open(os.path.join(out, 'META', 'dates.json'), 'w') as file:
            file.write(json.dumps(DATES, indent=4))


def display_box(features, labels):
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])

    for n in range(labels.shape[-1]):
        labels[:, :, n] *= n + 1

    labels = sum(labels, axis=-1)
    features = features.numpy().squeeze()
    lat, lon = features[:, :, -3].mean(), features[:, :, -2].mean()
    boxes = zeros_like(labels)
    boxes = where()

    fig, ax = plt.subplots(ncols=5)
    ax[0].imshow(labels, cmap=cmap)
    plt.suptitle('{:.3f}, {:.3f}'.format(lat, lon))
    plt.show()


def date_parser(filepath):
    """
    returns the date (as int) from the file name of an S2 image
    """
    filename = os.path.split(filepath)[-1]
    return int(str(filename).split('.')[0])


def list_extension(folder, extension='tif'):
    """
    Lists files in folder with the specified extension
    """
    return [f for f in os.listdir(folder) if str(f).endswith(extension)]


def get_dates(input_folder):
    tifs = list_extension(input_folder, '.tif')
    tifs = sort(tifs)
    dates = [int(t.replace('.tif', '')) for t in tifs]

    ndates = len(dates)

    date_index = dict(zip(dates, range(ndates)))

    print('{} dates found in input folder '.format(ndates))
    tifs = [os.path.join(input_folder, f) for f in tifs]

    return tifs, date_index


def prepare_output(output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'META'), exist_ok=True)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tf_recs = os.path.join(home, 'IrrigationGIS', 'tfrecords')
    out_path = './PixelSet-S2-2017-T31TFM'

    write_pixel_set(out_path, tf_recs, label_names=['CODE_GROUP'])

# ========================= EOF ====================================================================
