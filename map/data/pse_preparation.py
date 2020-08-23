
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from tqdm import tqdm
import json
import os


def crop_and_write(output_path, tifs, date_index, polygons, lab_rpg):
    nparcels = len(polygons)
    ndates = len(date_index)

    prepare_output(output_path)

    count = 0
    dates = {}
    sizes = {}
    labels = dict([(l, {}) for l in lab_rpg.keys()])

    for tifname in tifs:

        date = date_parser(tifname)
        print(date)
        didx = date_index[date]

        dates[didx] = date
        with open(os.path.join(output_path, 'META', 'dates.json'), 'w') as file:
            file.write(json.dumps(dates, indent=4))

        with rasterio.open(tifname) as src:
            for parcel_id, p in tqdm(polygons.items()):
                try:
                    # crop parcel
                    try:
                        crop, transform = rasterio.mask.mask(src, [p], crop=True, nodata=np.iinfo(np.uint16).max)
                    except ValueError:
                        # Error when polygon does not complelety overlap raster, those (few) polygons are not included in the dataset
                        continue
                    # Get pixels with data for each channel
                    pixels = []
                    for i in range(crop.shape[0]):
                        c = crop[i][np.where(crop[i] != np.iinfo(np.uint16).max)]
                        pixels.append(c)
                    pixels = np.stack(pixels, axis=0)  # (C, S)

                    pixels = pixels.reshape((1, *pixels.shape))  # (1, C, S)

                    # Get pixels from already computed dates
                    if didx != 0:
                        previous_dates_pixels = np.load(
                            os.path.join(output_path, 'DATA', '{}.npy'.format(str(parcel_id))))
                        pixels = np.concatenate([previous_dates_pixels, pixels], axis=0)

                    # Write new array of pixels (shape TxCxS: time , channel, number of pixels)
                    np.save(os.path.join(output_path, 'DATA', str(parcel_id)), pixels)

                    if count == 0:  # We need to do this only once
                        sizes[parcel_id] = pixels.shape[-1]
                        for l in labels.keys():
                            labels[l][parcel_id] = lab_rpg[l][parcel_id]

                except KeyboardInterrupt:
                    raise
                except:
                    print('ERROR in {}'.format(parcel_id))

        if count == 0:
            with open(os.path.join(output_path, 'META', 'labels.json'), 'w') as file:
                file.write(json.dumps(labels, indent=4))
            with open(os.path.join(output_path, 'META', 'sizes.json'), 'w') as file:
                file.write(json.dumps(sizes, indent=4))

        count += 1


def list_extension(folder, extension='tif'):
    """
    Lists files in folder with the specified extension
    """
    return [f for f in os.listdir(folder) if str(f).endswith(extension)]


def date_parser(filepath):
    """
    returns the date (as int) from the file name of an S2 image
    """
    filename = os.path.split(filepath)[-1]
    return int(str(filename).split('.')[0])


def prepare_output(output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'META'), exist_ok=True)


if __name__ == '__main__':
    rpg_file = 'rpg_2017_T31TFM.geojson'
    input_folder = './S2-L2A-2017-T31TFM'
    out_path = './PixelSet-S2-2017-T31TFM'

    prepare_dataset(out_path, input_folder, rpg_file, label_names=['CODE_GROUP'])
if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
