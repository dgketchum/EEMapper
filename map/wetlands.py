import os
import sys
from pprint import pprint
import numpy as np
import random
from collections import OrderedDict

import fiona
from rtree import index
from rasterstats import zonal_stats
from shapely.geometry import Polygon

pare = os.path.dirname(__file__)
proj = os.path.dirname(os.path.dirname(pare))
sys.path.append(pare)

MGRS_PATH = '/media/research/IrrigationGIS/wetlands/mgrs/MGRS_TILE.shp'

# from rtree import index
from shapely.geometry import shape, Polygon, LineString

irrmapper_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

east_states = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']


def cdl_crops():
    return {1: 'Corn',
            2: 'Cotton',
            3: 'Rice',
            4: 'Sorghum',
            5: 'Soybeans',
            6: 'Sunflower',
            10: 'Peanuts',
            11: 'Tobacco',
            12: 'Sweet Corn',
            13: 'Pop or Orn Corn',
            14: 'Mint',
            21: 'Barley',
            22: 'Durum Wheat',
            23: 'Spring Wheat',
            24: 'Winter Wheat',
            25: 'Other Small Grains',
            26: 'Dbl Crop WinWht / Soybeans',
            27: 'Rye',
            28: 'Oats',
            29: 'Millet',
            30: 'Speltz',
            31: 'Canola',
            32: 'Flaxseed',
            33: 'Safflower',
            34: 'Rape Seed',
            35: 'Mustard',
            36: 'Alfalfa',
            37: 'Other Hay / NonAlfalfa',
            38: 'Camelina',
            39: 'Buckwheat',
            41: 'Sugarbeets',
            42: 'Dry Beans',
            43: 'Potatoes',
            44: 'Other Crops',
            45: 'Sugarcane',
            46: 'Sweet Potatoes',
            47: 'Misc Vegs & Fruits',
            48: 'Watermelons',
            49: 'Onions',
            50: 'Cucumbers',
            51: 'Chick Peas',
            52: 'Lentils',
            53: 'Peas',
            54: 'Tomatoes',
            55: 'Caneberries',
            56: 'Hops',
            57: 'Herbs',
            58: 'Clover/Wildflowers',
            59: 'Sod/Grass Seed',
            61: 'Fallow/Idle Cropland',
            66: 'Cherries',
            67: 'Peaches',
            68: 'Apples',
            69: 'Grapes',
            70: 'Christmas Trees',
            71: 'Other Tree Crops',
            72: 'Citrus',
            74: 'Pecans',
            75: 'Almonds',
            76: 'Walnuts',
            77: 'Pears',
            204: 'Pistachios',
            205: 'Triticale',
            206: 'Carrots',
            207: 'Asparagus',
            208: 'Garlic',
            209: 'Cantaloupes',
            210: 'Prunes',
            211: 'Olives',
            212: 'Oranges',
            213: 'Honeydew Melons',
            214: 'Broccoli',
            216: 'Peppers',
            217: 'Pomegranates',
            218: 'Nectarines',
            219: 'Greens',
            220: 'Plums',
            221: 'Strawberries',
            222: 'Squash',
            223: 'Apricots',
            224: 'Vetch',
            225: 'Dbl Crop WinWht/Corn',
            226: 'Dbl Crop Oats/Corn',
            227: 'Lettuce',
            229: 'Pumpkins',
            230: 'Dbl Crop Lettuce/Durum Wht',
            231: 'Dbl Crop Lettuce/Cantaloupe',
            232: 'Dbl Crop Lettuce/Cotton',
            233: 'Dbl Crop Lettuce/Barley',
            234: 'Dbl Crop Durum Wht/Sorghum',
            235: 'Dbl Crop Barley/Sorghum',
            236: 'Dbl Crop WinWht/Sorghum',
            237: 'Dbl Crop Barley/Corn',
            238: 'Dbl Crop WinWht/Cotton',
            239: 'Dbl Crop Soybeans/Cotton',
            240: 'Dbl Crop Soybeans/Oats',
            241: 'Dbl Crop Corn/Soybeans',
            242: 'Blueberries',
            243: 'Cabbage',
            244: 'Cauliflower',
            245: 'Celery',
            246: 'Radishes',
            247: 'Turnips',
            248: 'Eggplants',
            249: 'Gourds',
            250: 'Cranberries',
            254: 'Dbl Crop Barley/Soybeans'}


def cdl_key():
    return {1: 'Corn',
            2: 'Cotton',
            3: 'Rice',
            4: 'Sorghum',
            5: 'Soybeans',
            6: 'Sunflower',
            7: '',
            8: '',
            9: '',
            10: 'Peanuts',
            11: 'Tobacco',
            12: 'Sweet Corn',
            13: 'Pop or Orn Corn',
            14: 'Mint',
            15: '',
            16: '',
            17: '',
            18: '',
            19: '',
            20: '',
            21: 'Barley',
            22: 'Durum Wheat',
            23: 'Spring Wheat',
            24: 'Winter Wheat',
            25: 'Other Small Grains',
            26: 'Dbl Crop WinWht/Soybeans',
            27: 'Rye',
            28: 'Oats',
            29: 'Millet',
            30: 'Speltz',
            31: 'Canola',
            32: 'Flaxseed',
            33: 'Safflower',
            34: 'Rape Seed',
            35: 'Mustard',
            36: 'Alfalfa',
            37: 'Other Hay/Non Alfalfa',
            38: 'Camelina',
            39: 'Buckwheat',
            40: '',
            41: 'Sugarbeets',
            42: 'Dry Beans',
            43: 'Potatoes',
            44: 'Other Crops',
            45: 'Sugarcane',
            46: 'Sweet Potatoes',
            47: 'Misc Vegs & Fruits',
            48: 'Watermelons',
            49: 'Onions',
            50: 'Cucumbers',
            51: 'Chick Peas',
            52: 'Lentils',
            53: 'Peas',
            54: 'Tomatoes',
            55: 'Caneberries',
            56: 'Hops',
            57: 'Herbs',
            58: 'Clover/Wildflowers',
            59: 'Sod/Grass Seed',
            60: 'Switchgrass',
            61: 'Fallow/Idle Cropland',
            62: 'Pasture/Grass',
            63: 'Forest',
            64: 'Shrubland',
            65: 'Barren',
            66: 'Cherries',
            67: 'Peaches',
            68: 'Apples',
            69: 'Grapes',
            70: 'Christmas Trees',
            71: 'Other Tree Crops',
            72: 'Citrus',
            73: '',
            74: 'Pecans',
            75: 'Almonds',
            76: 'Walnuts',
            77: 'Pears',
            78: '',
            79: '',
            80: '',
            81: 'Clouds/No Data',
            82: 'Developed',
            83: 'Water',
            84: '',
            85: '',
            86: '',
            87: 'Wetlands',
            88: 'Nonag/Undefined',
            89: '',
            90: '',
            91: '',
            92: 'Aquaculture',
            93: '',
            94: '',
            95: '',
            96: '',
            97: '',
            98: '',
            99: '',
            100: '',
            101: '',
            102: '',
            103: '',
            104: '',
            105: '',
            106: '',
            107: '',
            108: '',
            109: '',
            110: '',
            111: 'Open Water',
            112: 'Perennial Ice/Snow',
            113: '',
            114: '',
            115: '',
            116: '',
            117: '',
            118: '',
            119: '',
            120: '',
            121: 'Developed/Open Space',
            122: 'Developed/Low Intensity',
            123: 'Developed/Med Intensity',
            124: 'Developed/High Intensity',
            125: '',
            126: '',
            127: '',
            128: '',
            129: '',
            130: '',
            131: 'Barren',
            132: '',
            133: '',
            134: '',
            135: '',
            136: '',
            137: '',
            138: '',
            139: '',
            140: '',
            141: 'Deciduous Forest',
            142: 'Evergreen Forest',
            143: 'Mixed Forest',
            144: '',
            145: '',
            146: '',
            147: '',
            148: '',
            149: '',
            150: '',
            151: '',
            152: 'Shrubland',
            153: '',
            154: '',
            155: '',
            156: '',
            157: '',
            158: '',
            159: '',
            160: '',
            161: '',
            162: '',
            163: '',
            164: '',
            165: '',
            166: '',
            167: '',
            168: '',
            169: '',
            170: '',
            171: '',
            172: '',
            173: '',
            174: '',
            175: '',
            176: 'Grassland/Pasture',
            177: '',
            178: '',
            179: '',
            180: '',
            181: '',
            182: '',
            183: '',
            184: '',
            185: '',
            186: '',
            187: '',
            188: '',
            189: '',
            190: 'Woody Wetlands',
            191: '',
            192: '',
            193: '',
            194: '',
            195: 'Herbaceous Wetlands',
            196: '',
            197: '',
            198: '',
            199: '',
            200: '',
            201: '',
            202: '',
            203: '',
            204: 'Pistachios',
            205: 'Triticale',
            206: 'Carrots',
            207: 'Asparagus',
            208: 'Garlic',
            209: 'Cantaloupes',
            210: 'Prunes',
            211: 'Olives',
            212: 'Oranges',
            213: 'Honeydew Melons',
            214: 'Broccoli',
            215: 'Avocados',
            216: 'Peppers',
            217: 'Pomegranates',
            218: 'Nectarines',
            219: 'Greens',
            220: 'Plums',
            221: 'Strawberries',
            222: 'Squash',
            223: 'Apricots',
            224: 'Vetch',
            225: 'Dbl Crop WinWht/Corn',
            226: 'Dbl Crop Oats/Corn',
            227: 'Lettuce',
            228: '',
            229: 'Pumpkins',
            230: 'Dbl Crop Lettuce/Durum Wht',
            231: 'Dbl Crop Lettuce/Cantaloupe',
            232: 'Dbl Crop Lettuce/Cotton',
            233: 'Dbl Crop Lettuce/Barley',
            234: 'Dbl Crop Durum Wht/Sorghum',
            235: 'Dbl Crop Barley/Sorghum',
            236: 'Dbl Crop WinWht/Sorghum',
            237: 'Dbl Crop Barley/Corn',
            238: 'Dbl Crop WinWht/Cotton',
            239: 'Dbl Crop Soybeans/Cotton',
            240: 'Dbl Crop Soybeans/Oats',
            241: 'Dbl Crop Corn/Soybeans',
            242: 'Blueberries',
            243: 'Cabbage',
            244: 'Cauliflower',
            245: 'Celery',
            246: 'Radishes',
            247: 'Turnips',
            248: 'Eggplants',
            249: 'Gourds',
            250: 'Cranberries',
            251: '',
            252: '',
            253: '',
            254: 'Dbl Crop Barley/Soybeans',
            255: ''}


def zonal_cdl(in_shp, in_raster, out_shp=None,
              select_codes=None, write_non_crop=False):
    ct = 1
    geo = []
    bad_geo_ct = 0
    with fiona.open(in_shp) as src:
        meta = src.meta
        for feat in src:
            try:
                _ = feat['geometry']['type']
                geo.append(feat)
            except TypeError:
                bad_geo_ct += 1

    input_feats = len(geo)
    temp_file = out_shp.replace('.shp', '_temp.shp')
    with fiona.open(temp_file, 'w', **meta) as tmp:
        for feat in geo:
            tmp.write(feat)

    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('FID', 'int:9'), ('CDL', 'int:9')]), 'geometry': 'Polygon'}

    stats = zonal_stats(temp_file, in_raster, stats=['majority'], nodata=0.0, categorical=False)

    if select_codes:
        include_codes = select_codes
    else:
        include_codes = [k for k in cdl_crops().keys()]

    ct_inval = 0
    with fiona.open(out_shp, mode='w', **meta) as out:
        for attr, g in zip(stats, geo):
            try:
                cdl = int(attr['majority'])
            except TypeError:
                cdl = 0

            if attr['majority'] in include_codes and not write_non_crop:
                feat = {'type': 'Feature',
                        'properties': {'FID': ct,
                                       'CDL': cdl},
                        'geometry': g['geometry']}
                if not feat['geometry']:
                    ct_inval += 1
                elif not shape(feat['geometry']).is_valid:
                    ct_inval += 1
                else:
                    out.write(feat)
                    ct += 1

            elif write_non_crop and cdl not in include_codes:
                feat = {'type': 'Feature',
                        'properties': {'FID': ct,
                                       'CDL': cdl},
                        'geometry': g['geometry']}
                if not feat['geometry']:
                    ct_inval += 1
                elif not shape(feat['geometry']).is_valid:
                    ct_inval += 1
                else:
                    out.write(feat)
                    ct += 1

        print('{} in, {} out, {} invalid, {}'.format(input_feats, ct - 1, ct_inval, out_shp))
        os.remove(temp_file)


def split_by_mgrs(shapes, out_dir):
    """attribute source code, split into MGRS tiles"""
    out_features = []
    tiles = []
    idx = index.Index()
    in_features = []

    with fiona.open(shapes[0]) as src:
        meta = src.meta
        schema = src.schema
        schema['properties']['MGRS_TILE'] = 'str:15'
        schema['properties']['popper'] = 'float:19.11'
        meta['schema'] = schema

    for _file in shapes:
        with fiona.open(_file) as src:
            for f in src:
                a = f['properties']['Shape_Area']
                l = f['properties']['Shape_Leng']
                p = (4 * np.pi * a) / (l ** 2.)
                f['properties']['popper'] = p
                in_features.append(f)

    print('{} features in {}'.format(len(in_features), shapes))

    with fiona.open(MGRS_PATH, 'r') as mgrs:
        [idx.insert(i, shape(tile['geometry']).bounds) for i, tile in enumerate(mgrs)]
        for f in in_features:
            try:
                point = shape(f['geometry']).centroid
                for j in idx.intersection(point.coords[0]):
                    if point.within(shape(mgrs[j]['geometry'])):
                        tile = mgrs[j]['properties']['MGRS_TILE']
                        if tile not in tiles:
                            tiles.append(tile)
                        break
                f['properties']['MGRS_TILE'] = tile
                out_features.append(f)

            except AttributeError as e:
                print(e)

    for tile in tiles:
        dir_ = os.path.join(out_dir, tile)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
        file_name = '{}.shp'.format(tile)
        out_shape = os.path.join(dir_, file_name)
        with fiona.open(out_shape, 'w', **meta) as output:
            ct = 0
            for feat in out_features:
                if feat['properties']['MGRS_TILE'] == tile:
                    if not feat['geometry']:
                        print('None Geo, skipping')
                    elif not shape(feat['geometry']).is_valid:
                        print('Invalid Geo, skipping')
                    else:
                        output.write(feat)
                        ct += 1
        if ct == 0:
            [os.remove(os.path.join(dir_, x)) for x in os.listdir(dir_) if file_name in x]
            print('Not writing {}'.format(file_name))
        else:
            print('wrote {}, {} features'.format(out_shape, ct))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    alt_home = '/media/research'
    if os.path.isdir(alt_home):
        home = alt_home
    else:
        home = os.path.join(home, 'data')

    states = irrmapper_states + east_states
    gis = os.path.join(home, 'IrrigationGIS', 'wetlands')
    raw = os.path.join(gis, 'raw_test')
    mgrs = os.path.join(gis, 'mgrs')
    for s in states:
        files_ = [os.path.join(raw, x) for x in os.listdir(raw) if s in x and x.endswith('.shp')]
        s_dir = os.path.join(mgrs, 'split', s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        split_by_mgrs(files_, s_dir)

# ========================= EOF ====================================================================