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

nhd_props = ['Shape_Leng', 'ACRES', 'WETLAND_TY', 'ATTRIBUTE', 'Shape_Area']


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
    print('{} features in {}'.format(input_feats, in_shp))
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
    ct_crop = 0
    ct_non_crop = 0
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
                    ct_crop += 1

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
                    ct_non_crop += 1

        print('{} in, {} out, {} invalid, {}'.format(input_feats, ct - 1, ct_inval, out_shp))
        os.remove(temp_file)


def process_pad(in_shp, out_shp):
    # after getting duplicates, I used QGIS for to_single_part and delete_duplicates
    # processing toolbox; select 'batch'
    ct, p_excl, overlaps = 0, 0, 0
    multi_ct, simple_ct, from_multi = 0, 0, 0
    excess_coords = 0
    in_features, existing_geo = [], []
    bad_geo_ct = 0
    with fiona.open(in_shp) as src:
        meta = src.meta
        meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
            [('FID', 'int:9'),
             ('area', 'float:19.11'),
             ('length', 'float:19.11'),
             ('popper', 'float:19.11')]),
                          'geometry': 'Polygon'}
        _features = []
        for feat in src:
            # sort by area
            geo = shape(feat['geometry'])
            a = geo.area
            _features.append((a, feat))

        sorted_features = sorted(_features, key=lambda x: x[0], reverse=True)
        for a, feat in sorted_features:
            if a < 2e5:
                continue
            try:
                simple_ct += 1
                geo = shape(feat['geometry'])

                overlapping = False
                for g in existing_geo:
                    if geo.intersects(g):
                        overlaps += 1
                        overlapping = True

                coord_ct = len(feat['geometry']['coordinates'][0])
                if coord_ct > 1000:
                    excess_coords += 1
                    continue

                l = geo.length
                p = (4 * np.pi * a) / (l ** 2.)
                feat['properties']['popper'] = p
                if p < 0.3:
                    p_excl += 1
                    continue

                feat['properties']['area'] = a
                feat['properties']['length'] = l
                feat['properties']['geometry'] = geo

                if not geo.is_valid:
                    bad_geo_ct += 1
                    continue

                if not overlapping:
                    existing_geo.append(geo)
                    in_features.append(feat)

                if len(in_features) % 1000 == 0:
                    print('{} features'.format(len(in_features)))

            except TypeError:
                bad_geo_ct += 1

    print('{} features\n{} bad\n{} overlaps excluded, {} multi\n{} single polygons'
          '\n{} from multipolygons\n{} excessive coordinates\n{} popper excluded'.format(
                                                            len(in_features), bad_geo_ct, overlaps,
                                                            multi_ct, simple_ct,
                                                            from_multi, excess_coords, p_excl))

    ct_inval = 0
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        for feat in in_features:
            f = {'type': 'Feature',
                 'properties': {'FID': ct,
                                'area': feat['properties']['area'],
                                'length': feat['properties']['length'],
                                'popper': feat['properties']['popper']},
                 'geometry': feat['geometry']}
            output.write(f)
            ct += 1
            if ct > 1000:
                exit()

    print('wrote {} features, {} invalid, to {}'.format(ct, ct_inval, out_shp))


def zonal_crop_mask(in_shp, in_raster, out_shp=None):
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
    print('{} features in {}'.format(input_feats, in_shp))
    temp_file = out_shp.replace('.shp', '_temp.shp')
    with fiona.open(temp_file, 'w', **meta) as tmp:
        for feat in geo:
            tmp.write(feat)

    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('FID', 'int:9'),
         ('mean', 'float:19.11'),
         ('popper', 'float:19.11'),
         ('area', 'float:19.11'),
         ('length', 'float:19.11')]),
                      'geometry': 'Polygon'}

    stats = zonal_stats(temp_file, in_raster, stats=['mean'], nodata=0.0, categorical=False)

    ct_inval = 0
    with fiona.open(out_shp, mode='w', **meta) as out:
        for attr, f in zip(stats, geo):
            cdl = attr['mean']
            feat = {'type': 'Feature',
                    'properties': {'FID': ct,
                                   'mean': cdl,
                                   'popper': f['properties']['popper'],
                                   'area': f['properties']['area'],
                                   'length': f['properties']['length']},
                    'geometry': f['geometry']}
            out.write(feat)
            ct += 1

    print('{} in, {} out, {} invalid, {}'.format(input_feats, ct - 1, ct_inval, out_shp))


def select_wetlands(shapes, out_shape, popper=0.15, min_acres=10):
    """attribute source code, split into MGRS tiles"""
    out_features = []
    inval_ct = 0
    with fiona.open(shapes[0]) as src:
        meta = src.meta
        schema = src.schema
        schema['properties']['popper'] = 'float:19.11'
        meta['schema'] = schema

    ct = 0
    for _file in shapes:
        print('reading {}'.format(_file))
        with fiona.open(_file) as src:
            for f in src:
                ct += 1
                acres = f['properties']['ACRES']
                a = f['properties']['Shape_Area']
                l = f['properties']['Shape_Leng']
                p = (4 * np.pi * a) / (l ** 2.)
                f['properties']['popper'] = p
                if p > popper and acres > min_acres:
                    out_features.append(f)

    print('{} features of {} in {}'.format(len(out_features), ct, shapes))

    with fiona.open(out_shape, 'w', **meta) as output:
        ct = 0
        for feat in out_features:
            if not feat['geometry']:
                inval_ct += 1
            elif not shape(feat['geometry']).is_valid:
                inval_ct += 1
            else:
                output.write(feat)
                ct += 1

    print('wrote {} features, {} invalid, to {}'.format(ct, inval_ct, out_shape))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    alt_home = '/media/research'
    if os.path.isdir(alt_home):
        home = alt_home
    else:
        home = os.path.join(home, 'data')

    states = irrmapper_states + east_states
    pad = os.path.join(home, 'IrrigationGIS', 'training_data', 'uncultivated', 'USGS_PAD')
    out = os.path.join(pad, 'pop')
    cdl = os.path.join(home, 'IrrigationGIS', 'cdl', 'crop_mask')
    for s in states:
        try:
            raster = os.path.join(cdl, 'CMASK_2019_{}.tif'.format(s))
            shape_ = os.path.join(pad, 'singlepart_nodupes',
                                  'PADUS2_0Combined_DOD_Fee_Designation_Easement_{}.shp'.format(s))
            popper = os.path.join(pad, 'popper', 'PAD_{}.shp'.format(s))
            process_pad(shape_, popper)
            exit()
            # cdl_attrs = os.path.join('/home/dgketchum/Downloads/cdl_popper', 'PAD_{}.shp'.format(s))
            # zonal_crop_mask(popper, raster, cdl_attrs)
        except Exception as e:
            print(s, e)
            pass
# ========================= EOF ====================================================================
