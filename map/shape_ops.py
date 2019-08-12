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
from collections import OrderedDict

import fiona
from geopandas import GeoDataFrame, read_file
from pandas import DataFrame, read_csv, concat
from rasterstats import zonal_stats
from shapely.geometry import Polygon, Point

CLU_UNNEEDED = ['ca', 'nv', 'ut', 'wa']
CLU_USEFUL = ['ne']  # ['az', 'co', 'id', 'mt', 'nm', 'or']
CLU_ONLY = ['ks', 'nd', 'ok', 'sd', 'tx']


def fiona_merge_MT(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'properties': OrderedDict(
        [('Irr_2009', 'int:5'), ('Irr_2010', 'int:5'), ('Irr_2011', 'int:5'),
         ('Irr_2012', 'int:5'), ('Irr_2013', 'int:5')]),
        'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for features in fiona.open(s):
                try:
                    feat = {'properties': OrderedDict([('Irr_2009', int(features['properties']['Irr_2009'])),
                                                       ('Irr_2010', int(features['properties']['Irr_2010'])),
                                                       ('Irr_2011', int(features['properties']['Irr_2011'])),
                                                       ('Irr_2012', int(features['properties']['Irr_2012'])),
                                                       ('Irr_2013', int(features['properties']['Irr_2013']))]),
                            'geometry': features['geometry']}
                    output.write(feat)
                except Exception as e:
                    pass

    return None


def fiona_merge(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for features in fiona.open(s):
                output.write(features)

    return None


def fiona_merge_attribute(out_shp, file_list):
    """ Use to merge and keep the year attribute """
    years = []
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('YEAR', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        for s in file_list:
            year, source = int(s.split('.')[0][-4:]), os.path.basename(s.split('.')[0][:-5])
            if year not in years:
                years.append(year)
            for feat in fiona.open(s):
                feat = {'type': 'Feature', 'properties': {'SOURCE': source, 'YEAR': year},
                        'geometry': feat['geometry']}
                output.write(feat)
                ct += 1
        print(sorted(years))


def fiona_merge_no_attribute(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        []), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for feat in fiona.open(s):
                feat = {'type': 'Feature', 'properties': {},
                        'geometry': feat['geometry']}
                output.write(feat)


def get_list(_dir):
    l = []
    for path, subdirs, files in os.walk(_dir):
        for name in files:
            p = os.path.join(path, name)
            if p not in l and p.endswith('.shp'):
                l.append(p)
    return l


def count_acres(shp):
    acres = 0.0
    with fiona.open(shp, 'r') as s:
        for feat in s:
            a = feat['properties']['ACRES']
            acres += a
        print(acres)


def get_area(shp):
    """use AEA conical for sq km result"""

    print(shp)
    area = 0.0
    geos = []
    dupes = 0
    unq = 0
    with fiona.open(shp, 'r') as s:
        for feat in s:

            if feat['geometry']['type'] == 'Polygon':
                coords = feat['geometry']['coordinates'][0]

                if 'irrigated' in shp and coords not in geos or 'irrigated' not in shp:
                    geos.append(coords)
                    a = Polygon(coords).area / 1e6
                    area += a
                    unq += 1
                elif 'irrigated' in shp and coords in geos:
                    dupes += 1
                else:
                    raise TypeError

            elif feat['geometry']['type'] == 'MultiPolygon':
                for l_ring in feat['geometry']['coordinates']:
                    coords = l_ring[0]
                    if 'irrigated' in shp and coords not in geos or 'irrigated' not in shp:
                        geos.append(coords)
                        a = Polygon(coords).area / 1e6
                        area += a
                        unq += 1
                    elif 'irrigated' in shp and coords in geos:
                        dupes += 1
                    else:
                        raise TypeError
            else:
                raise TypeError
        print(area)
        print(area * 247.105)


def wa_county_acreage(in_shp, out_table):
    counties = {}
    with fiona.open(in_shp, 'r') as src:
        for feat in src:
            co = feat['properties']['County']
            if co not in counties.keys():
                counties[co] = feat['properties']['ExactAcres']
            else:
                counties[co] += feat['properties']['ExactAcres']

    df = DataFrame(counties)
    df.to_csv(out_table)


def clean_geometry(in_shp, out_shp):
    out_list = []
    ct = 0
    meta = fiona.open(in_shp).meta
    for feat in fiona.open(in_shp, 'r'):
        if feat['geometry']['type'] == 'Polygon':
            ct += 1
            out_list.append(feat)
        if feat['geometry']['type'] == 'Multipoint':
            print(feat)
    with fiona.open(out_shp, 'w', **meta) as output:
        for feat in out_list:
            output.write(feat)
    return None


def compile_shapes(in_shapes, out_shape):
    out_features = []
    out_geometries = []
    err = False
    first = True
    err_count = 0
    for _file in in_shapes:
        print(_file)
        if first:
            with fiona.open(_file) as src:
                meta = src.meta
                meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                    [('OBJECTID', 'int:9')]), 'geometry': 'Polygon'}
                raw_features = [x for x in src]
            for f in raw_features:
                try:
                    base_geo = Polygon(f['geometry']['coordinates'][0])
                    out_geometries.append(base_geo)
                    out_features.append(f)
                except Exception as e:
                    try:
                        base_geo = Polygon(f['geometry']['coordinates'][0])
                        out_geometries.append(base_geo)
                        out_features.append(f)
                    except Exception as e:
                        err_count += 1
            print('base geometry errors: {}'.format(err_count))
            first = False
        else:
            # for the following shapefiles:
            f_count = 0
            add_err_count = 0
            for feat in fiona.open(_file):
                inter = False
                f_count += 1

                try:
                    poly = Polygon(feat['geometry']['coordinates'][0])
                except Exception as e:
                    try:
                        poly = Polygon(feat['geometry']['coordinates'][0][0])
                    except Exception as e:
                        add_err_count += 1
                        err = True
                        break
                for _, out_geo in enumerate(out_geometries):
                    if poly.intersects(out_geo):
                        inter = True
                        break
                if not inter and not err:
                    out_features.append(feat)

                if f_count % 10000 == 0:
                    if f_count == 0:
                        pass
                    else:
                        print(f_count, '{} base features'.format(len(out_features)))
            print('added geometry errors: {}'.format(err_count))

    with fiona.open(out_shape, 'w', **meta) as output:
        ct = 0
        for feat in out_features:
            feat = {'type': 'Feature', 'properties': {'OBJECTID': ct},
                    'geometry': feat['geometry']}
            output.write(feat)
            ct += 1


def compile_shapes_nm_wrri(in_shapes, out_shape):
    out_features = []
    err = False
    first = True
    err_count = 0
    for _file in in_shapes:
        print(_file)
        if first:
            with fiona.open(_file) as src:
                meta = src.meta
                meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                    [('OBJECTID', 'int:9')]), 'geometry': 'Polygon'}
                for feat in src:
                    if feat['properties']['LC_L1'] == 'Irrigated Agriculture':
                        out_features.append(feat)
            first = False
        else:
            f_count = 0
            for feat in fiona.open(_file):
                f_count += 1
                inter = False
                if feat['properties']['LC_L1'] == 'Irrigated Agriculture':
                    try:
                        poly = Polygon(feat['geometry']['coordinates'][0])
                    except:
                        err_count += 1
                        err = True
                        break

                    for out_geo in out_features:
                        try:
                            out_poly = Polygon(out_geo['geometry']['coordinates'][0])
                        except:
                            err_count += 1
                            err = True
                            break
                        if poly.intersects(out_poly):
                            inter = True
                            break

                    if not inter and not err:
                        out_features.append(feat)

                    if f_count % 500 == 0:
                        if f_count == 0:
                            pass
                        else:
                            print(f_count, '{} features'.format(len(out_features)))

    with fiona.open(out_shape, 'w', **meta) as output:
        ct = 0
        for feat in out_features:
            feat = {'type': 'Feature', 'properties': {'OBJECTID': ct},
                    'geometry': feat['geometry']}
            output.write(feat)
            ct += 1
    print('errors: {}'.format(err_count))


def zonal_cdl(in_shp, in_raster, out_shp):
    ct = 1
    crops = crop_map()
    geo = []
    with fiona.open(in_shp) as src:
        meta = src.meta
        for feat in src:
            geo.append(feat)

    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('FID', 'int:9'), ('CDL', 'int:9')]), 'geometry': 'Polygon'}
    stats = zonal_stats(in_shp, in_raster, stats=['majority'], nodata=0.0)
    with fiona.open(out_shp, mode='w', **meta) as out:
        for attr, g in zip(stats, geo):
            if attr['majority'] in crops.keys():
                feat = {'type': 'Feature', 'properties': {'FID': ct,
                                                          'CDL': int(attr['majority'])},
                        'geometry': g['geometry']}
                out.write(feat)
                ct += 1


def clean_clu(in_shp, out_shp):
    geo = []
    with fiona.open(in_shp) as src:
        meta = src.meta
        for feat in src:
            geo.append(feat)

    with fiona.open(out_shp, mode='w', **meta) as out:
        for f in geo:
            if f['geometry'] is None:
                print(feat['id'])
            else:
                out.write(f)


def crop_map():
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


def band_extract_to_shp(table, out_shp):
    df = read_csv(table)
    drops = [x for x in df.columns if x not in ['LAT_GCS', 'Lon_GCS', 'POINT_TYPE']]
    df.drop(columns=drops, inplace=True)
    geo = [Point(x, y) for x, y in zip(df['Lon_GCS'].values, df['LAT_GCS'].values)]
    gpd = GeoDataFrame(df['POINT_TYPE'], crs={'init': 'epsg:4326'}, geometry=geo)
    gpd.to_file(out_shp)


def sample_shp(in_shp, out_shp, n):
    gpd = read_file(in_shp)
    sample = [DataFrame(gpd[gpd['POINT_TYPE'] == x]).sample(n=n) for x in [0, 1, 2, 3]]
    df = concat(sample)
    gpd = GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=df['geometry'])
    gpd.to_file(out_shp)
    pass


if __name__ == '__main__':
    home = os.path.expanduser('~')
    table = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'validation_points', 'points_9JUL_validation.shp')
    out = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'validation_points', 'validation_pts_samp40k.shp')
    sample_shp(table, out, n=10000)
# ========================= EOF ====================================================================
