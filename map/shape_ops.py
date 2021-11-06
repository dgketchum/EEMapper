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
from random import shuffle
from collections import OrderedDict

import fiona
from geopandas import GeoDataFrame, read_file, points_from_xy, clip
from pandas import DataFrame, read_csv, concat
# from rasterstats import zonal_stats
from shapely.geometry import Polygon, Point, mapping, MultiPolygon, shape

CLU_UNNEEDED = ['ca', 'nv', 'ut', 'wa', 'wy']
CLU_USEFUL = ['az', 'co', 'id', 'mt', 'nm', 'or']
CLU_ONLY = ['ne', 'ks', 'nd', 'ok', 'sd', 'tx']

irrmapper_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

east_states = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']


def fiona_merge(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('FID', 'int:9')]), 'geometry': 'Polygon'}
    ct = 1
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            sub_ct = 0
            first = True
            for feat in fiona.open(s):
                # centroid = shape(feat['geometry']).centroid
                # if abs(centroid.y) > 50.0:
                #     print(centroid)
                #     continue
                feat = {'type': 'Feature', 'properties': {'FID': ct},
                        'geometry': feat['geometry']}
                output.write(feat)
                ct += 1
                sub_ct += 1
            print('{} features in {}'.format(sub_ct, s))

    print('{} features'.format(ct))
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
            if  os.path.basename(s.split('.')[0][:2]) in east_states:
                pass
            else:
                year, source = int(s.split('.')[0][-4:]), os.path.basename(s.split('.')[0][:-5])
                print(year, s)
                if year not in years:
                    years.append(year)
                for feat in fiona.open(s):
                    feat = {'type': 'Feature', 'properties': {'SOURCE': source, 'YEAR': year},
                            'geometry': feat['geometry']}
                    output.write(feat)
                    ct += 1
        print(sorted(years))


def get_area(shp, intersect_shape=None, add_duplicate_area=True):
    """use AEA conical for sq km result"""

    if intersect_shape:
        with fiona.open(intersect_shape, 'r') as inter:
            poly = [c for c in inter][0]['geometry']
            try:
                polys = [Polygon(p[0]) for p in poly['coordinates']]
                areas = [p.area for p in polys]
                shape = polys[areas.index(max(areas))]
            except (ValueError, TypeError):
                try:
                    shape = Polygon(poly['coordinates'][0])
                except ValueError:
                    print(intersect_shape, 'failed')
                    return None

    with fiona.open(shp, 'r') as s:
        area = 0.0
        geos = []
        dupes = 0
        unq = 0
        ct = 0
        for feat in s:
            if feat['geometry']['type'] == 'Polygon':
                coords = feat['geometry']['coordinates'][0]

                if 'irrigated' in shp and coords not in geos or 'irrigated' not in shp:
                    geos.append(coords)
                    p = Polygon(coords)
                    if intersect_shape:
                        if p.intersects(shape):
                            a = p.area / 1e6
                            area += a
                            ct += 1
                            unq += 1
                elif 'irrigated' in shp and coords in geos:
                    dupes += 1
                    if add_duplicate_area:
                        geos.append(coords)
                        p = Polygon(coords)
                        if intersect_shape:
                            if p.intersects(shape):
                                a = p.area / 1e6
                                area += a
                                ct += 1
                else:
                    raise TypeError

            elif feat['geometry']['type'] == 'MultiPolygon':
                for l_ring in feat['geometry']['coordinates']:
                    coords = l_ring[0]
                    if 'irrigated' in shp and coords not in geos or 'irrigated' not in shp:
                        geos.append(coords)
                        p = Polygon(coords)
                        if intersect_shape:
                            if not p.intersects(shape):
                                break
                        a = p.area / 1e6
                        area += a
                        unq += 1
                        ct += 1
                    elif 'irrigated' in shp and coords in geos:
                        dupes += 1
                        if add_duplicate_area:
                            geos.append(coords)
                            p = Polygon(coords)
                            if intersect_shape:
                                if not p.intersects(shape):
                                    break
                            a = p.area / 1e6
                            area += a
                            ct += 1
                    else:
                        raise TypeError
            else:
                raise TypeError

        if intersect_shape:
            print(ct, unq, area, os.path.basename(shp), os.path.basename(intersect_shape))
        else:
            print(area)


def band_extract_to_shp(table, out_shp):
    df = read_csv(table)
    drops = [x for x in df.columns if x not in ['LAT_GCS', 'Lon_GCS', 'POINT_TYPE']]
    df.drop(columns=drops, inplace=True)
    geo = [Point(x, y) for x, y in zip(df['Lon_GCS'].values, df['LAT_GCS'].values)]
    gpd = GeoDataFrame(df['POINT_TYPE'], crs={'init': 'epsg:4326'}, geometry=geo)
    gpd.to_file(out_shp)


def clip_bands_to_polygon(bands, out_bands, mask):
    with fiona.open(mask, 'r') as src:
        feat = [f for f in src]
    bounds = shape(feat[0]['geometry'])
    df = read_csv(bands)
    gdf = GeoDataFrame(df, geometry=points_from_xy(y=df['LAT_GCS'], x=df['Lon_GCS']))
    gdf = clip(gdf, mask=bounds)
    df = DataFrame(gdf.drop(columns='geometry'))
    df.to_csv(out_bands)


def count_points(shp):
    with fiona.open(shp, 'r') as src:
        dct = {}
        for f in src:
            y = f['properties']['YEAR']
            t = f['properties']['POINT_TYPE']
            if y not in dct.keys():
                dct[y] = [0,0,0,0,0]
            dct[y][t] += 1
    for k, v in dct.items():
        print(k, v, sum(v))
    return list(set(dct.keys()))


def subselect_points_shapefile(shp, out_shp, limit=10000):
    with fiona.open(shp, 'r') as src:
        meta = src.meta
        features = [f for f in src]
        shuffle(features)

    dct = {}
    out_features = []
    for f in features:
        y = f['properties']['YEAR']
        if y not in dct.keys():
            dct[y] = 0
        if dct[y] < limit:
            dct[y] += 1
            out_features.append(f)
    for k, v in dct.items():
        print(k, v)

    with fiona.open(out_shp, 'w', **meta) as dst:
        for f in out_features:
            dst.write(f)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    gis = os.path.join('/media/research', 'IrrigationGIS')
    inspected = os.path.join(gis, 'training_data', 'fallow', 'inspected')
    files_ = [os.path.join(inspected, x) for x in os.listdir(inspected) if x.endswith('.shp')]
    out_ = os.path.join(gis, 'EE_sample', 'wgs', 'dryland_11NOV2021.shp')
    fiona_merge_attribute(out_, files_)
# ========================= EOF ====================================================================
