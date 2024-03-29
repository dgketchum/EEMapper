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
from subprocess import check_call

import numpy as np
import fiona
import fiona.crs
from geopandas import GeoDataFrame, read_file, points_from_xy, clip
from pandas import DataFrame, read_csv, concat
from pandas import errors
from shapely.geometry import Polygon, Point, mapping, MultiPolygon, shape

CLU_UNNEEDED = ['ca', 'nv', 'ut', 'wa', 'wy']
CLU_USEFUL = ['az', 'co', 'id', 'mt', 'nm', 'or']
CLU_ONLY = ['ne', 'ks', 'nd', 'ok', 'sd', 'tx']

irrmapper_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

east_states = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

os.environ['GDAL_DATA'] = 'miniconda3/envs/gcs/share/gdal/'

OGR = '/usr/bin/ogr2ogr'


def fiona_merge(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('FID', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    ct, bad_geo = 1, 0
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            sub_ct = 0
            first = True
            src = os.path.basename(s).split('.')[0][:10]
            for feat in fiona.open(s):
                centroid = shape(feat['geometry']).centroid
                if abs(centroid.y) > 50.0:
                    print(centroid)
                    continue
                geo = feat['geometry']
                if geo['type'] != 'Polygon':
                    bad_geo += 1
                    continue
                feat = {'type': 'Feature', 'properties': {'FID': ct, 'SOURCE': src},
                        'geometry': geo}
                output.write(feat)
                ct += 1
                sub_ct += 1
            print('{} features in {}'.format(sub_ct, s))

    print('{} features, {} non-polygons'.format(ct, bad_geo))


def fiona_merge_attribute(out_shp, file_list):
    """ Use to merge and keep the year attribute """
    years = []
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('YEAR', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        for s in file_list:
            if os.path.basename(s.split('.')[0][:2]) in east_states:
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
    print('wrote ', out_shp)


def to_aea(in_shp, out_shp):
    cmd = [OGR, '-f', 'ESRI Shapefile', '-s_srs', WGS, '-t_srs', AEA, out_shp, in_shp]
    check_call(cmd)
    print('projected ', os.path.basename(out_shp))


def to_geographic(in_shp, out_shp):
    cmd = [OGR, '-f', 'ESRI Shapefile', '-t_srs', WGS, '-s_srs', AEA, out_shp, in_shp]
    check_call(cmd)
    print('de-projected ', os.path.basename(out_shp))


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

            if feat['geometry'] is None:
                continue

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
                dct[y] = [0, 0, 0, 0, 0]
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


def join_shp_csv(in_shp, csv_dir, out_shp, join_on='OBJECTID'):
    with fiona.open(in_shp) as src:
        meta = src.meta
        features = [f for f in src]

    csv_l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if x.endswith('.csv')]
    first = True
    for csv in csv_l:
        y = int(os.path.basename(csv).split('_')[3])
        m = int(os.path.basename(csv).split('_')[4].replace('.csv', ''))
        try:
            if first:
                df = read_csv(csv, index_col=join_on)
                df = df.rename(columns={'irr': 'irr_{}'.format(y)})
                print(df.shape, csv)
                first = False
            else:
                c = read_csv(csv, index_col=join_on)
                c = c.rename(columns={'irr': 'irr_{}'.format(y)})
                df = concat([df, c['irr_{}'.format(y)]], axis=1)
                print(c.shape, csv)
        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass

    df['mean'] = np.mean(df.values, axis=1)
    [meta['schema']['properties'].update({col: 'float:19.11'}) for col in df.columns]
    in_feat = len(features)
    ct = 0
    with fiona.open(out_shp, 'w', **meta) as output:
        for feat in features:
            try:
                _id = feat['properties'][join_on]
                feat['properties'].update(df.loc[_id])
                output.write(feat)
                ct += 1
            except Exception as e:
                print(feat['properties'][join_on], e)
    print('{} of {} features joined'.format(ct, in_feat))


def popper_test(shp, out_shp, threshold=0.79,
                min_area=325000., min_thresh=0.78):
    meta = fiona.open(shp).meta
    meta['schema']['properties']['popper'] = 'float'

    def popper(geometry):
        p = (4 * np.pi * geometry.area) / (geometry.boundary.length ** 2.)
        return p

    features = []
    ct = 0
    non_polygon = 0
    popper_ct = 0
    with fiona.open(shp, 'r') as src:
        for feat in src:
            ct += 1
            geo = shape(feat['geometry'])
            if not isinstance(geo, Polygon):
                print(type(geo))
                non_polygon += 1
                continue
            popper_ = float(popper(geo))
            if threshold > popper_ > min_thresh:
                feat['properties']['popper'] = popper_
                popper_ct += 1
                if popper_ct % 1000 == 0:
                    print('{} potential polygons of {}, {} non-polygon geometries'.format(popper_ct, ct, non_polygon))
                features.append(feat)
    with fiona.open(out_shp, 'w', **meta) as dst:
        write_ct = 0
        for i, feat in enumerate(features):
            try:
                dst.write(feat)
                write_ct += 1
            except:
                pass
    print('{} passing objects, {} written, {}'.format(popper_ct, write_ct, out_shp))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    csv_ = '/home/dgketchum/Downloads/missoula_irrigation.csv'
    shp_ = '/media/research/IrrigationGIS/Montana/geointernship/Missoula_063/flu_missoula_co_2019.shp'
    shp_out = '/home/dgketchum/Downloads/missoula_irrigation_11MAY2023.shp'
    join_shp_csv(shp_, csv_, out_shp=shp_out, join_on='FID')
# ========================= EOF ====================================================================
