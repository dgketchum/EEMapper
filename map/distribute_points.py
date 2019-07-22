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

import fiona
from numpy import linspace, max
from numpy.random import shuffle, choice
from pandas import DataFrame
from shapely.geometry import shape, Point, mapping

from map.call_ee import YEARS

training = os.path.join(os.path.expanduser('~'), 'IrrigationGIS', 'EE_sample')

WETLAND = os.path.join(training, 'wetlands_8NOV.shp')
UNCULTIVATED = os.path.join(training, 'uncultivated_4APR.shp')
IRRIGATED = os.path.join(training, 'irrigated_15JUL.shp')
UNIRRIGATED = os.path.join(training, 'unirrigated_9JUL.shp')
FALLOW = os.path.join(training, 'fallow_11FEB.shp')


class PointsRunspec(object):

    def __init__(self, root, buffer, **kwargs):
        self.root = root
        self.features = []
        self.object_id = 0
        self.year = None
        self.crs = None
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE', 'YEAR'])

        self.buffer = buffer

        self.irr_path = IRRIGATED
        self.unirr_path = UNIRRIGATED
        self.uncult_path = UNCULTIVATED
        self.wetland_path = WETLAND
        self.fallow_path = FALLOW

        if 'fallowed' in kwargs.keys():
            self.fallowed(kwargs['fallowed'])
        if 'irrigated' in kwargs.keys():
            self.irrigated(kwargs['irrigated'])
        if 'unirrigated' in kwargs.keys():
            self.unirrigated(kwargs['unirrigated'])
        if 'wetlands' in kwargs.keys():
            self.wetlands(kwargs['wetlands'])
        if 'uncultivated' in kwargs.keys():
            self.uncultivated(kwargs['uncultivated'])

    def wetlands(self, n):
        print('wetlands: {}'.format(n))
        self.create_sample_points(n, self.wetland_path, code=3)

    def uncultivated(self, n):
        print('uncultivated: {}'.format(n))
        self.create_sample_points(n, self.uncult_path, code=2)

    def unirrigated(self, n):
        print('unirrigated: {}'.format(n))
        self.create_sample_points(n, self.unirr_path, code=1)

    def irrigated(self, n):
        print('irrigated: {}'.format(n))
        self.create_sample_points(n, self.irr_path, code=0, attribute='YEAR')

    # def fallowed(self, n):
    #     print('fallow: {}'.format(n))
    #     self.create_sample_points(n, self.fallow_path, code=4, attribute='YEAR')

    def create_sample_points(self, n, shp, code, attribute=None):

        instance_ct = 0
        polygons = self._get_polygons(shp, attr=attribute)
        shuffle(polygons)
        if attribute:
            years, polygons = [x[1] for x in polygons], [x[0] for x in polygons]

        positive_area = sum([x.area for x in polygons])
        print('area: {} in {} features'.format(positive_area / 1e6, len(polygons)))
        bad_polygons = 0
        for i, poly in enumerate(polygons):

            if attribute:
                self.year = years[i]
            else:
                self.year = choice(YEARS)

            if self.buffer:
                buf_poly = poly.buffer(self.buffer, resolution=128)
            else:
                buf_poly = poly

            fractional_area = poly.area / positive_area
            required_points = max([1, fractional_area * n])

            try:
                x_range, y_range = self._random_points(buf_poly.bounds, n)
            except IndexError:
                x_range, y_range = self._random_points(poly.bounds, n)

            poly_pt_ct = 0
            for coord in zip(x_range, y_range):
                try:
                    if Point(coord[0], coord[1]).within(poly):
                        self._add_entry(coord, val=code)
                        poly_pt_ct += 1
                        instance_ct += 1
                except Exception as e:
                    print(poly)
                    print(e)
                    break

                if poly_pt_ct >= required_points:
                    break
            if instance_ct > n:
                break
        print('bad class {} polygons: {}'.format(code, bad_polygons))

    @staticmethod
    def _random_points(coords, n):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * n)
        y_range = linspace(min_y, max_y, num=2 * n)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):

        self.extracted_points = self.extracted_points.append({'FID': int(self.object_id),
                                                              'X': coord[0],
                                                              'Y': coord[1],
                                                              'POINT_TYPE': val,
                                                              'YEAR': int(self.year)},
                                                             ignore_index=True)
        self.object_id += 1

    def save_sample_points(self, path):

        points_schema = {
            'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10'), ('YEAR', 'int:10')]),
            'geometry': 'Point'}
        meta = {'driver': 'ESRI Shapefile', 'schema': points_schema, 'crs': self.crs}

        with fiona.open(path, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']),
                              ('POINT_TYPE', row['POINT_TYPE']),
                              ('YEAR', row['YEAR'])])

                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})
        return None

    def _get_polygons(self, vector, attr=None):
        with fiona.open(vector, 'r') as src:
            if not self.crs:
                self.crs = src.crs
            else:
                assert src.crs == self.crs
            polys = []
            bad_geo_count = 0
            for feat in src:
                try:
                    geo = shape(feat['geometry'])
                    if attr:
                        attribute = feat['properties'][attr]
                        polys.append((geo, attribute))
                    else:
                        polys.append(geo)
                except AttributeError:
                    bad_geo_count += 1

        return polys


if __name__ == '__main__':
    home = os.path.expanduser('~')
    data = os.path.join(home, 'IrrigationGIS', 'EE_sample')
    extract = os.path.join(home, 'IrrigationGIS', 'EE_extracts', 'point_shp')

    kwargs = {
        'irrigated': 40000,
        'wetlands': 60000,
        'uncultivated': 40000,
        'unirrigated': 40000,
    }

    prs = PointsRunspec(data, buffer=-20, **kwargs)
    prs.save_sample_points(os.path.join(extract, 'points_17JUL_val.shp'.format()))

# ========================= EOF ====================================================================
