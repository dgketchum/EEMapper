import os

import fiona
from numpy import linspace, max
from numpy.random import shuffle, choice
from pandas import DataFrame
from shapely.geometry import shape, Point, mapping, Polygon
from shapely.errors import TopologicalError

from call_ee import TARGET_STATES, E_STATES

YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019]


class PointsRunspec(object):

    def __init__(self, root, buffer, **kwargs):
        self.root = root
        self.features = []
        self.object_id = 0
        self.year = None
        self.crs = None
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE', 'YEAR'])

        self.buffer = buffer

        self.years = None

        self.irr_path = IRRIGATED
        self.unirr_path = UNIRRIGATED
        self.uncult_path = UNCULTIVATED
        self.wetland_path = WETLAND
        self.fallow_path = FALLOW

        if 'intersect' in kwargs.keys():
            self.intersect = kwargs['intersect']
        if 'irrigated' in kwargs.keys():
            self.irrigated(kwargs['irrigated'])
        if 'fallowed' in kwargs.keys():
            self.fallowed(kwargs['fallowed'])
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
        self.create_sample_points(n, self.irr_path, code=0, attribute='YEAR', set_years=True)

    def fallowed(self, n):
        print('fallow: {}'.format(n))
        self.create_sample_points(n, self.fallow_path, code=4, attribute='YEAR')

    def create_sample_points(self, n, shp, code, attribute=None, set_years=False):

        instance_ct = 0
        polygons = self._get_polygons(shp, attr=attribute)
        shuffle(polygons)
        if attribute:
            years, polygons = [x[1] for x in polygons], [x[0] for x in polygons]
            if set_years:
                self.years = years

        positive_area = sum([x.area for x in polygons])
        print('area: {} in {} features'.format(positive_area / 1e6, len(polygons)))
        bad_polygons = 0
        for i, poly in enumerate(polygons):
            try:
                if attribute:
                    self.year = years[i]
                elif self.years:
                    self.year = choice(self.years)
                else:
                    self.year = choice(YEARS)

                # too much data in 2013, only extract irrigated and fallow
                if self.year == 2013 and code in [1, 2, 3]:
                    continue

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
            except Exception as e:
                print(e)
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
        print('write {}'.format(path))
        return None

    def _get_polygons(self, vector, attr=None):
        if self.intersect:
            with fiona.open(self.intersect, 'r') as inter_f:
                inter_geo = shape([f['geometry'] for f in inter_f][0])
        with fiona.open(vector, 'r') as src:
            polys = []
            bad_geo_count = 0
            for feat in src:
                try:
                    geo = shape(feat['geometry'])

                    try:
                        if self.intersect and not inter_geo.intersects(geo):
                            continue
                    except TopologicalError:
                        continue

                    if attr:
                        attribute = feat['properties'][attr]
                        polys.append((geo, attribute))
                    else:
                        polys.append(geo)
                except AttributeError:
                    bad_geo_count += 1

        return polys


def get_training_years(shapes):
    years = []
    for shape in shapes:
        with fiona.open(shape, 'r') as src:
            for feat in src:
                yr = feat['properties']['YEAR']
                if yr not in years:
                    years.append(yr)
    return years


if __name__ == '__main__':
    # home = os.path.expanduser('~')
    home = '/media/research/IrrigationGIS'
    data = os.path.join(home, 'EE_sample', 'aea')

    FALLOW = os.path.join(data, 'fallow_5NOV2021.shp')
    IRRIGATED = os.path.join(data, 'irrigated_5NOV2021.shp')
    UNCULTIVATED = os.path.join(data, 'uncultivated_11JAN2021.shp')
    UNIRRIGATED = os.path.join(data, 'dryland_11JAN2021.shp')
    WETLAND = os.path.join(data, 'wetlands_11JAN2021.shp')

    for state in E_STATES[-1:]:
        print('Dist Points ', state)
        intersect_shape = '/media/research/IrrigationGIS/boundaries/states_tiger_aea/{}.shp'.format(state)

        kwargs = {
            'irrigated': 1500,
            'wetlands': 1500,
            'fallowed': 1500,
            'uncultivated': 1500,
            'unirrigated': 1500,
            'intersect': intersect_shape,
        }
        out_name = os.path.join(home, 'EE_extracts', 'point_shp', 'state_aea', 'points_{}_5NOV2021.shp'.format(state))
        prs = PointsRunspec(data, buffer=-20, **kwargs)
        prs.save_sample_points(out_name)

# ========================= EOF ====================================================================
