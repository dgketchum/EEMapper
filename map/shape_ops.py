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
from shapely.geometry import shape

CLU_UNNEEDED = ['ca', 'nv', 'ut', 'wa', 'wy']
CLU_USEFUL = ['az', 'co', 'id', 'mt', 'nm', 'or']
CLU_ONLY = ['ne', 'ks', 'nd', 'ok', 'sd', 'tx']

irrmapper_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

SHAPE_COMPILATION = {
    # 'AZ': (
    #     ('UCRBGS', '/home/dgketchum/IrrigationGIS/openET/AZ/az_ucrb.shp'),
    #     ('LCRVPD', '/home/dgketchum/IrrigationGIS/openET/AZ/az_lcrb.shp'),
    #     ('CLU', '/home/dgketchum/IrrigationGIS/clu/crop_vector_v2_wgs/az_cropped_v2_wgs.shp')),
    'CO': (
        ('UCRBGS', '/home/dgketchum/IrrigationGIS/openET/CO/co_ucrb_add.shp'),
        ('CODWR', '/home/dgketchum/IrrigationGIS/raw_field_polygons/CO/CO_irrigated_latest_wgs.shp'),
        ('CLU', '/home/dgketchum/IrrigationGIS/clu/crop_vector_v2_wgs/co_cropped_v2_wgs.shp')),
    # 'WY': (
    #     ('UCRBGS', '/home/dgketchum/IrrigationGIS/openET/WY/WY_USGS_UCRB.shp'),
    #     ('BRC', '/home/dgketchum/IrrigationGIS/openET/WY/WY_BRC.shp'),
    #     ('WYSWP', '/home/dgketchum/IrrigationGIS/raw_field_polygons/WY/Irrigated_Land/Irrigated_Land_wgs.shp')),

}


def fiona_merge_attribute(out_shp, file_list):
    """ Use to merge and keep the year attribute """
    years = []
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('YEAR', 'int:9'), ('SOURCE', 'str:80')]), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        for s in file_list:
            print(s)
            year, source = int(s.split('.')[0][-4:]), os.path.basename(s.split('.')[0][:-5])
            if year not in years:
                years.append(year)
            for feat in fiona.open(s):
                feat = {'type': 'Feature', 'properties': {'SOURCE': source, 'YEAR': year},
                        'geometry': feat['geometry']}
                output.write(feat)
                ct += 1
        print(sorted(years))


def fiona_merge_no_attribute(out_shp, file_list, clean=False):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        []), 'geometry': 'Polygon'}
    with fiona.open(out_shp, 'w', **meta) as output:
        ct = 0
        none_geo = 0
        inval_geo = 0
        for s in file_list:
            mgrs = s.split('/')[-1].strip('.shp')
            print(mgrs)
            for feat in fiona.open(s):

                if not feat['geometry']:
                    none_geo += 1
                    continue
                geo = shape(feat['geometry'])
                if not geo.is_valid:
                    inval_geo += 1
                    continue
                if geo.area == 0.0:
                    continue
                if geo.bounds[0] < -180.0:
                    inval_geo += 1
                    print(s)
                    continue
                ct += 1
                feat = {'type': 'Feature', 'properties': {},
                        'geometry': feat['geometry']}
                output.write(feat)

    print('wrote {}, {} none, {} invalid'.format(ct, none_geo, inval_geo))


if __name__ == '__main__':
    # home = os.path.expanduser('~')
    home = '/media/research/'
    training = os.path.join(home, 'IrrigationGIS', 'training_data')
    class_ = os.path.join(training, 'unirrigated', 'to_merge')
    files_ = [os.path.join(class_, x) for x in os.listdir(class_) if '.shp' in x]
    local = os.path.join(os.path.expanduser('~'), 'IrrigationGIS', 'EE_sample', 'dryland_5AUG2020.shp')
    fiona_merge_no_attribute(local, files_)
# ========================= EOF ====================================================================
