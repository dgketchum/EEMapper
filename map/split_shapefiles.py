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
from datetime import datetime
from subprocess import check_call

import fiona
from numpy import arange
from numpy.random import shuffle


# WETLAND_SHAPEFILES = [
#     # 'MT_shapefile_wetlands/MT_Wetlands_West.shp',
#     # 'MT_shapefile_wetlands/MT_Wetlands_East.shp',
#     # 'AZ_shapefile_wetlands/AZ_Wetlands.shp',
#     # 'CA_shapefile_wetlands/CA_Wetlands_North.shp',
#     # 'CA_shapefile_wetlands/CA_Wetlands_NorthCentral.shp',
#     # 'CA_shapefile_wetlands/CA_Wetlands_SouthCentral.shp',
#     'CA_shapefile_wetlands/CA_Wetlands_South.shp',
#     # 'CO_shapefile_wetlands/CO_Wetlands_West.shp',
#     # 'CO_shapefile_wetlands/CO_Wetlands_East.shp',
#     # 'ID_shapefile_wetlands/ID_Wetlands.shp',
#     # 'NM_shapefile_wetlands/NM_Wetlands.shp',
#     # 'NV_shapefile_wetlands/NV_Wetlands_North.shp',
#     # 'NV_shapefile_wetlands/NV_Wetlands_South.shp',
#     # 'OR_shapefile_wetlands/OR_Wetlands_East.shp',
#     # 'OR_shapefile_wetlands/OR_Wetlands_West.shp',
#     # 'UT_shapefile_wetlands/UT_Wetlands.shp',
#     # 'WA_shapefile_wetlands/WA_Wetlands_West.shp',
#     # 'WA_shapefile_wetlands/WA_Wetlands_East.shp',
#     # 'WY_shapefile_wetlands/WY_Wetlands_West.shp',
#     # 'WY_shapefile_wetlands/WY_Wetlands_East.shp',
# ]
#
# ID_ESPA = [('ID_1986_ESPA_WGS84.shp', 'STATUS_198'),
#            ('ID_1996_ESPA_WGS84.shp', 'STATUS_199'),
#            ('ID_2002_ESPA_WGS84.shp', 'STATUS_200'),
#            ('ID_2006_ESPA_WGS84.shp', 'STATUS_200'),
#            ('ID_2008_ESPA_WGS84.shp', 'STATUS_200'),
#            ('ID_2009_ESPA_WGS84.shp', 'STATUS_200'),
#            ('ID_2010_ESPA_WGS84.shp', 'STATUS_201'),
#            ('ID_2011_ESPA_WGS84.shp', 'STATUS_201')]
#
# OPEN_WATER = [
#     #     'MT_Wetlands_Eastopen_water.shp',
#     #               'WA_Wetlands_Westopen_water.shp',
#     #               'CA_Wetlands_NorthCentralopen_water.shp',
#     #               'CA_Wetlands_SouthCentralopen_water.shp',
#     'CA_Wetlands_Southopen_water.shp',
#     #               'WY_Wetlands_Eastopen_water.shp',
#     #               'OR_Wetlands_Eastopen_water.shp',
#     #               'NM_Wetlandsopen_water.shp',
#     #               'CO_Wetlands_Westopen_water.shp',
#     #               'ID_Wetlandsopen_water.shp',
#     #               'AZ_Wetlandsopen_water.shp',
#     #               'CO_Wetlands_Eastopen_water.shp',
#     #               'MT_Wetlands_Westopen_water.shp',
#     #               'WA_Wetlands_Eastopen_water.shp',
#     #               'NV_Wetlands_Southopen_water.shp',
#     #               'OR_Wetlands_Westopen_water.shp',
#     #               'CA_Wetlands_Northopen_water.shp',
#     #               'WY_Wetlands_Westopen_water.shp',
#     #               'UT_Wetlandsopen_water.shp',
#     #               'NV_Wetlands_Northopen_water.shp'
# ]
#
# WETLAND = [
#     # 'MT_Wetlands_Eastwetlands.shp',
#     #        'WA_Wetlands_Westwetlands.shp',
#     #        'CA_Wetlands_NorthCentralwetlands.shp',
#     #        'CA_Wetlands_SouthCentralwetlands.shp',
#     'CA_Wetlands_Southwetlands.shp',
#     #        'WY_Wetlands_Eastwetlands.shp',
#     #        'OR_Wetlands_Eastwetlands.shp',
#     #        'NM_Wetlandswetlands.shp',
#     #        'CO_Wetlands_Westwetlands.shp',
#     #        'ID_Wetlandswetlands.shp',
#     #        'AZ_Wetlandswetlands.shp',
#     #        'CO_Wetlands_Eastwetlands.shp',
#     #        'MT_Wetlands_Westwetlands.shp',
#     #        'WA_Wetlands_Eastwetlands.shp',
#     #        'NV_Wetlands_Southwetlands.shp',
#     #        'OR_Wetlands_Westwetlands.shp',
#     #        'CA_Wetlands_Northwetlands.shp',
#     #        'WY_Wetlands_Westwetlands.shp',
#     #        'UT_Wetlandswetlands.shp',
#     #        'NV_Wetlands_Northwetlands.shp'
# ]


def split_wetlands(in_shp, out):
    surface = []
    surface_parameters = ['Riverine', 'Lake', 'Freshwater Pond']
    wetland = []
    wetland_parameters = ['Freshwater Emergent Wetland', 'Freshwater Forested/Shrub Wetland']

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            if feat['properties']['WETLAND_TY'] in wetland_parameters:
                wetland.append(feat)
            if feat['properties']['WETLAND_TY'] in surface_parameters:
                surface.append(feat)

    for _type in [('open_water', surface), ('wetlands', wetland)]:
        s = os.path.basename(in_shp)
        name = s.replace('.shp', '{}.shp'.format(_type[0]))
        out_file = os.path.join(out, name)
        l = _type[1]
        print(out_file)
        with fiona.open(out_file, 'w', **meta) as output:
            for feat in l:
                output.write(feat)

    return None


def split_idaho(in_shp, prop='STATUS_201'):
    irr = []
    non_irr = []

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            if feat['properties'][prop] == 'irrigated':
                irr.append(feat)
            if feat['properties'][prop] == 'non-irrigated':
                non_irr.append(feat)

    for _type in [('irr', irr), ('non-irrigated', non_irr)]:
        print(_type[0])
        name = in_shp.replace('.shp', '_{}.shp'.format(_type[0]))
        l = _type[1]
        with fiona.open(name, 'w', **meta) as output:
            for feat in l:
                output.write(feat)


def split_utah(in_shp, out_dir):
    years = {}
    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            s = feat['properties']['SURV_YEAR']
            if s not in years.keys():
                years[s] = [feat]
            else:
                years[s].append(feat)

    for k, v in years.items():
        shp = os.path.join(out_dir, 'UT_UnIrr_WGS84_{}.shp'.format(k))
        with fiona.open(shp, 'w', **meta) as dst:
            for feat in v:
                dst.write(feat)


def split_ucrb(in_shp, out_dir):
    years = {}
    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            s = feat['properties']['Source_yea']
            if s not in years.keys():
                years[s] = [feat]
            else:
                years[s].append(feat)

    for k, v in years.items():
        shp = os.path.join(out_dir, 'UCRB_UnIrr_WGS84_{}.shp'.format(k))
        with fiona.open(shp, 'w', **meta) as dst:
            for feat in v:
                dst.write(feat)


def split_washington_irrigated(master):
    years = {}
    with fiona.open(master, 'r') as src:
        meta = src.meta
        for feat in src:
            s = datetime.strptime(feat['properties']['InitialSur'], '%Y/%m/%d %H:%M:%S.%f').year
            e = datetime.strptime(feat['properties']['LastSurvey'], '%Y/%m/%d %H:%M:%S.%f').year

            if s not in years.keys():
                years[s] = [feat]
            else:
                years[s].append(feat)
            if e not in years.keys():
                years[e] = [feat]
            else:
                years[e].append(feat)
    dirname = os.path.dirname(master)
    for k, v in years.items():
        shp = os.path.join(dirname, 'WA_NonIrr_WGS84_{}.shp'.format(k))
        with fiona.open(shp, 'w', **meta) as dst:
            for feat in v:
                dst.write(feat)


def reduce_shapefiles(root, outdir, n, shapefiles):
    for s in shapefiles:
        shp = os.path.join(root, s)
        dst_file = os.path.join(outdir, s.replace('Forest_Practices_Applications',
                                                  'WA_Forest_WGS84'.format(n)))
        if os.path.isfile(dst_file):
            print(dst_file, 'exists')
        else:
            with fiona.open(shp, 'r') as src:
                count = len([x for x in src])
                meta = src.meta
            idx = arange(0, count)
            shuffle(idx)
            out_features = []
            with fiona.open(shp, 'r') as src:
                i = 0
                for feat in src:
                    if i in idx[:n]:
                        out_features.append(feat)
                    i += 1
            print(dst_file)
            with fiona.open(dst_file, 'w', **meta) as dst:
                for feat in out_features:
                    dst.write(feat)


def batch_reproject_vector(ogr_path, in_dir, out_dir, name_append, t_srs, s_srs):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp')]
    for s in l:
        name_in = os.path.basename(s)
        name_out = name_in.replace('.shp', '_{}.shp'.format(name_append))
        out_shp = os.path.join(out_dir, name_out)
        cmd = ['{}'.format(ogr_path), '{}'.format(out_shp), '{}'.format(s),
               '-t_srs', 'EPSG:{}'.format(t_srs), '-s_srs', 'EPSG:{}'.format(s_srs)]
        check_call(cmd)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    irr = os.path.join(home, 'IrrigationGIS', 'training_raw', 'UCRB')
    out = os.path.join(home, 'IrrigationGIS', 'EE_sample', 'unirrigated_ag', 'UCRB')
    shp = os.path.join(irr, 'UCRB_UnIrrigated_WGS84.shp')
    split_ucrb(shp, out)
# ========================= EOF ====================================================================
