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
from datetime import datetime
from subprocess import check_call

import fiona
from numpy import arange
from numpy.random import shuffle


def extract_fallow_montana(in_shp, out_dir):
    fallow = []
    for year in [2009, 2010, 2011, 2012, 2013]:
        with fiona.open(in_shp, 'r') as src:
            meta = src.meta
            meta['schema']['properties'] = OrderedDict([('FID', 'int:5')])
            count = 0
            for feat in src:
                if feat['properties']['Irr_{}'.format(year)] == 0:
                    new_feature = feat
                    new_feature['properties'] = {}
                    new_feature['properties']['FID'] = count
                    new_feature['id'] = count
                    count += 1
                    fallow.append(new_feature)

        new_shapefile = os.path.join(out_dir, 'MT_Fallow_{}.shp'.format(year))
        with fiona.open(new_shapefile, 'w', **meta) as output:
            for write_feat in fallow:
                output.write(write_feat)


def split_wetlands(in_shp, out):
    wetland = []
    wetland_parameters = ['Freshwater Emergent Wetland', 'Freshwater Forested/Shrub Wetland']

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            if feat['properties']['WETLAND_TY'] in wetland_parameters:
                wetland.append(feat)

    for _type in [('wetlands', wetland)]:
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


def split_nevada(in_dir, out_dir):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('agpoly.shp')]
    for shp in l:
        f = []
        yr = os.path.basename(shp)[:4]
        out_name = os.path.join(out_dir, 'NV_{}_WGS84.shp'.format(yr))
        with fiona.open(shp, 'r') as src:
            meta = src.meta
            for feat in src:
                if feat['properties']['NDVI_MEAN'] > 0.7:
                    f.append(feat)
        with fiona.open(out_name, 'w', **meta) as dst:
            for feat in f:
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


def split_wrri_lulc(in_dir, out_dir):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if
         x.startswith(('Alcalde', 'El_Rito', 'Hondo')) and x.endswith('.shp')]
    years = {}
    for shp in l:
        y = shp.replace('.shp', '')[-4:]
        with fiona.open(shp, 'r') as src:
            meta = src.meta
            for feat in src:
                if feat['properties']['LC_L1'] == 'Irrigated Agriculture':
                    feat = {'type': 'Feature', 'properties': {'OBJECTID': feat['properties']['OBJECTID']},
                            'geometry': feat['geometry']}
                    if y not in years.keys():
                        years[y] = [feat]
                    else:
                        years[y].append(feat)
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('OBJECTID', 'float:20')]), 'geometry': 'Polygon'}
    for k, v in years.items():
        shp = os.path.join(out_dir, 'NM_WRRI_Irr_WGS84_{}.shp'.format(k))
        with fiona.open(shp, 'w', **meta) as dst:
            for feat in v:
                dst.write(feat)


def reduce_shapefiles(root, outdir, n, shapefiles):
    for s in shapefiles:
        shp = os.path.join(root, s)
        dst_file = os.path.join(outdir, s.replace('raw',
                                                  'select_{}'.format(n)))
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


def batch_reproject_vector(ogr_path, in_dir, out_dir, name_append, t_srs):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp')]
    for s in l:
        name_in = os.path.basename(s)
        name_out = name_in.replace('.shp', '_{}.shp'.format(name_append))
        out_shp = os.path.join(out_dir, name_out)
        if not os.path.isfile(out_shp):
            cmd = ['{}'.format(ogr_path), '{}'.format(out_shp), '{}'.format(s),
                   '-t_srs', 'EPSG:{}'.format(t_srs)]
            check_call(cmd)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    # in_ = os.path.join(home, 'IrrigationGIS', 'clu', 'crop_vector')
    # out_ = os.path.join(home, 'IrrigationGIS', 'clu', 'crop_vector_wgs')
    # ogr = os.path.join(home, 'miniconda2', 'envs', 'irri', 'bin', 'ogr2ogr')
    # batch_reproject_vector(ogr, in_, out_, 'wgs', t_srs=4326)
    # /home/dgketchum/IrrigationGIS/training_data/irrigated/AZ/AZ_v2_raw.shp
    shp = os.path.join(home, 'IrrigationGIS', 'training_data', 'irrigated', 'AZ', 'AZ_v2_raw.shp')
    r = os.path.join(home, 'IrrigationGIS', 'training_data', 'irrigated', 'AZ')
    reduce_shapefiles(r, r, 5000, [shp])

# ========================= EOF ====================================================================
