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
from subprocess import check_call


def convert_kml_to_shp(ogr_path, in_dir, out_dir, t_srs, s_srs, skip_existing=False):
    l = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.kml')]
    l.sort()
    for s in l:
        try:
            name_in = os.path.basename(s)
            name_out = name_in.replace('ee_export.kml', '.shp')
            out_shp = os.path.join(out_dir, name_out)
            if not os.path.exists(out_shp):
                cmd = ['{}'.format(ogr_path), '{}'.format(out_shp), '{}'.format(s),
                       '-t_srs', 'EPSG:{}'.format(t_srs), '-s_srs', 'EPSG:{}'.format(s_srs),
                       '-skipfailures', '-nlt', 'geometry']
                check_call(cmd)
            elif not skip_existing:
                cmd = ['{}'.format(ogr_path), '{}'.format(out_shp), '{}'.format(s),
                       '-t_srs', 'EPSG:{}'.format(t_srs), '-s_srs', 'EPSG:{}'.format(s_srs),
                       '-skipfailures', '-nlt', 'geometry']
                check_call(cmd)
            else:
                pass
        except:
            print(s, 'failed')


def rename(_dir, glob, find, replace):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if glob in x]
    for _file in l:
        # print(_file, _file.replace(find, replace))
        os.rename(_file, _file.replace(find, replace))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    irr = os.path.join(home, 'IrrigationGIS', 'training_data', 'irrigated')
    _in = os.path.join(irr, 'filtered_kml')
    out = os.path.join(irr, 'filtered_shapefiles')
    ogr = os.path.join(home, 'miniconda2', 'envs', 'irri', 'bin', 'ogr2ogr')
    convert_kml_to_shp(ogr, _in, out, '4326', '4326', skip_existing=True)
# ========================= EOF ====================================================================
