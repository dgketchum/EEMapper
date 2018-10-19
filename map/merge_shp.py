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

OBJECT_MAP = {'MTH': 'Montana',
              'NV': 'Nevada',
              'OR': 'Oregon',
              'UT': 'Utah',
              'WA': 'Washington'}


def fiona_merge(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for features in fiona.open(s):
                output.write(features)

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    samples = 'sample_points.shp'
    s_dir = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'model_data', 'allstates_3')
    l = []
    for k in OBJECT_MAP.keys():
        d = os.path.join(s_dir, k, samples)
        l.append(d)
    fiona_merge(os.path.join(s_dir, 'merge_5.shp'), l)
# ========================= EOF ====================================================================
