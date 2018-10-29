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

from pandas import read_csv, concat


def concatenate(root, out_dir, glob='None'):
    l = [os.path.join(root, x) for x in os.listdir(root) if glob in x]
    l.sort()
    first = True
    for csv in l:
        if first:
            df = read_csv(csv)
            first = False
        else:
            df = concat([df, read_csv(csv)])
    df.drop(columns=['system:index', '.geo'], inplace=True)
    out_file = os.path.join(out_dir, '{}_b.csv'.format(glob))
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    extracts = os.path.join(home, 'IrrigationGIS', 'EE_extracts')
    rt = os.path.join(extracts, 'to_concatenate')
    out = os.path.join(extracts, 'concatenated')
    concatenate(rt, out, glob='sample_')
# ========================= EOF ====================================================================
