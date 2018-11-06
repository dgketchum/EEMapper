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
import csv
from subprocess import Popen, PIPE, check_call


def delete_assets(ee_path, loc):
    command = 'ls'

    for year in range(2008, 2014):
        _dir = os.path.join(loc, str(year))
        cmd = ['{}'.format(ee_path), '{}'.format(command), '{}'.format(_dir)]
        l = Popen(cmd, stdout=PIPE)
        stdout, stderr = l.communicate()
        reader = csv.DictReader(stdout.decode('ascii').splitlines(),
                                delimiter=' ', skipinitialspace=True,
                                fieldnames=['name'])
    for r in reader:
        command = 'rm'
        cmd = ['{}'.format(ee_path), '{}'.format(command), '{}'.format(r['name'])]
        check_call(cmd)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    loc = os.path.join('users', 'dgketchum', 'ssebop', 'MT')
    ogr = os.path.join(home, 'miniconda2', 'envs', 'ee', 'bin', 'earthengine')
    delete_assets(ogr, loc)

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
