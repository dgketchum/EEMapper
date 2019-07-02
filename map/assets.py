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
import csv
import os
from pprint import pprint
from subprocess import Popen, PIPE, check_call

EDIT_STATES = ['KS', 'ND', 'NE', 'OK', 'SD']
TEST_YEARS = [1986, 1996, 2006, 2016]

home = os.path.expanduser('~')
EXEC = os.path.join(home, 'miniconda2', 'envs', 'ee', 'bin', 'earthengine')


def change_permissions(ee_asset, user=None):
    reader = list_assets(ee_asset)
    _list = [x for x in reader if x[-4:] in ['2016', '2017', '2018']]
    for r in _list:
        command = 'acl'
        cmd = ['{}'.format(EXEC), '{}'.format(command), 'set', 'public', '{}'.format(r)]
        print(cmd)
        check_call(cmd)


def delete_assets(ee_asset_path, years_=None):
    reader = None

    if years_:
        for year in range(2008, 2014):
            _dir = os.path.join(ee_asset_path, str(year))
            reader = list_assets(_dir)
    else:
        reader = list_assets(ee_asset_path)

    for r in reader:
        command = 'rm'
        cmd = ['{}'.format(EXEC), '{}'.format(command), '{}'.format(r)]
        check_call(cmd)


def rename_assets(ee_asset_path, new_path, years_=None):
    reader = None

    if years_:
        for year in range(1986, 2014):
            _dir = os.path.join(ee_asset_path, str(year))
            reader = list_assets(_dir)
    else:
        reader = list_assets(ee_asset_path)

    for old_name in reader:
        command = 'mv'
        new_name = os.path.join(new_path, os.path.basename(old_name))
        cmd = ['{}'.format(EXEC), '{}'.format(command), old_name, new_name]
        check_call(cmd)
        print(old_name, new_name)


def list_assets(location):
    command = 'ls'
    cmd = ['{}'.format(EXEC), '{}'.format(command), '{}'.format(location)]
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    reader = csv.DictReader(stdout.decode('ascii').splitlines(),
                            delimiter=' ', skipinitialspace=True,
                            fieldnames=['name'])
    assets = [x['name'] for x in reader]
    return assets


if __name__ == '__main__':
    loc = os.path.join('users', 'dgketchum', 'classy')
    change_permissions(loc)

# ========================= EOF ====================================================================
