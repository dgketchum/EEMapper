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
from subprocess import Popen, PIPE, check_call

EDIT_STATES = ['KS', 'ND', 'NE', 'OK', 'SD']
TEST_YEARS = [1986, 1996, 2006, 2016]

home = os.path.expanduser('~')
EXEC = os.path.join(home, 'miniconda2', 'envs', 'gee', 'bin', 'earthengine')


def change_permissions(ee_asset, user=None):
    reader = list_assets(ee_asset)
    _list = [x for x in reader if x[-4:] in ['2016', '2017', '2018']]
    for r in _list:
        command = 'acl'
        cmd = ['{}'.format(EXEC), '{}'.format(command), 'set', 'private', '{}'.format(r)]
        print(cmd)
        check_call(cmd)


def copy_asset(ee_asset, dst):
    reader = list_assets(ee_asset)
    _list = [x for x in reader if x[-4:] in ['2016', '2017', '2018']]
    for r in _list:
        cmd = ['{}'.format(EXEC), 'cp', '{}'.format(r), '{}'.format(os.path.join(dst, os.path.basename(r)))]
        print(cmd)
        check_call(cmd)


def duplicate_asset(ee_asset):
    reader = list_assets(ee_asset)
    _list = [x for x in reader if x[-4:] in ['2016', '2017', '2018']]
    for r in _list:
        if '2018' in r:
            cmd = ['{}'.format(EXEC), 'cp', '{}'.format(r), '{}'.format(r.replace('2018', '2019'))]
            print(cmd)
            check_call(cmd)


def set_metadata(ee_asset, key, value):
    reader = list_assets(ee_asset)
    for r in reader:
        if 'dgketchum' in r:
            cmd = ['{}'.format(EXEC), 'asset', 'set', '-p',
                   '{}={}'.format(key, value), r]
            print(cmd)
            check_call(cmd)


def get_metadata(ee_asset):
    reader = list_assets(ee_asset)
    for r in reader:
        if 'project' in r:
            cmd = ['{}'.format(EXEC), 'asset', 'info', r]
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
        if 'projects' in r:
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
    assets = [x for x in assets if 'Running' not in x]
    return assets


if __name__ == '__main__':
    loc = os.path.join('users', 'dgketchum', 'IrrMapper', 'version_2')
    dst = os.path.join('projects', 'openet', 'irrigation', 'IrrMapper_v2')
    k = 'version'
    v = '2'
    # copy_asset(loc, dst)
    # set_metadata(loc, key=k, value=v)
    get_metadata(dst)
    # delete_assets(dst)
    # duplicate_asset(dst)
# ========================= EOF ====================================================================
