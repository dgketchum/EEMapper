import csv
import os
from subprocess import Popen, PIPE, check_call

import ee

EDIT_STATES = ['KS', 'ND', 'NE', 'OK', 'SD']
TEST_YEARS = [1986, 1996, 2006, 2016]

home = os.path.expanduser('~')
EXEC = os.path.join(home, 'miniconda3', 'envs', 'gcs', 'bin', 'earthengine')


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


def set_metadata(ee_asset, property='--time_start'):
    reader = list_assets(ee_asset)
    for r in reader:
        year = os.path.basename(r)
        if int(year) > 1986:
            cmd = ['{}'.format(EXEC), 'asset', 'set',
                   '{}'.format(property), '{}-12-31T00:00:00'.format(year), r]
            print(' '.join(cmd))
            check_call(cmd)


def get_metadata(ee_asset):
    reader = list_assets(ee_asset)
    for r in reader:
        if 'project' in r:
            cmd = ['{}'.format(EXEC), 'asset', 'info', r]
            print(cmd)
            check_call(cmd)


def delete_assets(ee_asset_path):
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


def cancel_tasks():
    task_list = ee.data.getTaskList()
    for t in task_list:
        if t['state'] == 'READY' and 'IM_' in t['description']:
            cmd = ['{}'.format(EXEC), 'task', 'cancel', '{}'.format(t['id'])]
            check_call(cmd)
            print(cmd)


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


def is_authorized():
    try:
        ee.Initialize()  # investigate (use_cloud_api=True)
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


if __name__ == '__main__':
    is_authorized()
    asset = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
    cancel_tasks()
# ========================= EOF ====================================================================
