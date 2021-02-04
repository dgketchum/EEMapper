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
    for r in reader:
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
        print(cmd)
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


def mask_move(min_years=3):

    asset_root = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper_RF2'

    image_list = list_assets('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp')
    image_list = [x for x in image_list if 'MT' in x]
    coll = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp').select('classification')
    remap = coll.map(lambda x: x.lt(1))
    sum_mask = remap.sum().lt(min_years)

    for image in image_list:
        desc = os.path.basename(image)
        yr = int(desc[-4:])
        img = ee.Image(image).remap([0, 1, 2, 3], [1, 0, 0, 0]).mask(sum_mask)
        img = img.unmask(0).select(['remapped'], ['classification'])
        img = img.set({
            'system:index': ee.Date('{}-01-01'.format(yr)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-01-01'.format(yr)).millis(),
            'system:time_end': ee.Date('{}-12-31'.format(yr)).millis(),
            'image_name': desc,
            'class_key': '1: irrigated, 0: unirrigated'})

        task = ee.batch.Export.image.toAsset(
            image=img,
            description=desc,
            assetId=os.path.join(asset_root, desc),
            scale=30,
            pyramidingPolicy={'.default': 'mode'},
            maxPixels=1e13)
        task.start()
        print(desc)


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
    images = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper_UCRB'
    dest = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapper22JAN2021'
    delete_assets(images)
# ========================= EOF ====================================================================
