from pprint import pprint
import csv
import os
import json
from datetime import date
import subprocess
from subprocess import Popen, PIPE, check_call
import json
import ee

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']

EDIT_STATES = ['KS', 'ND', 'NE', 'OK', 'SD']
TEST_YEARS = [1986, 1996, 2006, 2016]

home = os.path.expanduser('~')
EE = os.path.join(home, 'miniconda3', 'envs', 'irri', 'bin', 'earthengine')
GSUTIL = os.path.join(home, 'miniconda3', 'envs', 'irri', 'bin', 'gsutil')


def change_permissions(ee_asset, user=None):
    reader = list_assets(ee_asset)
    _list = [x for x in reader if x[-4:] in ['2016', '2017', '2018']]
    for r in _list:
        command = 'acl'
        cmd = ['{}'.format(EE), '{}'.format(command), 'set', 'private', '{}'.format(r)]
        print(cmd)
        check_call(cmd)


def copy_asset(ee_asset, dst):
    cmd = ['{}'.format(EE), 'cp', ee_asset, dst]
    check_call(cmd)


def asset_info(ee_asset):
    cmd = ['{}'.format(EE), 'asset', 'info', ee_asset]
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    str_ = stdout.decode('utf-8').replace("'", '"')
    data = json.loads(str_)
    return data


def duplicate_asset(ee_asset):
    reader = list_assets(ee_asset)
    _list = [x for x in reader if x[-4:] in ['2016', '2017', '2018']]
    for r in _list:
        if '2018' in r:
            cmd = ['{}'.format(EE), 'cp', '{}'.format(r), '{}'.format(r.replace('2018', '2019'))]
            print(cmd)
            check_call(cmd)


def set_metadata(ee_asset, property='--time_start'):
    reader = list_assets(ee_asset)
    for r in reader:
        year = os.path.basename(r)
        cmd = ['{}'.format(EE), 'asset', 'set',
               '--time_start', '{}-12-31T00:00:00'.format(year), r]
        print(' '.join(cmd))
        check_call(cmd)
        cmd = ['{}'.format(EE), 'asset', 'set',
               '--time_end', '{}-12-31T00:00:00'.format(year), r]
        print(' '.join(cmd))
        check_call(cmd)


def get_metadata(ee_asset):
    cmd = ['{}'.format(EE), 'asset', 'info', ee_asset]
    meta = check_call(cmd)
    return meta


def delete_asset(ee_asset_path):
    command = 'rm'
    cmd = ['{}'.format(EE), '{}'.format(command), '{}'.format(ee_asset_path)]
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
        cmd = ['{}'.format(EE), '{}'.format(command), old_name, new_name]
        try:
            check_call(cmd)
            print(old_name, new_name)
        except subprocess.CalledProcessError:
            print('move {} failed'.format(old_name))


def move_asset(asset, out_name):
    command = 'mv'
    cmd = ['{}'.format(EE), command, asset, out_name]
    try:
        print(asset, out_name)
        check_call(cmd)
    except subprocess.CalledProcessError as e:
        print('move {} failed \n {}'.format(asset, e))


def export_to_cloud(location, bucket):
    images = list_assets(location)
    count = [i for i in images if 'count' in i]
    refet = [i for i in images if 'et_reference' in i]
    etrf = [i for i in images if 'et_actual' in i]
    exports = count + etrf + refet

    for i in exports:
        task = ee.batch.Export.image.toCloudStorage(
            image=ee.Image(i),
            description=os.path.basename(i),
            bucket=bucket,
            scale=30,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            # dimensions='41448x53138',
            crs="EPSG:3857")
        # task.start()
        cmd = ['{}'.format(EE), 'rm', i]
        check_call(cmd)
        print('rm', i)


def cancel_tasks():
    task_list = ee.data.getTaskList()
    for t in task_list:
        if t['state'] == 'READY' and 'IM_' in t['description']:
            cmd = ['{}'.format(EE), 'task', 'cancel', '{}'.format(t['id'])]
            check_call(cmd)
            print(cmd)


def clip():
    root = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
    dst = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp_'
    for s in ['MT']:
        geo = ee.FeatureCollection('users/dgketchum/boundaries/{}'.format(s)).geometry()
        for y in [x for x in range(1997, 2018)]:
            i = ee.Image(os.path.join(root, 'IM_{}_{}'.format(s, y)))
            target = os.path.join(dst, 'IM_{}_{}'.format(s, y))
            i = i.clip(geo)
            task = ee.batch.Export.image.toAsset(
                image=i,
                description=os.path.basename(target),
                assetId=target,
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)
            task.start()
            print(target)


def convert_to_boolean(in_col, out_col):
    l = list_assets(in_col)
    exist = list_assets(out_col)

    for a in l:
        i = ee.Image(a)
        info = i.getInfo()['properties']
        info['postprocessing_date'] = date.today().strftime('%Y-%m-%d')
        i = i.mask(i.eq(0))
        i = i.set(info)
        desc = '{}'.format(info['system:index'])
        out_id = os.path.join(out_col, desc)
        check_id = os.path.join(out_col, desc)

        if check_id in exist:
            continue

        task = ee.batch.Export.image.toAsset(
            image=i,
            description=desc,
            assetId=out_id,
            scale=30,
            pyramidingPolicy={'.default': 'mode'},
            maxPixels=1e13)
        task.start()
        print(desc)


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
    cmd = ['{}'.format(EE), '{}'.format(command), '{}'.format(location)]
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    reader = csv.DictReader(stdout.decode('ascii').splitlines(),
                            delimiter=' ', skipinitialspace=True,
                            fieldnames=['name'])
    assets = [x['name'] for x in reader]
    assets = [x for x in assets if 'Running' not in x]
    return assets


def filter_by_metadata(collection):
    assets = list_assets(collection)
    dct = {}
    for a in assets:
        try:
            cmd = ['{}'.format(EE), 'asset', 'info', '{}'.format(a)]
            popr = Popen(cmd, stdout=PIPE)
            stdout, stderr = popr.communicate()
            info = json.loads(stdout[86:])
            yr = int(info['properties']['date'][:4])
            version = info['properties']['tool_version']

            if yr not in dct.keys():
                dct[yr] = {version: 1}
            else:
                if not version in dct[yr].keys():
                    dct[yr][version] = 1
                else:
                    dct[yr][version] += 1
        except Exception as e:
            print(e)
            print(a)

    print(dct)


def is_authorized():
    try:
        ee.Initialize()  # investigate (use_cloud_api=True)
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False


def clean_gcs():
    command = 'ls'
    cmd = ['{}'.format(EE), '{}'.format(command), 'users/dgketchum/bands/state']
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    str_ = stdout.decode('utf-8').replace("'", '"')
    ee_l = str_.split()
    ee_l = [os.path.basename(x) for x in ee_l]

    cmd = ['{}'.format(GSUTIL), '{}'.format(command), 'gs://wudr/state_points']
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    str_ = stdout.decode('utf-8').replace("'", '"')
    gcs_l = str_.split()
    for csv in gcs_l:
        name_ = os.path.basename(csv).split('.')[0][7:]
        if name_ not in ee_l:
            cmd = [GSUTIL, 'rm', csv]
            # print(cmd)
            check_call(cmd)
    pass


if __name__ == '__main__':
    is_authorized()
    c = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
    o = 'projects/ee-dgketchum/assets/IrrMapper/version1_2'
    convert_to_boolean(c, o)

# ========================= EOF ====================================================================
