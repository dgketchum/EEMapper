import ee

ee.Initialize()
import time
import os

from datetime import datetime
import fiona
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from map.openet.collection import get_target_dates, Collection, get_target_bands
except ModuleNotFoundError:
    from openet.collection import get_target_dates, Collection, get_target_bands

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'ts_data'

BOUNDARIES = 'users/dgketchum/boundaries'
MGRS = os.path.join(BOUNDARIES, 'MGRS_TILE')
EE_DATA = 'users/dgketchum/grids'
MT = os.path.join(BOUNDARIES, 'MT')

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

CLASSES = ['uncultivated', 'dryland', 'fallow', 'irrigated']

YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997, 1998,
         2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

STATE_YEARS = {'AZ': [2001, 2003, 2004, 2007, 2010, 2012, 2016],
               'CA': [1995, 1998, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2014, 2016],
               'CO': [1998, 2003, 2006, 2008, 2010, 2011, 2013, 2016],
               'ID': [1986, 1988, 1996, 1997, 1998, 2001, 2002, 2003, 2006, 2008, 2009, 2010, 2011, 2013, 2017],
               'KS': [2002, 2006, 2009, 2012, 2013, 2014, 2015, 2016],
               'MT': [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015],
               'ND': [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016],
               'NE': [1993, 2003, 2009, 2012, 2013, 2014, 2015, 2016],
               'NM': [1987, 1988, 1989, 1994, 2001, 2002, 2004, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016],
               'NV': [2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009],
               'OK': [2006, 2007, 2011, 2012, 2013, 2014, 2015, 2016],
               'OR': [1994, 1996, 1997, 2001, 2011, 2013],
               'SD': [2007, 2008, 2009, 2012, 2013, 2014, 2015, 2016],
               'TX': [2005, 2006, 2009, 2012, 2013, 2014, 2015, 2016],
               'UT': [1998, 2001, 2003, 2005, 2006, 2008, 2009, 2010, 2011, 2012, 2013, 2016],
               'WA': [1988, 1996, 1997, 1998, 2001, 2006],
               'WY': [1998, 2003, 2006, 2013, 2016]}

DEBUG_SELECT = {'test': [202, 192, 220],
                'train': [676, 675, 715, 716, 677, 696, 717],
                'val': [225, 212, 239]}


def masks(roi, year_):
    root = 'users/dgketchum/training_polygons/'

    irr_mask = ee.FeatureCollection(os.path.join(root, 'irrigated_mask')).filterBounds(roi)
    irr_mask = irr_mask.filter(ee.Filter.eq("YEAR", year_))

    fallow_mask = ee.FeatureCollection(os.path.join(root, 'fallow_mask')).filterBounds(roi)
    fallow_mask = fallow_mask.filter(ee.Filter.eq("YEAR", year_))

    dryland_mask = ee.FeatureCollection(os.path.join(root, 'dryland_mask')).filterBounds(roi)

    root = 'users/dgketchum/uncultivated/'
    w_wetlands = ee.FeatureCollection(os.path.join(root, 'west_wetlands'))
    c_wetlands = ee.FeatureCollection(os.path.join(root, 'central_wetlands'))
    e_wetlands = ee.FeatureCollection(os.path.join(root, 'east_wetlands'))
    usgs_pad = ee.FeatureCollection(os.path.join(root, 'usgs_pad'))
    rf_uncult = ee.FeatureCollection(os.path.join(root, 'uncultivated'))
    uncultivated_mask = w_wetlands.merge(c_wetlands).merge(e_wetlands).merge(usgs_pad).merge(rf_uncult)
    uncultivated_mask = uncultivated_mask.filterBounds(roi)

    masks_ = {'irrigated': irr_mask,
              'fallow': fallow_mask,
              'dryland': dryland_mask,
              'uncultivated': uncultivated_mask}
    return masks_


def class_codes():
    return {'irrigated': 2,
            'fallow': 3,
            'dryland': 4,
            'uncultivated': 5}


def create_class_labels(name_fc):
    class_labels = ee.Image(1).byte()
    # paint irrigated last
    for name in ['uncultivated', 'dryland', 'fallow', 'irrigated']:
        class_labels = class_labels.paint(name_fc[name], class_codes()[name])
    label = class_labels.updateMask(class_labels).rename('irr')

    return label


def get_ancillary(year):
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select(['elevation', 'slope', 'aspect']) \
        .resample('bilinear').rename(['elv', 'slp', 'asp'])

    if 2007 < year < 2017:
        cdl = ee.ImageCollection('USDA/NASS/CDL') \
            .filter(ee.Filter.date('{}-01-01'.format(year), '{}-12-31'.format(year))) \
            .first().select(['cropland', 'confidence']).rename(['cdl', 'cconf'])
    else:
        cdl = ee.ImageCollection('USDA/NASS/CDL') \
            .filter(ee.Filter.date('2017-01-01', '2017-12-31')) \
            .first().select(['cropland', 'confidence']).rename(['cdl', 'cconf'])

    return terrain, cdl


def get_sr_stack(yr, s, e, interval, mask, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr + 1, e, 1)
    target_interval = interval
    interp_days = 32

    target_dates = get_target_dates(s, e, interval_=target_interval)

    model_obj = Collection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        mask=mask,
        geometry=geo_,
        cloud_cover_max=60)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
    interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
    return interp, target_rename


def extract_by_patch(feature_id=1440, split=None, cloud_mask=True):
    if not split:
        raise NotImplementedError

    grid = os.path.join(EE_DATA, '{}_grid'.format(split))
    roi = ee.FeatureCollection(grid).filter(ee.Filter.eq('FID', feature_id))
    geo = roi.geometry()

    points = os.path.join(EE_DATA, '{}_pts'.format(split))
    points = ee.FeatureCollection(points).filter(ee.Filter.eq('POLYFID', feature_id))
    points = points.toList(points.size())

    size = roi.size().getInfo()
    if size != 1:
        print('{} has {} features'.format(feature_id, size))
        return None

    info_ = roi.first().getInfo()
    state, irr = info_['properties']['STUSPS'], info_['properties']['IRR']
    years = STATE_YEARS[state]

    for year in years:
        if 2007 < year < 2017:
            s, e, interval_ = 1, 1, 30
            image_stack, features = get_sr_stack(year, s, e, interval_, cloud_mask, geo)

            masks_ = masks(geo, year)
            irr = create_class_labels(masks_)
            terrain_, cdl_ = get_ancillary(year)
            coords_ = image_stack.pixelLonLat().rename(['lon', 'lat'])
            image_stack = ee.Image.cat([image_stack, terrain_, coords_, cdl_, irr]).float()
            features = features + ['elv', 'slp', 'asp', 'lon', 'lat', 'cdl', 'cconf', 'irr']

            projection = ee.Projection('EPSG:5070')
            image_stack = image_stack.reproject(projection, None, 30)

            out_filename = '{}_{}_{}_{}'.format(split, state, feature_id, year)
            data_stack = image_stack.neighborhoodToArray(KERNEL)
            geometry_sample = ee.ImageCollection([])

            for i in range(9):
                region = ee.Feature(points.get(i)).geometry()
                sample = data_stack.sample(region=region,
                                           scale=30,
                                           numPixels=1,
                                           tileScale=16,
                                           dropNulls=False)
                geometry_sample = geometry_sample.merge(sample)

            task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                bucket=GS_BUCKET,
                description=out_filename,
                fileNamePrefix=out_filename,
                fileFormat='TFRecord',
                selectors=features)

            try:
                task.start()
            except ee.ee_exception.EEException:
                print('waiting to export, sleeping for 50 minutes. Holding at\
                        {}, feature {}'.format(year, feature_id))
                time.sleep(3000)
                task.start()
            print('exported', split, state, feature_id, year)


def extract_by_point(year, points_to_extract=None, cloud_mask=False,
                     n_shards=10, feature_id=1440, max_sample=100):
    if cloud_mask:
        cloud = 'cm'
    else:
        cloud = 'nm'
    roi = ee.FeatureCollection(os.path.join(EE_DATA, 'train_grid')).filter(ee.Filter.eq('FID', feature_id)).geometry()

    s, e, interval_ = 1, 1, 30
    image_stack, features = get_sr_stack(year, s, e, interval_, cloud_mask, roi)

    projection = ee.Projection('EPSG:5070')
    image_stack = image_stack.reproject(projection, None, 30)

    masks_ = masks(roi, year)
    if masks_['irrigated'].size().getInfo() == 0:
        print('no irrigated in {} in {}'.format(year, feature_id))
        return

    irr = create_class_labels(masks_)
    terrain_, cdl_ = get_ancillary(year)
    coords_ = image_stack.pixelLonLat().rename(['lon', 'lat'])
    image_stack = ee.Image.cat([image_stack, terrain_, coords_, cdl_, irr]).float()
    features = features + ['elv', 'slp', 'asp', 'lon', 'lat', 'cdl', 'cconf', 'irr']

    data_stack = image_stack.neighborhoodToArray(KERNEL)

    for fc in points_to_extract:
        class_ = os.path.basename(fc)
        points = ee.FeatureCollection(fc).filterBounds(roi)
        n_features = points.size().getInfo()

        if n_features == 0:
            print('no {} in {} in {}'.format(class_, year, feature_id))
            break

        points = points.toList(points.size())

        if max_sample and n_features > max_sample:
            # Error: List.get: List index must be between 0 and 99, or -100 and -1. Found 115.
            points = points.slice(0, max_sample)
            print(n_features, fc)

        geometry_sample = ee.ImageCollection([])
        ct = 1
        for i in range(n_features):
            # points.get() will only fetch to 99
            sample = data_stack.sample(
                region=ee.Feature(points.get(i)).geometry(),
                scale=30,
                numPixels=1,
                tileScale=16)

            geometry_sample = geometry_sample.merge(sample)
            if (i + 1) % n_shards == 0:

                out_filename = '{}_{}_{}_{}_{}'.format(class_, str(year), ct, cloud, feature_id)

                task = ee.batch.Export.table.toCloudStorage(
                    collection=geometry_sample,
                    bucket=GS_BUCKET,
                    description=out_filename,
                    fileNamePrefix=out_filename,
                    fileFormat='TFRecord',
                    selectors=features)

                try:
                    task.start()
                except ee.ee_exception.EEException:
                    print('waiting to export, sleeping for 50 minutes. Holding at\
                                {} {}, index {}'.format(year, class_, i))
                    time.sleep(3000)
                    task.start()

                print('exported {} {}, {} features'.format(feature_id, year, n_features))
                geometry_sample = ee.ImageCollection([])
                ct += 1
            if ct >= 11:
                break

        if ct > 1:
            ct += 1
        out_filename = '{}_{}_{}_{}_{}'.format(class_, str(year), ct, cloud, feature_id)
        task = ee.batch.Export.table.toCloudStorage(
            collection=geometry_sample,
            bucket=GS_BUCKET,
            description=out_filename,
            fileNamePrefix=out_filename,
            fileFormat='TFRecord',
            selectors=features)
        task.start()
        print('exported {} {}, {} features'.format(feature_id, year, n_features))


def run_extract_patches(shp, split):
    with fiona.open(shp, 'r') as src:
        for f in src:
            fid_ = f['properties']['FID']
            extract_by_patch(fid_, split, cloud_mask=True)


def run_extract_points(shp, points_assets, last_touch=None):
    fids = subsample_fid()
    if last_touch:
        fids = fids[fids.index(last_touch):]
    dct = {fid: [] for fid in fids[1:]}
    with fiona.open(shp, 'r') as src:
        for i, f in enumerate(src):
            fid = f['properties']['FID']
            yr = f['properties']['YEAR']
            try:
                if yr not in dct[fid]:
                    dct[fid].append(yr)
            except KeyError:
                pass
    for fid, years in dct.items():
        for year in years:
            extract_by_point(year, points_to_extract=points_assets, feature_id=fid, cloud_mask=True)


def subsample_fid():
    return [2, 3, 5, 6, 7, 8, 11, 12, 14, 21, 22, 23, 24, 25, 26, 27, 29, 30, 34, 35, 36, 38, 39, 40, 42, 43, 44, 45,
            46, 47, 48, 49, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 68, 70, 71, 72, 73, 74, 76, 79, 80, 81, 82, 83,
            84, 86, 87, 89, 92, 94, 95, 96, 97, 98, 100, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
            116, 117, 118, 119, 121, 123, 124, 125, 127, 128, 129, 130, 131, 132, 135, 136, 137, 140, 141, 145, 146,
            147, 148, 149, 150, 151, 152, 154, 157, 158, 160, 162, 163, 165, 166, 168, 171, 174, 177, 178, 179, 180,
            182, 183, 184, 185, 188, 190, 191, 192, 193, 197, 198, 199, 203, 204, 205, 206, 207, 208, 209, 211, 212,
            213, 215, 216, 217, 220, 223, 225, 226, 228, 229, 231, 232, 233, 235, 237, 238, 240, 241, 242, 243, 245,
            246, 249, 250, 251, 252, 255, 256, 257, 260, 261, 263, 265, 266, 267, 268, 270, 271, 272, 273, 274, 275,
            277, 278, 279, 280, 283, 284, 285, 286, 288, 289, 290, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305,
            306, 308, 312, 313, 314, 315, 316, 318, 319, 321, 323, 324, 328, 329, 330, 331, 332, 333, 334, 335, 339,
            340, 341, 346, 347, 357, 358, 361, 362, 363, 364, 369, 370, 371, 375, 376, 377, 378, 379, 384, 385, 388,
            389, 395, 397, 402, 406, 407, 411, 412, 415, 422, 423, 425, 430, 433, 434, 439, 441, 442, 443, 444, 445,
            448, 452, 454, 456, 459, 461, 462, 463, 467, 468, 469, 476, 477, 480, 481, 482, 483, 484, 485, 486, 488,
            491, 493, 494, 495, 496, 497, 499, 501, 502, 508, 510, 511, 513, 514, 519, 520, 522, 526, 527, 529, 530,
            532, 533, 534, 536, 538, 541, 543, 547, 548, 549, 550, 553, 554, 555, 556, 557, 558, 567, 572, 573, 574,
            575, 576, 578, 579, 581, 582, 584, 585, 587, 590, 591, 592, 593, 594, 598, 599, 601, 603, 609, 610, 611,
            612, 613, 616, 617, 618, 620, 633, 636, 637, 638, 639, 640, 644, 646, 652, 653, 654, 658, 659, 660, 661,
            662, 666, 671, 672, 674, 675, 676, 682, 683, 684, 686, 687, 695, 702, 703, 704, 707, 708, 713, 714, 715,
            716, 717, 719, 721, 722, 723, 724, 725, 727, 737, 740, 741, 742, 743, 745, 746, 749, 752, 753, 766, 770,
            771, 773, 775, 777, 781, 791, 794, 795, 799, 800, 806, 807, 817, 819, 820, 821, 822, 823, 826, 834, 842,
            843, 845, 846, 847, 848, 849, 850, 851, 852, 853, 863, 864, 865, 866, 867, 869, 873, 874, 890, 891, 893,
            894, 895, 896, 897, 910, 911, 913, 914, 915, 916, 917, 918, 919, 921, 929, 931, 933, 934, 938, 941, 947,
            949, 950, 951, 952, 953, 956, 957, 958, 967, 971, 972, 973, 974, 975, 976, 977, 978, 980, 990, 991, 992,
            993, 994, 995, 996, 999, 1000, 1001, 1015, 1016, 1017, 1020, 1036, 1038, 1039, 1040, 1041, 1042, 1043, 1044,
            1057, 1058, 1059, 1062, 1075, 1086, 1087, 1100, 1101, 1102, 1104, 1105, 1107, 1108, 1109, 1123, 1124, 1125,
            1140, 1141, 1142, 1143, 1144, 1145, 1157, 1159, 1160, 1161, 1179, 1180, 1183, 1189, 1199, 1201, 1202, 1204,
            1209, 1216, 1223, 1224, 1225, 1226, 1232, 1234, 1235, 1243, 1245, 1255, 1256, 1257, 1258, 1259, 1260, 1261,
            1262, 1264, 1266, 1267, 1274, 1276, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1293, 1294, 1296, 1301,
            1302, 1303, 1304, 1305, 1306, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1337, 1345, 1346, 1347, 1349,
            1361, 1362, 1364, 1368, 1370, 1376, 1380, 1387, 1388, 1394, 1395, 1397, 1415, 1416, 1417, 1424, 1427, 1429,
            1430, 1432, 1436, 1437, 1443, 1444, 1446, 1448, 1449, 1451, 1452, 1455, 1462, 1463, 1464]


if __name__ == '__main__':
    home = os.path.expanduser('~')
    alt_home = os.path.join(home, 'data')
    if os.path.isdir(alt_home):
        home = alt_home
    grids = os.path.join(home, 'IrrigationGIS', 'EE_sample', 'grid')
    centroids = os.path.join(home, 'IrrigationGIS', 'EE_sample', 'centroids')

    # splits = ['train', 'test', 'val']
    # shapes = [os.path.join(grids, '{}_grid_select.shp'.format(splt)) for splt in splits]
    # for shape, split_ in zip(shapes, splits):
    #     run_extract(shape, split_)

    pts_root = 'users/dgketchum/training_points'
    pts_training = [os.path.join(pts_root, x) for x in ['irrigated', 'fallow']]
    pts_irr = os.path.join(centroids, 'irrigated_train_buf.shp')
    run_extract_points(pts_irr, pts_training, last_touch=874)
    # for yr_ in YEARS:
    #     for fid in subsample_fid():
    #         extact_by_point(yr_, points_to_extract=pts_training, feature_id=fid)
# =====================================================================================================================
