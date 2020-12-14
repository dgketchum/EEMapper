import os
import ee
import numpy as np
import os
from collections import defaultdict

ee.Initialize()


def confusion(irr_labels, unirr_labels, irr_image, unirr_image, state):
    domain = 'users/dgketchum/boundaries/{}'.format(state)
    domain = ee.FeatureCollection(domain)
    domain = domain.toList(domain.size()).get(0)
    domain = ee.Feature(domain)

    true_positive = irr_image.eq(irr_labels)  # pred. irrigated, labeled irrigated
    false_positive = irr_image.eq(unirr_labels)  # pred. irrigated, labeled unirrigated

    true_negative = unirr_image.eq(unirr_labels)  # pred unirrigated, labeled unirrigated
    false_negative = unirr_image.eq(irr_labels)  # pred unirrigated, labeled irrigated

    TP = true_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FP = false_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FN = false_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    TN = true_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )

    out = {}
    out['TP'] = TP.getInfo()
    out['FP'] = FP.getInfo()
    out['FN'] = FN.getInfo()
    out['TN'] = TN.getInfo()
    return out


def create_lanid_labels(year):
    begin = '{}-01-01'.format(year)
    end = '{}-12-31'.format(year)
    lanid = ee.ImageCollection('projects/openet/irrigated_area/LANID').filterDate(begin, end).first().select("irr_land")
    irr_mask = lanid.eq(1)
    unmasked = lanid.unmask(0)
    unirr_image = ee.Image(1).byte().updateMask(unmasked.Not())
    irr_image = ee.Image(1).byte().updateMask(irr_mask)
    return irr_image, unirr_image


def create_rf_labels(year):
    begin = '{}-01-01'.format(year)
    end = '{}-12-31'.format(year)
    rf = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp')
    rf = rf.filter(ee.Filter.date(begin, end)).select('classification').mosaic()
    irrMask = rf.lt(1)
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not())
    irrImage = ee.Image(1).byte().updateMask(irrMask)
    return irrImage, unirrImage


def create_irrigated_labels(all_data, year):
    if all_data:
        non_irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/dryland')
        fallow = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/fallow')
        irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))
    else:
        root = 'projects/ee-dgketchum/assets/validation_polygons/'
        non_irrigated = ee.FeatureCollection(root + 'uncultivated_3DEC2020')
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'unirrigated_29NOV2020'))
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'wetlands_14JUL2020'))

        fallow = ee.FeatureCollection(root + 'fallow_2DEC2020')
        irrigated = ee.FeatureCollection(root + 'irrigated_7DEC2020')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))

    irr_labels = ee.Image(1).byte().paint(irrigated, 0)
    irr_labels = irr_labels.updateMask(irr_labels.Not())
    unirr_labels = ee.Image(1).byte().paint(non_irrigated, 0)
    unirr_labels = unirr_labels.updateMask(unirr_labels.Not())

    return irr_labels, unirr_labels


def metrics(arr):
    recall = arr[0, 0] / (arr[0, 1] + arr[0, 0])
    precision = arr[0, 0] / (arr[1, 0] + arr[0, 0])
    return precision, recall


if __name__ == '__main__':

    # im = np.array([[804828, 56825], [563617, 42072843]])
    # lid = np.array([[710976, 150677], [273559, 42362906]])

    conf = np.zeros((2, 2))
    for year in [2013]:
        print('\n {}'.format(year))
        for state in ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']:
            irr_labels, unirr_labels = create_irrigated_labels(False, year)
            irr_image, unirr_image = create_lanid_labels(year)
            cmt = confusion(irr_labels, unirr_labels, irr_image, unirr_image, state)
            print('{} {}'.format(state, cmt))
            for pos, ct in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['TP', 'FN', 'FP', 'TN']):
                conf[pos] += cmt[ct]['constant']
    print(conf)
    p, r = metrics(conf)
    print('prec {}, rec {}'.format(p, r))

# ========================= EOF ====================================================================
