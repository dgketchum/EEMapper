import os
import ee
import os
from collections import defaultdict

ee.Initialize()


def confusion(irr_labels, unirr_labels, irr_image, unirr_image):
    mt = 'users/tcolligan0/Montana'
    mt = ee.FeatureCollection(mt)
    mt = mt.toList(mt.size()).get(0)
    mt = ee.Feature(mt)

    true_positive = irr_image.eq(irr_labels)  # pred. irrigated, labeled irrigated
    false_positive = irr_image.eq(unirr_labels)  # pred. irrigated, labeled unirrigated

    true_negative = unirr_image.eq(unirr_labels)  # pred unirrigated, labeled unirrigated
    false_negative = unirr_image.eq(irr_labels)  # pred unirrigated, labeled irrigated

    TP = true_positive.reduceRegion(
        geometry=mt.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e9,
        crs='EPSG:5070',
        scale=30
    )
    FP = false_positive.reduceRegion(
        geometry=mt.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e9,
        crs='EPSG:5070',
        scale=30
    )
    FN = false_negative.reduceRegion(
        geometry=mt.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e9,
        crs='EPSG:5070',
        scale=30
    )
    TN = true_negative.reduceRegion(
        geometry=mt.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e9,
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
    lanid = ee.Image('projects/openet/irrigated_area/LANID').filterDate(begin, end).first().select("irr_land")
    irr_mask = lanid.eq(1)  # lanid is already masked
    unmasked = lanid.unmask(0)
    unirr_image = ee.Image(1).byte().updateMask(unmasked.Not())
    irr_image = ee.Image(1).byte().updateMask(irr_mask)
    return irr_image, unirr_image


def create_unet_labels(year):
    unet = ee.Image('users/tcolligan0/irrigationMT/irrMT{}'.format(year))
    unet = unet.select(['b1', 'b2', 'b3'], ['irr', 'uncult', 'unirr'])
    irrImage = unet.select('irr')
    irrMask = irrImage.gt(unet.select('uncult'))
    irrMask = irrMask.And(unet.select('irr').gt(unet.select('unirr')))
    unirrImage = ee.Image(1).byte().updateMask(irrMask.neq(1))
    irrImage = ee.Image(1).byte().updateMask(irrMask.eq(1))
    return irrImage, unirrImage


def create_rf_labels(year):
    begin = '{}-01-01'.format(year)
    end = '{}-12-31'.format(year)
    rf = ee.ImageCollection('users/dgketchum/IrrMapper/version_2')
    rf = rf.filter(ee.Filter.date(begin, end)).select('classification').mosaic()
    irrMask = rf.lt(1)
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not())
    irrImage = ee.Image(1).byte().updateMask(irrMask)
    return irrImage, unirrImage


def create_mirad_labels(year):
    mirad = ee.Image('users/tcolligan0/MIRAD/mirad{}MT'.format(year))
    irrMask = mirad.eq(1)
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not())
    irrImage = ee.Image(1).byte().updateMask(irrMask)
    return irrImage, unirrImage


def create_irrigated_labels(all_data, year):
    if all_data:
        non_irrigated = ee.FeatureCollection('users/tcolligan0/merged_shapefile_unirr_wetlands_unc')
        fallow = ee.FeatureCollection('users/tcolligan0/fallow_11FEB')
        irrigated = ee.FeatureCollection('users/tcolligan0/irrigated_MT_13MAR2020')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))
    else:
        root = 'users/tcolligan0/test-data-aug24/'
        non_irrigated = ee.FeatureCollection(root + 'uncultivated_test')
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'unirrigated_test'))
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'wetlands_buffered_test'))

        fallow = ee.FeatureCollection(root + 'fallow_test')
        irrigated = ee.FeatureCollection(root + 'irrigated_test')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))

    irr_labels = ee.Image(1).byte().paint(irrigated, 0)
    irr_labels = irr_labels.updateMask(irr_labels.Not())
    unirr_labels = ee.Image(1).byte().paint(non_irrigated, 0)
    unirr_labels = unirr_labels.updateMask(unirr_labels.Not())

    return irr_labels, unirr_labels


if __name__ == '__main__':
    year = 2013

    irr_labels, unirr_labels = create_irrigated_labels(False, year)
    irr_image, unirr_image = create_rf_labels(year)

    # irr image: binary image w/ 1s where there are irrigated labels, 0 o.w.
    # unirr image: binary image w/ 1s where there are non-irrigated labels, 0 o.w.
    # irr labels: image of predicted irrigation
    # unirr labels: image of predicted non-irrigation
    cmt = confusion(irr_labels, unirr_labels, irr_image, unirr_image)
    # could dump this dict to json after srunning
    print(cmt)
    
# ========================= EOF ====================================================================