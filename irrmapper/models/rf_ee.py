"""Random Forest classification on Earth Engine.

The production classifier: trains ee.Classifier.smileRandomForest on a
band-extract training table and classifies the annual feature stack,
exporting per state-year to an EE asset collection or GCS.
"""

import os
from datetime import date
from pprint import pprint

import ee

from irrmapper.features.stack import get_alpha_earth_bands, stack_bands


def export_classification(out_name, table, asset_root, region, years, alpha_earth=False,
                          export='asset', bag_fraction=0.5, input_props=None, southern=False,
                          extra_props=None):
    """
    Trains a Random Forest classifier using a training table input, creates a stack of raster images of the same
    features, and classifies it.  I run this over a for-loop iterating state by state.
    :param region:
    :param asset_root:
    :param out_name:
    :param asset:
    :param export:
    :param bag_fraction:
    :param extra_props: dict of additional properties set on the exported image (run provenance)
    :return:
    """
    fc = ee.FeatureCollection(table)
    roi = ee.FeatureCollection(region)

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=150,
        minLeafPopulation=1,
        bagFraction=bag_fraction).setOutputMode('CLASSIFICATION')

    if not input_props:
        input_props = fc.first().propertyNames().remove('YEAR').remove('POINT_TYPE').remove('system:index')
    else:
        input_props = ee.List(input_props)

    trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

    for yr in years:

        if alpha_earth:
            input_bands = get_alpha_earth_bands(yr, roi)
        else:
            input_bands = stack_bands(yr, roi, southern)

        b, p = input_bands.bandNames().getInfo(), input_props.getInfo()
        check = [x for x in p if x not in b]
        if check:
            pprint(check)
            revised = [f for f in p if f not in check]
            input_props = ee.List(revised)
            trained_model = classifier.train(fc, 'POINT_TYPE', input_props)

        annual_stack = input_bands.select(input_props)
        classified_img = annual_stack.unmask().classify(trained_model).int().set({
            'system:index': ee.Date('{}-01-01'.format(yr)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-01-01'.format(yr)).millis(),
            'system:time_end': ee.Date('{}-12-31'.format(yr)).millis(),
            'date_ingested': str(date.today()),
            'image_name': out_name,
            'training_data': table,
            'bag_fraction': bag_fraction,
            'class_key': '0: irrigated, 1: rainfed, 2: uncultivated, 3: wetland'})

        if extra_props:
            classified_img = classified_img.set(extra_props)

        classified_img = classified_img.clip(roi.geometry())

        if export == 'asset':
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                assetId=os.path.join(asset_root, '{}_{}'.format(out_name, yr)),
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)

        elif export == 'cloud':
            # pyramidingPolicy applies only to asset exports; the current
            # earthengine-api rejects it for GeoTIFF export
            task = ee.batch.Export.image.toCloudStorage(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                bucket='wudr',
                fileNamePrefix='{}_{}'.format(yr, out_name),
                scale=30,
                maxPixels=1e13)
        else:
            raise NotImplementedError('choose asset or cloud for export')

        task.start()
        print(os.path.join(asset_root, '{}_{}'.format(out_name, yr)))
