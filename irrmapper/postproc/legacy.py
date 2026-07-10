"""FROZEN v1.2 reference post-processing (verbatim from map/call_ee.py).

This is the hand-written per-state if/elif rule chain that produced the
v1.2 IrrMapperComp collection. The config-driven replacement is
irrmapper.postproc.exports.export_special; tests/test_export_special_expressions.py
locks this function's expression strings as golden fixtures and
tests/test_gate2_graph_parity.py proves the replacement builds identical
export graphs. Retire this module after the runner's first real production
postprocess run. Do not edit the rule logic.
"""

import os

import ee

from irrmapper.assets.ops import copy_asset
from irrmapper.ingest.cdl import get_cdl
from irrmapper.ingest.landsat import landsat_composites


def export_special(input_coll, out_coll, roi, description):
    fc = ee.FeatureCollection(roi)
    ned = ee.Image('USGS/NED')
    slope = ee.Terrain.products(ned).select('slope')

    for year in range(2022, 2023):
        start, end = '{}-03-01'.format(year), '{}-12-30'.format(year)
        ndvi = landsat_composites(year, start, end, fc, 'gs', composites_only=True).select('nd_max_gs')

        cropland = get_cdl(year)[1].select('cropland')

        target = ee.Image(os.path.join(input_coll, '{}_{}'.format(description, year)))
        props = target.getInfo()['properties']
        target = target.select('classification').clip(fc.geometry())

        sum_coll = ee.ImageCollection(input_coll)
        remap = ee.ImageCollection(sum_coll).map(lambda x: x.select('classification').remap([0, 1, 2, 3],
                                                                                            [1, 0, 0, 0]))
        sum = remap.sum().rename('sum')
        if description == 'MT':
            if year < 2010:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2009').filterBounds(fc)
            elif year < 2012:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2011').filterBounds(fc)
            elif year < 2014:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2013').filterBounds(fc)
            elif year < 2019:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2015').filterBounds(fc)
            else:
                pivot = ee.FeatureCollection('users/dgketchum/openet/pivots/mt_pivot_2019').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 6
            if year < 2008:

                expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                              ': (IRR == 0) && (SUM < {t}) ? 1' \
                              ': (IRR == 0) && (SLOPE > 10) ? 3' \
                              ': IRR'.format(t=threshold)

                target = expr.expression(expression_,
                                         {'IRR': expr.select('classification'),
                                          'SUM': expr.select('sum'),
                                          'NDVI': expr.select('nd_max_gs'),
                                          'SLOPE': expr.select('slope')})
            else:
                expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                              ': (IRR == 0) && (SUM < {t}) ? 1' \
                              ': (IRR == 0) && (SLOPE > 10) ? 3' \
                              ': IRR'.format(t=threshold)

                target = expr.expression(expression_,
                                         {'IRR': expr.select('classification'),
                                          'SUM': expr.select('sum'),
                                          'NDVI': expr.select('nd_max_gs'),
                                          'SLOPE': expr.select('slope')})

                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) ? 0' \
                              ': IRR'.format()

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})

        elif description == 'ID':
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2011 else (2023 - year - 1)
            if threshold < 0:
                threshold = 0
            threshold = 5

            expression_ = ' (IRR == 0) && (NDVI < 0.68) && (SUM > {t}) ? 1' \
                          ': (IRR != 0) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs')})

            expression_ = ' (IRR == 0) && (SLOPE > 6) ? 3' \
                          ': IRR'.format()

            target = target.expression(expression_,
                                       {'IRR': target.select('classification'),
                                        'SLOPE': expr.select('slope')})

            if year > 2010:
                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) && (SUM > {t}) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})

        elif description in ['WA', 'OR', 'NM', 'NV']:
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2016 else (2025 - year - 1)
            if threshold < 0:
                threshold = 0

            expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': (IRR == 0) && (SLOPE > 10) ? 3' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs'),
                                      'SLOPE': expr.select('slope')})

            if year > 2011:
                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) && (SUM > {t}) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})

        elif description in ['CO', 'WY', 'UT']:
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2016 else (2025 - year - 1)
            if threshold < 0:
                threshold = 0

            expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': (IRR == 0) && (SLOPE > 4) ? 3' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs'),
                                      'SLOPE': expr.select('slope')})

        elif description in ['CA']:
            pivot = ee.FeatureCollection('users/dgketchum/openet/western_17_pivots').filterBounds(fc)

            class_labels = ee.Image(0).byte()
            pivot = class_labels.paint(pivot, 1).rename('pivot')
            expr = target.addBands([sum, ndvi, slope, cropland, pivot])

            threshold = 5 if year < 2016 else (2021 - year - 1)
            if threshold < 0:
                threshold = 0

            expression_ = '(IRR == 1) && (NDVI > 0.75) && (SUM > {t}) ? 0' \
                          ': (IRR == 0) && (SUM < {t}) ? 1' \
                          ': (IRR == 0) && (SLOPE > 10) ? 3' \
                          ': IRR'.format(t=threshold)

            target = expr.expression(expression_,
                                     {'IRR': expr.select('classification'),
                                      'SUM': expr.select('sum'),
                                      'NDVI': expr.select('nd_max_gs'),
                                      'SLOPE': expr.select('slope')})

            if year > 2011:
                expression_ = '(IRR != 0) && (NDVI > 0.68) && (PIVOT == 1) && (SUM > {t}) ? 0' \
                              ': IRR'.format(t=threshold)

                target = target.expression(expression_,
                                           {'IRR': target.select('classification'),
                                            'SUM': expr.select('sum'),
                                            'NDVI': expr.select('nd_max_gs'),
                                            'PIVOT': expr.select('pivot')})
        else:
            src = os.path.join(input_coll, '{}_{}'.format(description, year))
            dst = os.path.join(out_coll, '{}_{}'.format(description, year))
            print('No rule written for this {}, copying'.format(description))
            copy_asset(src, dst)
            continue

        props.update({'post_process': expression_})
        target = target.set(props)
        target = target.rename('classification')

        desc = '{}_{}'.format(description, year)
        _id = os.path.join(out_coll, desc)
        task = ee.batch.Export.image.toAsset(
            target,
            description=desc,
            pyramidingPolicy={'.default': 'mode'},
            assetId=_id,
            scale=30,
            maxPixels=1e13)

        task.start()
        print(year, _id)
