"""Point extracts of the feature stack (training data) and of classifications
(validation samples), exported as CSV to GCS.
"""

import ee

from irrmapper.assets.ops import list_assets
from irrmapper.features.stack import stack_bands

# list of years we have verified irrigated fields
YEARS = [1986, 1987, 1988, 1989, 1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
         2016, 2017, 2018, 2019]


def request_validation_extract(roi, file_prefix='validation'):
    """
    This takes a sample points set and extracts the classification result.  This is a roundabout cross-validation.
    Rather than using holdouts in the Random Forest classifier, we just run all the training data to train the
    classifier, and come back later with this function and a seperate set of points (with known classes) to
    independently test classifier accuracy.
    Other options to achieve this is to use out-of-bag cross validation, or set up a sckikit-learn RF classifier and
    use k-folds cross validation.
    :param file_prefix:
    :return:
    """
    roi = ee.FeatureCollection(roi)
    plots = ee.FeatureCollection(None).filterBounds(roi)
    image_list = list_assets('users/dgketchum/IrrMapper/version_2')

    for yr in YEARS:
        yr_img = [x for x in image_list if x.endswith(str(yr))]
        coll = ee.ImageCollection(yr_img)
        classified = coll.mosaic().select('classification')

        filtered = plots.filter(ee.Filter.eq('YEAR', yr))

        plot_sample_regions = classified.sampleRegions(
            collection=filtered,
            properties=['POINT_TYPE', 'YEAR', 'FID'],
            scale=30)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print(yr)


def request_band_extract(file_prefix, points_layer, region, years, filter_bounds=False, buffer=None,
                         southern=False, filter_years=True, diagnose=False, properties=None):
    """
    Extract raster values from a points kml file in Fusion Tables. Send annual extracts .csv to GCS wudr bucket.
    Concatenate them using irrmapper.sampling.tables.concatenate_band_extract().
    :param region:
    :param points_layer:
    :param file_prefix:
    :param filter_bounds: Restrict extract to within a geographic extent.
    :return:
    """
    roi = ee.FeatureCollection(region)
    if buffer:
        roi = ee.Feature(roi.first()).buffer(buffer)
        roi = ee.FeatureCollection([roi])
    plots = ee.FeatureCollection(points_layer)
    for yr in years:
        stack = stack_bands(yr, roi, southern)

        if filter_bounds:
            plots = plots.filterBounds(roi)

        if filter_years:
            filtered = plots.filter(ee.Filter.eq('YEAR', yr))
        else:
            filtered = plots

        # if tables are coming out empty, use this to find missing bands
        if diagnose:
            filtered = ee.FeatureCollection([filtered.first()])
            bad_ = []
            bands = stack.bandNames().getInfo()
            for b in bands:
                stack_ = stack.select([b])

                def sample_regions(i, points):
                    red = ee.Reducer.toCollection(i.bandNames())
                    reduced = i.reduceRegions(points, red, 30, stack_.select(b).projection())
                    fc = reduced.map(lambda f: ee.FeatureCollection(f.get('features'))
                                     .map(lambda q: q.copyProperties(f, None, ['features'])))
                    return fc.flatten()

                data = sample_regions(stack_, filtered)
                try:
                    print(b, data.getInfo()['features'][0]['properties'][b])
                except Exception as e:
                    print(b, 'not there', e)
                    bad_.append(b)
            print(bad_)
            return None

        props = ['POINT_TYPE', 'YEAR']
        if properties:
            props = properties

        plot_sample_regions = stack.sampleRegions(
            collection=filtered,
            properties=props,
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description='{}_{}'.format(file_prefix, yr),
            bucket='wudr',
            fileNamePrefix='{}_{}'.format(file_prefix, yr),
            fileFormat='CSV')

        task.start()
        print('{}_{}'.format(file_prefix, yr))
