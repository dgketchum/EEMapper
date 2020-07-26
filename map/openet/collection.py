import os
from pprint import pprint
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
import ee

from map.openet.image import Image
import map.openet.inerpolate as interp
from map.openet.utils import date_0utc


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated

    https://stevenloria.com/lazy-properties/
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class Collection:
    """"""

    def __init__(
            self,
            collections,
            start_date,
            end_date,
            geometry,
            variables=None,
            cloud_cover_max=70):

        self.collections = collections
        self.start_date = start_date
        self.end_date = end_date
        self.start_str = self.start_date.strftime('%Y-%m-%d')
        self.end_str = self.end_date.strftime('%Y-%m-%d')

        self.variables = variables
        self.geometry = geometry
        self.cloud_cover_max = cloud_cover_max
        self._interp_vars = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir', 'ndvi']

        # If collections is a string, place in a list
        if type(self.collections) is str:
            self.collections = [self.collections]

    def _build(self, interp_vars):
        """Build a merged model variable image collection

        Parameters
        ----------
        variables : list
            Set a variable list that is different than the class variable list.
        start_date : str, optional
            Set a start_date that is different than the class start_date.
            This is needed when defining the scene collection to have extra
            images for interpolation.
        end_date : str, optional
            Set an exclusive end_date that is different than the class end_date.

        Returns
        -------
        ee.ImageCollection

        Raises
        ------
        ValueError if collection IDs are invalid.
        ValueError if variables is not set here and in class init.

        """

        # Build the variable image collection
        variable_coll = ee.ImageCollection([])
        for coll_id in self.collections:
            input_coll = ee.ImageCollection(coll_id) \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(self.geometry) \
                .filterMetadata('CLOUD_COVER_LAND', 'less_than',
                                self.cloud_cover_max)

            # Time filters are to remove bad (L5) and pre-op (L8) images
            if 'LT05' in coll_id:
                input_coll = input_coll.filter(ee.Filter.lt(
                    'system:time_start', ee.Date('2011-12-31').millis()))
            elif 'LC08' in coll_id:
                input_coll = input_coll.filter(ee.Filter.gt(
                    'system:time_start', ee.Date('2013-03-24').millis()))

            def compute_lsr(image):
                model_obj = Image.from_landsat_c1_sr(
                    sr_image=ee.Image(image))
                return model_obj.calculate(interp_vars)

            variable_coll = variable_coll.merge(
                ee.ImageCollection(input_coll.map(compute_lsr)))

        return variable_coll

    def scenes(self, variables):
        interp_vars = [band for band in variables]
        interp_vars.append('time')
        scene_coll = self._build(interp_vars)
        return ee.ImageCollection(scene_coll).select(variables)

    def interpolate(self, variables, interp_days=32, dates=None):

        ref_et = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
            .filter(ee.Filter.inList("system:time_start", dates)) \
            .select(['etr'], ['et_reference'])

        interp_vars = [band for band in variables]

        # Count will be determined using the aggregate_coll image masks
        if 'count' in variables:
            interp_vars.append('mask')

        interp_vars.append('time')

        # Build initial scene image collection
        scene_coll = self._build(interp_vars)
        # For count, compute the composite/mosaic image for the mask band only
        if 'count' in variables:
            aggregate_coll = interp.aggregate_to_daily(
                image_coll=scene_coll.select(['mask']),
                start_date=self.start_date, end_date=self.end_date)

            aggregate_coll = aggregate_coll.merge(
                ee.Image.constant(0).rename(['mask']).set({'system:time_start': ee.Date(self.start_str).millis()}))

        if 'mask' in interp_vars:
            interp_vars.remove('mask')

        # Interpolate to a daily time step
        daily_coll = interp.daily(
            target_coll=ref_et,
            source_coll=scene_coll.select(interp_vars),
            interp_days=interp_days)

        interp_properties = {
            'cloud_cover_max': self.cloud_cover_max,
            'collections': ', '.join(self.collections),
            'interp_days': interp_days,
            'interp_method': 'linear',
            'model_name': 'IrrMapper'}

        def aggregate_image(agg_start_date, agg_end_date, date_format):

            et_reference_img = daily_coll \
                .filterDate(agg_start_date, agg_end_date) \
                .select(['et_reference']).sum()

            image_list = [et_reference_img.float()]

            for var in variables:
                # Compute average ndvi over the aggregation period
                ndvi_img = daily_coll \
                    .filterDate(agg_start_date, agg_end_date) \
                    .mean().select([var]).float()
                image_list.append(ndvi_img)
            if 'count' in variables:
                count_img = aggregate_coll \
                    .filterDate(agg_start_date, agg_end_date) \
                    .select(['mask']).count().rename('count').uint8()
                image_list.append(count_img)

            return ee.Image(image_list) \
                .set(interp_properties) \
                .set({'system:index': ee.Date(agg_start_date).format(date_format),
                      'system:time_start': ee.Date(agg_start_date).millis(),
                      })

        def aggregate_daily(daily_img):
            agg_start_date = ee.Date(daily_img.get('system:time_start'))
            return aggregate_image(
                agg_start_date=agg_start_date,
                agg_end_date=ee.Date(agg_start_date).advance(1, 'day'),
                date_format='YYYYMMdd')

        return ee.ImageCollection(daily_coll.map(aggregate_daily)).select(variables)

    def get_image_ids(self):
        """Return image IDs of the input images

        Returns
        -------
        list

        Notes
        -----
        This image list is based on the collection start and end dates and may
        not include all of the images used for interpolation.

        """
        return sorted(list(self._build().aggregate_array('image_id').getInfo()))


def get_target_dates(s, e, interval_=15):
    d_times = [(d, d + timedelta(days=1)) for d in rrule(dtstart=s, until=e, interval=interval_, freq=DAILY)]
    d_strings = [(x.strftime('%Y-%m-%d'), y.strftime('%Y-%m-%d')) for x, y in d_times]
    gm = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    images = [gm.filterDate(s, e).first().getInfo()['properties']['system:time_start'] for s, e in d_strings]
    return images


def get_target_bands(s, e, interval_=15, vars=None):

    d_times = [d for d in rrule(dtstart=s, until=e, interval=interval_, freq=DAILY)]
    d_strings = [x.strftime('%Y%m%d') for x in d_times]

    collection_bands = [['{}_{}'.format(d, b) for b in vars] for d in d_strings]
    collection_bands = [item for sublist in collection_bands for item in sublist]

    rename_bands = [['{}_{}'.format(b, d) for b in vars] for d in d_strings]
    rename_bands = [item for sublist in rename_bands for item in sublist]

    return collection_bands, rename_bands


if __name__ == '__main__':
    ee.Initialize(use_cloud_api=True)

    ndvi_palette = ['#EFE7E1', '#003300']
    et_palette = [
        'DEC29B', 'E6CDA1', 'EDD9A6', 'F5E4A9', 'FFF4AD', 'C3E683', '6BCC5C',
        '3BB369', '20998F', '1C8691', '16678A', '114982', '0B2C7A']

    image_size = 768
    landsat_cs = 30

    collections = ['LANDSAT/LC08/C01/T1_SR',
                   'LANDSAT/LE07/C01/T1_SR',
                   'LANDSAT/LT05/C01/T1_SR']

    year = 2017
    s = datetime(year, 3, 1)
    e = datetime(year, 11, 1)
    target_interval = 15

    interp_days = 32
    test_xy = [ee.Feature(ee.Geometry.Point(-112.20878671819048, 47.5176895106640), {'POINT_TYPE': 0}),
               ee.Feature(ee.Geometry.Point(-111.61724610359778, 47.6963644206754), {'POINT_TYPE': 1}),
               ee.Feature(ee.Geometry.Point(-112.32898130320568, 47.4473514239175), {'POINT_TYPE': 2}),
               ee.Feature(ee.Geometry.Point(-111.74121361280626, 47.4209409545644), {'POINT_TYPE': 3})]

    fc = ee.FeatureCollection(test_xy)

    target_ids = get_target_dates(s, e, interval_=target_interval)

    study_region = ee.Geometry.Rectangle([-112.5, 47.3, -111.5, 48.0])

    model_obj = Collection(
        collections=collections,
        start_date=s,
        end_date=e,
        geometry=study_region,
        cloud_cover_max=70)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']
    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_ids)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)

    interp = interpolated.toBands().rename(target_rename)
    out_name = 'interpolated'

    # task = ee.batch.Export.image.toAsset(
    #     image=interp,
    #     description='{}_{}'.format(out_name, year),
    #     assetId=os.path.join('users/dgketchum/IrrMapper/landsat', '{}_{}'.format(out_name, year)),
    #     region=study_region,
    #     scale=30,
    #     maxPixels=1e13)
    #
    # task.start()
    # print(out_name)


    def sample_data():
        raw = model_obj.scenes(variables_)
        i_ndvi = interpolated.filterBounds(study_region).select(['red', 'nir']).toBands()
        r_ndvi = raw.filterBounds(study_region).select(['red', 'nir']).toBands()

        interp_series = i_ndvi.sampleRegions(collection=fc,
                                             properties=['id'],
                                             scale=30,
                                             tileScale=16)

        raw_series = r_ndvi.sampleRegions(collection=fc,
                                          properties=['id'],
                                          scale=30,
                                          tileScale=16)

        for data, name in zip([raw_series, interp_series], ['raw_rnir', 'interp_rnir']):
            task = ee.batch.Export.table.toCloudStorage(
                interp_series,
                description='{}_test'.format(name),
                bucket='wudr',
                fileNamePrefix='{}_test'.format(name),
                fileFormat='CSV')

            task.start()
            print(name)

    sample_data()

    def get_example():
        image = ee.Image(interpolated.first().select(['ndvi'])).reproject(crs='EPSG: 4326', scale=100)
        image_url = image.getThumbURL({'min': -1.0, 'max': 1.0, 'palette': ndvi_palette,
                                       'region': study_region, 'dimensions': image_size})
# =======================================================================================
