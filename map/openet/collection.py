import copy
import datetime
from pprint import pprint

from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
from dateutil.relativedelta import relativedelta
import ee

from map.openet import utils
from map.openet.image import Image
from IPython.display import Image as Img

import map.openet.inerpolate as interp


# Importing to get version number, is there a better way?


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


class Collection():
    """"""

    def __init__(
            self,
            collections,
            start_date,
            end_date,
            geometry,
            cloud_cover_max=70,
            model_args=None):

        self.collections = collections
        self.start_date = start_date
        self.end_date = end_date
        self.start_str = self.start_date.strftime('%Y-%m-%d')
        self.end_str = self.end_date.strftime('%Y-%m-%d')

        self.geometry = geometry
        self.cloud_cover_max = cloud_cover_max
        self.model_args = model_args
        self._interp_vars = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir', 'ndvi']

        # If collections is a string, place in a list
        if type(self.collections) is str:
            self.collections = [self.collections]

    def _build(self):
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
                model_object = Image.from_landsat_c1_sr(
                    sr_image=ee.Image(image))
                return model_object.calculate(variables)

            variable_coll = variable_coll.merge(
                ee.ImageCollection(input_coll.map(compute_lsr)))

        return variable_coll

    def interpolate(self, target, variables, interp_days=32):

        # The start/end date for the interpolation include more days
        # (+/- interp_days) than are included in the reference ET collection
        interp_start_dt = self.start_date - timedelta(days=interp_days)
        interp_end_dt = self.end_date + timedelta(days=interp_days)

        interp_vars = [band for band in variables]

        # Count will be determined using the aggregate_coll image masks
        if 'count' in variables:
            interp_vars.append('mask')
        # interp_vars.append('time')

        # Build initial scene image collection
        scene_coll = self._build()

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
            target_coll=target,
            source_coll=scene_coll.select(interp_vars), interp_days=interp_days)

        interp_properties = {
            'cloud_cover_max': self.cloud_cover_max,
            'collections': ', '.join(self.collections),
            'interp_days': interp_days,
            'interp_method': 'linear',
            'model_name': 'IrrMapper'}

        def aggregate_image(agg_start_date, agg_end_date, date_format):

            image_list = []
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

        return ee.ImageCollection(daily_coll.map(aggregate_daily))

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


def build_target(dates):

    def time(i, d):
        return i.double().multiply(0).add(d).rename(['time'])

    images = [ee.Image(ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select('pr').first()) for i, _ in enumerate(dates)]
    dates = [ee.Date.fromYMD(x.year, x.month, x.day).millis() for x in dates]
    images = [i.addBands([time(i, d)]) for i, d in zip(images, dates)]
    coll_ = ee.ImageCollection.fromImages(images)
    return coll_


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
    s = datetime(year, 1, 1)
    e = datetime(year + 1, 1, 1)
    d_times = [d for d in rrule(dtstart=s, until=e, interval=15, freq=DAILY)]
    # d_strings = [(x.strftime('%Y-%m-%d'), y.strftime('%Y-%m-%d'), x) for x, y in d_times]

    cloud_cover = 100
    interp_days = 32
    test_xy = [-121.5265, 38.7399]
    test_point = ee.Geometry.Point(test_xy)

    # study_area = ee.Geometry.Rectangle(-122.00, 38.60, -121.00, 39.0)
    study_area = ee.Geometry.Rectangle(
        test_xy[0] - 0.08, test_xy[1] - 0.04,
        test_xy[0] + 0.08, test_xy[1] + 0.04)
    study_region = study_area.bounds(1, 'EPSG:4326')
    study_crs = 'EPSG:32610'

    target = build_target(d_times)

    model_obj = Collection(
        collections=collections,
        start_date=s,
        end_date=e,
        geometry=test_point,
        cloud_cover_max=70)

    variables = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir', 'ndvi']
    daily_coll = model_obj.interpolate(target,
                                       variables=variables,
                                       interp_days=interp_days)

    image = ee.Image(daily_coll.select(['ndvi']).first()).reproject(crs=study_crs, scale=100)
    image_url = image.getThumbURL({'min': -1.0, 'max': 1.0, 'palette': ndvi_palette,
                      'region': study_region, 'dimensions': image_size})
    Img(image_url, embed=True, format='png')

# =======================================================================================
