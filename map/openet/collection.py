import copy
import datetime
import sys
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

    def interpolate(self, variables, interp_days=32):

        # The start/end date for the interpolation include more days
        # (+/- interp_days) than are included in the reference ET collection
        interp_start_dt = self.start_date - timedelta(days=interp_days)
        interp_end_dt = self.end_date + timedelta(days=interp_days)

        daily_et_ref_coll_id = 'projects/climate-engine/cimis/daily'
        daily_et_ref_coll = ee.ImageCollection(daily_et_ref_coll_id) \
            .filterDate(self.start_date, self.end_date) \
            .select(['ETr_ASCE'], ['et_reference'])

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
            target_coll=daily_et_ref_coll,
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

            image_list = []

            image_list.append(et_reference_img.float())
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


if __name__ == '__main__':
    pass
# =======================================================================================
