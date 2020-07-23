import datetime
import pprint

import ee

from map.openet import landsat
from map.openet import model
from map.openet import utils
import map.openet.common as common


PROJECT_FOLDER = 'projects/earthengine-legacy/assets/projects/usgs-ssebop'
# PROJECT_FOLDER = 'projects/usgs-ssebop'


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


class Image():
    """Earth Engine based SSEBop Image"""

    def __init__(
            self, image,
            et_reference_source=None,
            et_reference_band=None,
            et_reference_factor=None,
            et_reference_resample=None,
            dt_source='DAYMET_MEDIAN_V0',
            elev_source='SRTM',
            tcorr_source='DYNAMIC',
            tmax_source='DAYMET_MEDIAN_V2',
            elr_flag=False,
            dt_min=6,
            dt_max=25,
            et_fraction_type='alfalfa',
            **kwargs,
        ):
        """Construct a generic SSEBop Image

        Parameters
        ----------
        image : ee.Image
            A "prepped" SSEBop input image.
            Image must have bands: "ndvi" and "lst".
            Image must have properties: 'system:id', 'system:index', and
                'system:time_start'.
        et_reference_source : str, float, optional
            Reference ET source (the default is None).
            Parameter is required if computing 'et' or 'et_reference'.
        et_reference_band : str, optional
            Reference ET band name (the default is None).
            Parameter is required if computing 'et' or 'et_reference'.
        et_reference_factor : float, None, optional
            Reference ET scaling factor.  The default is None which is
            equivalent to 1.0 (or no scaling).
        et_reference_resample : {'nearest', 'bilinear', 'bicubic', None}, optional
            Reference ET resampling.  The default is None which is equivalent
            to nearest neighbor resampling.
        dt_source : {'DAYMET_MEDIAN_V0', 'DAYMET_MEDIAN_V1', or float}, optional
            dT source keyword (the default is 'DAYMET_MEDIAN_V1').
        elev_source : {'ASSET', 'GTOPO', 'NED', 'SRTM', or float}, optional
            Elevation source keyword (the default is 'SRTM').
        tcorr_source : {'DYNAMIC',
                        'SCENE', 'SCENE_DAILY', 'SCENE_MONTHLY',
                        'SCENE_ANNUAL', 'SCENE_DEFAULT', or float}, optional
            Tcorr source keyword (the default is 'DYNAMIC').
        tmax_source : {'CIMIS', 'DAYMET', 'GRIDMET', 'DAYMET_MEDIAN_V2',
                       'TOPOWX_MEDIAN_V0', or float}, optional
            Maximum air temperature source (the default is 'TOPOWX_MEDIAN_V0').
        elr_flag : bool, str, optional
            If True, apply Elevation Lapse Rate (ELR) adjustment
            (the default is False).
        dt_min : float, optional
            Minimum allowable dT [K] (the default is 6).
        dt_max : float, optional
            Maximum allowable dT [K] (the default is 25).
        et_fraction_type : {'alfalfa', 'grass'}, optional
            ET fraction  (the default is 'alfalfa').
        kwargs : dict, optional
            tmax_resample : {'nearest', 'bilinear'}
            dt_resample : {'nearest', 'bilinear'}

        Notes
        -----
        Input image must have a Landsat style 'system:index' in order to
        lookup Tcorr value from table asset.  (i.e. LC08_043033_20150805)

        """
        self.image = ee.Image(image)

        # Set as "lazy_property" below in order to return custom properties
        # self.lst = self.image.select('lst')
        # self.ndvi = self.image.select('ndvi')

        # Copy system properties
        self._id = self.image.get('system:id')
        self._index = self.image.get('system:index')
        self._time_start = self.image.get('system:time_start')
        self._properties = {
            'system:index': self._index,
            'system:time_start': self._time_start,
            'image_id': self._id,
        }

        # Build SCENE_ID from the (possibly merged) system:index
        scene_id = ee.List(ee.String(self._index).split('_')).slice(-3)
        self._scene_id = ee.String(scene_id.get(0)).cat('_')\
            .cat(ee.String(scene_id.get(1))).cat('_')\
            .cat(ee.String(scene_id.get(2)))

        # Build WRS2_TILE from the scene_id
        self._wrs2_tile = ee.String('p').cat(self._scene_id.slice(5, 8))\
            .cat('r').cat(self._scene_id.slice(8, 11))

        # Set server side date/time properties using the 'system:time_start'
        self._date = ee.Date(self._time_start)
        self._year = ee.Number(self._date.get('year'))
        self._month = ee.Number(self._date.get('month'))
        self._start_date = ee.Date(utils.date_to_time_0utc(self._date))
        self._end_date = self._start_date.advance(1, 'day')
        self._doy = ee.Number(self._date.getRelative('day', 'year')).add(1).int()
        self._cycle_day = self._start_date.difference(
            ee.Date.fromYMD(1970, 1, 3), 'day').mod(8).add(1).int()

        # Reference ET parameters
        self.et_reference_source = et_reference_source
        self.et_reference_band = et_reference_band
        self.et_reference_factor = et_reference_factor
        self.et_reference_resample = et_reference_resample

        # Check reference ET parameters
        if et_reference_factor and not utils.is_number(et_reference_factor):
            raise ValueError('et_reference_factor must be a number')
        if et_reference_factor and self.et_reference_factor < 0:
            raise ValueError('et_reference_factor must be greater than zero')
        et_reference_resample_methods = ['nearest', 'bilinear', 'bicubic']
        if (et_reference_resample and
                et_reference_resample.lower() not in et_reference_resample_methods):
            raise ValueError('unsupported et_reference_resample method')

        # Model input parameters
        self._dt_source = dt_source
        self._elev_source = elev_source
        self._tcorr_source = tcorr_source
        self._tmax_source = tmax_source
        self._elr_flag = elr_flag
        self._dt_min = float(dt_min)
        self._dt_max = float(dt_max)

        # Convert elr_flag from string to bool if necessary
        if type(self._elr_flag) is str:
            if self._elr_flag.upper() in ['TRUE']:
                self._elr_flag = True
            elif self._elr_flag.upper() in ['FALSE']:
                self._elr_flag = False
            else:
                raise ValueError('elr_flag "{}" could not be interpreted as '
                                 'bool'.format(self._elr_flag))

        # Image projection and geotransform
        self.crs = image.projection().crs()
        self.transform = ee.List(ee.Dictionary(
            ee.Algorithms.Describe(image.projection())).get('transform'))
        # self.crs = image.select([0]).projection().getInfo()['crs']
        # self.transform = image.select([0]).projection().getInfo()['transform']

        # Set the resample method as properties so they can be modified
        if 'dt_resample' in kwargs.keys():
            self._dt_resample = kwargs['dt_resample'].lower()
        else:
            self._dt_resample = 'bilinear'
        if 'tmax_resample' in kwargs.keys():
            self._tmax_resample = kwargs['tmax_resample'].lower()
        else:
            self._tmax_resample = 'bilinear'

        if et_fraction_type.lower() not in ['alfalfa', 'grass']:
            raise ValueError('et_fraction_type must "alfalfa" or "grass"')
        self.et_fraction_type = et_fraction_type.lower()
        # CGM - Should et_fraction_type be set as a kwarg instead?
        # if 'et_fraction_type' in kwargs.keys():
        #     self.et_fraction_type = kwargs['et_fraction_type'].lower()
        # else:
        #     self.et_fraction_type = 'alfalfa'

    def calculate(self, variables=[]):
        """Return a multiband image of calculated variables

        Parameters
        ----------
        variables : list

        Returns
        -------
        ee.Image

        """
        output_images = []
        for v in variables:
            if v.lower() == 'red':
                output_images.append(self.red.float())
            elif v.lower() == 'blue':
                output_images.append(self.blue.float())
            elif v.lower() == 'green':
                output_images.append(self.green.float())
            elif v.lower() == 'nir':
                output_images.append(self.nir.float())
            elif v.lower() == 'swir1':
                output_images.append(self.swir1.float())
            elif v.lower() == 'swir2':
                output_images.append(self.swir2.float())
            elif v.lower() == 'tir':
                output_images.append(self.tir.float())
            elif v.lower() == 'ndvi':
                output_images.append(self.ndvi.float())
            elif v.lower() == 'time':
                output_images.append(self.time)
            else:
                raise ValueError('unsupported variable: {}'.format(v))

        return ee.Image(output_images).set(self._properties)

    @lazy_property
    def ndvi(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['ndvi']).set(self._properties)

    @lazy_property
    def red(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['red']).set(self._properties)

    @lazy_property
    def blue(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['blue']).set(self._properties)

    @lazy_property
    def green(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['green']).set(self._properties)

    @lazy_property
    def nir(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['nir']).set(self._properties)

    @lazy_property
    def swir1(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['swir1']).set(self._properties)

    @lazy_property
    def swir2(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['swir2']).set(self._properties)

    @lazy_property
    def tir(self):
        """Input normalized difference vegetation index (NDVI)"""
        return self.image.select(['tir']).set(self._properties)

    @lazy_property
    def time(self):
        """Return an image of the 0 UTC time (in milliseconds)"""
        return self.mask\
            .double().multiply(0).add(utils.date_to_time_0utc(self._date))\
            .rename(['time']).set(self._properties)

    @lazy_property
    def dt(self):
        """

        Returns
        -------
        ee.Image

        Raises
        ------
        ValueError
            If `self._dt_source` is not supported.

        """
        if utils.is_number(self._dt_source):
            dt_img = ee.Image.constant(float(self._dt_source))
        # Use precomputed dT median assets
        elif self._dt_source.upper() == 'DAYMET_MEDIAN_V0':
            dt_coll = ee.ImageCollection(PROJECT_FOLDER + '/dt/daymet_median_v0')\
                .filter(ee.Filter.calendarRange(self._doy, self._doy, 'day_of_year'))
            dt_img = ee.Image(dt_coll.first())
        elif self._dt_source.upper() == 'DAYMET_MEDIAN_V1':
            dt_coll = ee.ImageCollection(PROJECT_FOLDER + '/dt/daymet_median_v1')\
                .filter(ee.Filter.calendarRange(self._doy, self._doy, 'day_of_year'))
            dt_img = ee.Image(dt_coll.first())
        # Compute dT for the target date
        elif self._dt_source.upper() == 'CIMIS':
            input_img = ee.Image(
                ee.ImageCollection('projects/earthengine-legacy/assets/'
                                   'projects/climate-engine/cimis/daily')\
                    .filterDate(self._start_date, self._end_date)\
                    .select(['Tx', 'Tn', 'Rs', 'Tdew'])
                    .first())
            # Convert units to T [K], Rs [MJ m-2 d-1], ea [kPa]
            # Compute Ea from Tdew
            dt_img = model.dt(
                tmax=input_img.select(['Tx']).add(273.15),
                tmin=input_img.select(['Tn']).add(273.15),
                rs=input_img.select(['Rs']),
                ea=input_img.select(['Tdew']).add(237.3).pow(-1)
                    .multiply(input_img.select(['Tdew']))\
                    .multiply(17.27).exp().multiply(0.6108).rename(['ea']),
                elev=self.elev,
                doy=self._doy)
        elif self._dt_source.upper() == 'DAYMET':
            input_img = ee.Image(
                ee.ImageCollection('NASA/ORNL/DAYMET_V3')\
                    .filterDate(self._start_date, self._end_date)\
                    .select(['tmax', 'tmin', 'srad', 'dayl', 'vp'])
                    .first())
            # Convert units to T [K], Rs [MJ m-2 d-1], ea [kPa]
            # Solar unit conversion from DAYMET documentation:
            #   https://daymet.ornl.gov/overview.html
            dt_img = model.dt(
                tmax=input_img.select(['tmax']).add(273.15),
                tmin=input_img.select(['tmin']).add(273.15),
                rs=input_img.select(['srad'])\
                    .multiply(input_img.select(['dayl'])).divide(1000000),
                ea=input_img.select(['vp'], ['ea']).divide(1000),
                elev=self.elev,
                doy=self._doy)
        elif self._dt_source.upper() == 'GRIDMET':
            input_img = ee.Image(
                ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')\
                    .filterDate(self._start_date, self._end_date)\
                    .select(['tmmx', 'tmmn', 'srad', 'sph'])
                    .first())
            # Convert units to T [K], Rs [MJ m-2 d-1], ea [kPa]
            q = input_img.select(['sph'], ['q'])
            pair = self.elev.multiply(-0.0065).add(293.0).divide(293.0).pow(5.26)\
                .multiply(101.3)
            # pair = self.elev.expression(
            #     '101.3 * pow((293.0 - 0.0065 * elev) / 293.0, 5.26)',
            #     {'elev': self.elev})
            dt_img = model.dt(
                tmax=input_img.select(['tmmx']),
                tmin=input_img.select(['tmmn']),
                rs=input_img.select(['srad']).multiply(0.0864),
                ea=q.multiply(0.378).add(0.622).pow(-1).multiply(q)\
                    .multiply(pair).rename(['ea']),
                elev=self.elev,
                doy=self._doy)
        else:
            raise ValueError('Invalid dt_source: {}\n'.format(self._dt_source))

        if (self._dt_resample and
                self._dt_resample.lower() in ['bilinear', 'bicubic']):
            dt_img = dt_img.resample(self._dt_resample)
        # TODO: A reproject call may be needed here also
        # dt_img = dt_img.reproject(self.crs, self.transform)

        return dt_img.clamp(self._dt_min, self._dt_max).rename('dt')

    @lazy_property
    def elev(self):
        """Elevation [m]

        Returns
        -------
        ee.Image

        Raises
        ------
        ValueError
            If `self._elev_source` is not supported.

        """
        if utils.is_number(self._elev_source):
            elev_image = ee.Image.constant(float(self._elev_source))
        elif self._elev_source.upper() == 'ASSET':
            elev_image = ee.Image(PROJECT_FOLDER + '/srtm_1km')
        elif self._elev_source.upper() == 'GTOPO':
            elev_image = ee.Image('USGS/GTOPO30')
        elif self._elev_source.upper() == 'NED':
            elev_image = ee.Image('USGS/NED')
        elif self._elev_source.upper() == 'SRTM':
            elev_image = ee.Image('USGS/SRTMGL1_003')
        elif (self._elev_source.lower().startswith('projects/') or
              self._elev_source.lower().startswith('users/')):
            elev_image = ee.Image(self._elev_source)
        else:
            raise ValueError('Unsupported elev_source: {}\n'.format(
                self._elev_source))

        return elev_image.select([0], ['elev'])

    @classmethod
    def from_landsat_c1_toa(cls, toa_image, cloudmask_args={}, **kwargs):
        """Returns a SSEBop Image instance from a Landsat Collection 1 TOA image

        Parameters
        ----------
        toa_image : ee.Image, str
            A raw Landsat Collection 1 TOA image or image ID.
        cloudmask_args : dict
            keyword arguments to pass through to cloud mask function
        kwargs : dict
            Keyword arguments to pass through to Image init function

        Returns
        -------
        Image

        """
        toa_image = ee.Image(toa_image)

        # Use the SPACECRAFT_ID property identify each Landsat type
        spacecraft_id = ee.String(toa_image.get('SPACECRAFT_ID'))

        # Rename bands to generic names
        # Rename thermal band "k" coefficients to generic names
        input_bands = ee.Dictionary({
            'LANDSAT_4': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'BQA'],
            'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'BQA'],
            'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6_VCID_1',
                          'BQA'],
            'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'BQA'],
        })
        output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir',
                        'BQA']
        k1 = ee.Dictionary({
            'LANDSAT_4': 'K1_CONSTANT_BAND_6',
            'LANDSAT_5': 'K1_CONSTANT_BAND_6',
            'LANDSAT_7': 'K1_CONSTANT_BAND_6_VCID_1',
            'LANDSAT_8': 'K1_CONSTANT_BAND_10',
        })
        k2 = ee.Dictionary({
            'LANDSAT_4': 'K2_CONSTANT_BAND_6',
            'LANDSAT_5': 'K2_CONSTANT_BAND_6',
            'LANDSAT_7': 'K2_CONSTANT_BAND_6_VCID_1',
            'LANDSAT_8': 'K2_CONSTANT_BAND_10',
        })
        prep_image = toa_image\
            .select(input_bands.get(spacecraft_id), output_bands)\
            .set({
                'k1_constant': ee.Number(toa_image.get(k1.get(spacecraft_id))),
                'k2_constant': ee.Number(toa_image.get(k2.get(spacecraft_id))),
                'SATELLITE': spacecraft_id,
            })

        # Build the input image
        input_image = ee.Image([
            landsat.tir(prep_image),
        ])

        # Apply the cloud mask and add properties
        input_image = input_image\
            .updateMask(common.landsat_c1_toa_cloud_mask(
                toa_image, **cloudmask_args))\
            .set({
                'system:index': toa_image.get('system:index'),
                'system:time_start': toa_image.get('system:time_start'),
                'system:id': toa_image.get('system:id'),
            })

        # Instantiate the class
        return cls(ee.Image(input_image), **kwargs)

    @classmethod
    def from_landsat_c1_sr(cls, sr_image, **kwargs):
        """Returns a SSEBop Image instance from a Landsat Collection 1 SR image

        Parameters
        ----------
        sr_image : ee.Image, str
            A raw Landsat Collection 1 SR image or image ID.

        Returns
        -------
        Image

        """
        sr_image = ee.Image(sr_image)

        # Use the SATELLITE property identify each Landsat type
        spacecraft_id = ee.String(sr_image.get('SATELLITE'))

        # Rename bands to generic names
        # Rename thermal band "k" coefficients to generic names
        input_bands = ee.Dictionary({
            'LANDSAT_4': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
            'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
            'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
            'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'pixel_qa'],
        })
        output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir',
                        'pixel_qa']
        # TODO: Follow up with Simon about adding K1/K2 to SR collection
        # Hardcode values for now
        k1 = ee.Dictionary({
            'LANDSAT_4': 607.76, 'LANDSAT_5': 607.76,
            'LANDSAT_7': 666.09, 'LANDSAT_8': 774.8853})
        k2 = ee.Dictionary({
            'LANDSAT_4': 1260.56, 'LANDSAT_5': 1260.56,
            'LANDSAT_7': 1282.71, 'LANDSAT_8': 1321.0789})
        prep_image = sr_image\
            .select(input_bands.get(spacecraft_id), output_bands)\
            .multiply([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.1, 1])\
            .set({'k1_constant': ee.Number(k1.get(spacecraft_id)),
                  'k2_constant': ee.Number(k2.get(spacecraft_id))})

        # Build the input image
        input_image = ee.Image([
            landsat.red(prep_image),
            landsat.blue(prep_image),
            landsat.green(prep_image),
            landsat.nir(prep_image),
            landsat.swir1(prep_image),
            landsat.swir2(prep_image),
            landsat.lst(prep_image),
            landsat.ndvi(prep_image),
        ])

        # Apply the cloud mask and add properties
        input_image = input_image\
            .updateMask(common.landsat_c1_sr_cloud_mask(sr_image))\
            .set({'system:index': sr_image.get('system:index'),
                  'system:time_start': sr_image.get('system:time_start'),
                  'system:id': sr_image.get('system:id'),
            })

        # Instantiate the class
        return cls(input_image, **kwargs)

    # # TODO: Move calculation to model.py
    # @lazy_property
    # def tcorr_image(self):
    #     """Compute Tcorr for the current image
    #
    #     Returns
    #     -------
    #     ee.Image of Tcorr values
    #
    #     """
    #     lst = ee.Image(self.lst)
    #     ndvi = ee.Image(self.ndvi)
    #     tmax = ee.Image(self.tmax)
    #
    #     # Compute Tcorr
    #     tcorr = lst.divide(tmax)
    #
    #     # Select high NDVI pixels that are also surrounded by high NDVI
    #     ndvi_smooth_mask = ndvi.focal_mean(radius=120, units='meters')\
    #       .reproject(crs=self.crs, crsTransform=self.transform)\
    #       .gt(0.7)
    #     ndvi_buffer_mask = ndvi.gt(0.7).reduceNeighborhood(
    #         ee.Reducer.min(), ee.Kernel.square(radius=60, units='meters'))
    #
    #     # Remove low LST and low NDVI
    #     tcorr_mask = lst.gt(270).And(ndvi_smooth_mask).And(ndvi_buffer_mask)
    #
    #     return tcorr.updateMask(tcorr_mask).rename(['tcorr'])\
    #         .set({'system:index': self._index,
    #               'system:time_start': self._time_start,
    #               'tmax_source': tmax.get('tmax_source'),
    #               'tmax_version': tmax.get('tmax_version')})
    #
    # @lazy_property
    # def tcorr_stats(self):
    #     """Compute the Tcorr 5th percentile and count statistics
    #
    #     Returns
    #     -------
    #     dictionary
    #
    #     """
    #     return ee.Image(self.tcorr_image).reduceRegion(
    #         reducer=ee.Reducer.percentile([5])\
    #             .combine(ee.Reducer.count(), '', True),
    #         crs=self.crs,
    #         crsTransform=self.transform,
    #         geometry=self.image.geometry().buffer(1000),
    #         bestEffort=False,
    #         maxPixels=2*10000*10000,
    #         tileScale=1,
    #     )


if __name__ == '__main__':
    pass

# =========================================
