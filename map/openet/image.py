import ee

from map.openet import landsat
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

    def __init__(self, image,):

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

        # Image projection and geotransform
        self.crs = image.projection().crs()
        self.transform = ee.List(ee.Dictionary(
            ee.Algorithms.Describe(image.projection())).get('transform'))

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
            elif v.lower() == 'mask':
                output_images.append(self.mask)
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
        return self.mask.double().multiply(0).add(utils.date_to_time_0utc(self._date))\
            .rename(['time']).set(self._properties)

    @lazy_property
    def mask(self):
        """Mask of all active pixels (based on the final et_fraction)"""
        return self.ndvi.multiply(0).add(1).updateMask(1) \
            .rename(['mask']).set(self._properties).uint8()

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
            landsat.tir(prep_image),
        ])

        # Apply the cloud mask and add properties
        input_image = input_image \
            .set({'system:index': sr_image.get('system:index'),
                  'system:time_start': sr_image.get('system:time_start'),
                  'system:id': sr_image.get('system:id'),
            })

        _cls = cls(input_image, **kwargs)
        return _cls


if __name__ == '__main__':
    pass

# =========================================
