"""Assembly of the annual feature stack the classifier consumes.

stack_bands() builds the Landsat seasonal composites, GRIDMET/WorldClim
climate, terrain, and ancillary bands for one year and ROI; its serialized
graph is locked by golden fixtures (tests/test_stack_bands_graph.py).
"""

import sys

import ee

from irrmapper.ingest.cdl import get_cdl
from irrmapper.ingest.climate import get_world_climate
from irrmapper.ingest.landsat import landsat_composites

# deep EE expression graphs exceed the default recursion limit on serialize
sys.setrecursionlimit(2000)


def get_alpha_earth_bands(yr, roi):
    dataset = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterDate(f'{yr}-01-01', f'{yr}-12-31').filterBounds(roi).mosaic())
    dataset = dataset.clip(roi)
    return dataset


def stack_bands(yr, roi, southern=False):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """

    water_year_start = '{}-10-01'.format(yr - 1)

    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    prev_s, prev_e = '{}-05-01'.format(yr - 1), '{}-09-30'.format(yr - 1),
    p_summer_s, p_summer_e = '{}-07-15'.format(yr - 1), '{}-09-30'.format(yr - 1)

    pprev_s, pprev_e = '{}-05-01'.format(yr - 2), '{}-09-30'.format(yr - 2),
    pp_summer_s, pp_summer_e = '{}-07-15'.format(yr - 2), '{}-09-30'.format(yr - 2)

    if southern:
        periods = [('gs', winter_s, fall_e),
                   ('1', winter_s, spring_e),
                   ('2', late_spring_s, late_spring_e),
                   ('3', summer_s, summer_e),
                   # modify to run in September
                   # ('4', fall_s, fall_e),

                   ('m1', prev_s, prev_e),
                   ('3_m1', p_summer_s, p_summer_e),

                   ('m2', pprev_s, pprev_e),
                   ('3_m2', pp_summer_s, pp_summer_e)]
    else:
        periods = [('gs', spring_e, fall_s),
                   ('1', spring_s, spring_e),
                   ('2', late_spring_s, late_spring_e),
                   ('3', summer_s, summer_e),
                   # modify to run in September
                   ('4', fall_s, fall_e),

                   ('m1', prev_s, prev_e),
                   ('3_m1', p_summer_s, p_summer_e),

                   ('m2', pprev_s, pprev_e),
                   ('3_m2', pp_summer_s, pp_summer_e)]

    first = True
    for name, start, end in periods:
        prev = 'm' in name
        bands = landsat_composites(yr, start, end, roi, name, composites_only=prev)
        if first:
            input_bands = bands
            proj = bands.select('B2_gs').projection().getInfo()
            first = False
        else:
            input_bands = input_bands.addBands(bands)

    integrated_composite_bands = []

    for feat in ['nd', 'gi', 'nw', 'evi']:
        # modify to run in September
        # periods = [x for x in range(2, 5)]
        periods = [x for x in range(2, 4)]
        c_bands = ['{}_{}'.format(feat, p) for p in periods]
        b = input_bands.select(c_bands).reduce(ee.Reducer.sum()).rename('{}_int'.format(feat))

        integrated_composite_bands.append(b)

    input_bands = input_bands.addBands(integrated_composite_bands)

    for s, e, n, m in [(spring_s, late_spring_e, 'spr', (3, 8)),
                       (water_year_start, spring_e, 'wy_spr', (10, 5)),
                       (water_year_start, summer_e, 'wy_smr', (10, 9))]:
        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
            roi).filterDate(s, e).select('pr', 'eto', 'tmmn', 'tmmx')

        temp = ee.Image(gridmet.select('tmmn').mean().add(gridmet.select('tmmx').mean()
                                                          .divide(ee.Number(2))).rename('tmp_{}'.format(n)))
        temp = temp.resample('bilinear').reproject(crs=proj['crs'], scale=30)

        ai_sum = gridmet.select('pr', 'eto').reduce(ee.Reducer.sum()).rename(
            'prec_tot_{}'.format(n), 'pet_tot_{}'.format(n)).resample('bilinear').reproject(crs=proj['crs'],
                                                                                            scale=30)
        wd_estimate = ai_sum.select('prec_tot_{}'.format(n)).subtract(ai_sum.select(
            'pet_tot_{}'.format(n))).rename('cwd_{}'.format(n))

        worldclim_prec = get_world_climate(proj=proj, months=m, param='prec')
        anom_prec = ai_sum.select('prec_tot_{}'.format(n)).subtract(worldclim_prec)
        worldclim_temp = get_world_climate(proj=proj, months=m, param='tavg')
        anom_temp = temp.subtract(worldclim_temp).rename('an_temp_{}'.format(n))

        input_bands = input_bands.addBands([temp, ai_sum, wd_estimate, anom_temp, anom_prec])

    coords = ee.Image.pixelLonLat().rename(['Lon_GCS', 'LAT_GCS']).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                  scale=30)
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect').reduceResolution(
        ee.Reducer.mean()).reproject(crs=proj['crs'], scale=30)

    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    input_bands = input_bands.addBands([coords, terrain, tpi_1250, tpi_250, tpi_150, anom_prec, anom_temp])

    nlcd = ee.Image('USGS/NLCD/NLCD2011').select('landcover').reproject(crs=proj['crs'], scale=30).rename('nlcd')

    cdl_cult, cdl_crop, cdl_simple = get_cdl(yr)

    gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
    occ_pos = gsw.select('occurrence').gt(0)
    water = occ_pos.unmask(0).rename('gsw')

    input_bands = input_bands.addBands([nlcd, cdl_cult, cdl_crop, cdl_simple, water])

    input_bands = input_bands.clip(roi)

    standard_names = []
    temp_ct = 1
    prec_ct = 1
    names = input_bands.bandNames().getInfo()
    for name in names:
        if 'tavg' in name and 'tavg' in standard_names:
            standard_names.append('tavg_{}'.format(temp_ct))
            temp_ct += 1
        elif 'prec' in name and 'prec' in standard_names:
            standard_names.append('prec_{}'.format(prec_ct))
            prec_ct += 1
        elif 'nd_cy' in name:
            standard_names.append('nd_max_cy')
        else:
            standard_names.append(name)

    input_bands = input_bands.rename(standard_names)
    return input_bands
