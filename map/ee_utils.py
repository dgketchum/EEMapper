from datetime import datetime, timedelta

import ee


def add_doy(image):
    """ Add day-of-year image """
    mask = ee.Date(image.get('system:time_start'))
    day = ee.Image.constant(image.date().getRelative('day', 'year')).clip(image.geometry())
    i = image.addBands(day.rename('DOY')).int().updateMask(mask)
    return i


def get_world_climate(proj, months, param='prec'):
    if months[0] > months[1]:
        months = [x for x in range(months[0], 13)] + [x for x in range(1, months[1] + 1)]
    else:
        months = [x for x in range(months[0], months[1] + 1)]
    months = [str(x).zfill(2) for x in months]
    assert param in ['tavg', 'prec']
    combinations = [(m, param) for m in months]

    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).
             select(param).resample('bilinear').reproject(crs=proj['crs'], scale=30) for m, p in combinations]
    i = ee.ImageCollection(l)
    if param == 'prec':
        i = i.sum()
    else:
        i = i.mean()
    return i


def daily_landsat(year, roi):
    start = '{}-01-01'.format(year)
    end_date = '{}-01-01'.format(year + 1)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
        ee.FeatureCollection(roi).geometry()).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
        ee.FeatureCollection(roi).geometry()).filterDate(start, end_date).map(ls57mask)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
        ee.FeatureCollection(roi).geometry()).filterDate(start, end_date).map(ls8mask)

    ls_sr_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))

    d1 = datetime(year, 1, 1)
    d2 = datetime(year + 1, 1, 1)
    d_times = [(d1 + timedelta(days=x), d1 + timedelta(days=x + 1)) for x in range((d2 - d1).days)]
    date_tups = [(x.strftime('%Y-%m-%d'), y.strftime('%Y-%m-%d')) for x, y in d_times]
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    l, empty = [], []
    final = False
    for s, e in date_tups:
        if s == '{}-12-31'.format(year):
            e = '{}-01-01'.format(year + 1)
            final = True
        dt = datetime.strptime(s, '%Y-%m-%d')
        doy = dt.strftime('%j')
        rename_bands = ['{}{}{}'.format(year, doy, b) for b in bands]
        b = ls_sr_masked.filterDate(s, e).mosaic().rename(rename_bands)

        try:
            _ = b.getInfo()['bands'][0]
        except IndexError:
            empty.append(s)
            continue

        b = b.unmask(-99)
        l.append(b)
        if final:
            break

    print('{} empty dates : {}'.format(len(empty), empty))
    i = ee.Image(l)
    return i


def ls57mask(img):
    sr_bands = img.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_sel = img_masked.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                                 ['B2', 'B3', 'B4', 'B5', 'B6', 'B10', 'B7'])
    mask_mult = mask_sel.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def ls8mask(img):
    sr_bands = img.select('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_mult = img_masked.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def ls5_edge_removal(lsImage):
    inner_buffer = lsImage.geometry().buffer(-3000)
    buffer = lsImage.clip(inner_buffer)
    return buffer


def landsat_masked(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l4_coll = ee.ImageCollection('LANDSAT/LT04/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls57mask)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls8mask)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll).merge(l4_coll))
    return lsSR_masked


def landsat_composites(year, start, end, roi, append_name, composites_only=False):
    start_year = datetime.strptime(start, '%Y-%m-%d').year
    if start_year != year:
        year = start_year

    def evi_(x):
        return x.expression('2.5 * ((NIR-RED) / (NIR + 6 * RED - 7.5* BLUE +1))', {'NIR': x.select('B5'),
                                                                                   'RED': x.select('B4'),
                                                                                   'BLUE': x.select('B2')})

    def gi_(x):
        return x.expression('NIR / GREEN', {'NIR': x.select('B5'),
                                            'GREEN': x.select('B3')})

    bands_means = None
    lsSR_masked = landsat_masked(year, roi)
    if not composites_only:
        bands_means = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'],
                               ['B2_{}'.format(append_name),
                                'B3_{}'.format(append_name),
                                'B4_{}'.format(append_name),
                                'B5_{}'.format(append_name),
                                'B6_{}'.format(append_name),
                                'B7_{}'.format(append_name),
                                'B10_{}'.format(append_name)]
                               )).mean())
    # TODO - doy of max ndvi, growing season mean (or at least period nd means)
    if append_name in ['m2', 'm1', 'gs']:
        ndvi_mx = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_max_{}'.format(append_name))

        ndvi_mean = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).mean()).rename('nd_mean_{}'.format(append_name))

        ndvi = lsSR_masked.filterDate(start, end).filter(ee.Filter.dayOfYear(121, 200)) \
            .map(lambda x: x.select().addBands(x.normalizedDifference(['B5', 'B4'])))

        def add_doy(i):
            doy = i.date().getRelative('day', 'year')
            doyBand = ee.Image.constant(doy).uint16().rename('doy')
            doyBand = doyBand.updateMask(i.select('nd').mask())
            return i.addBands(doyBand)

        nd_doy = ndvi.map(add_doy)

        def add_quality_mosic(band):
            _max = nd_doy.qualityMosaic(band)
            doy = _max.select(['doy']).rename('doy_{}'.format(append_name))
            return doy

        ndvi_doy = add_quality_mosic('nd')

        ndvi = ndvi_mx.addBands([ndvi_mean, ndvi_doy])

    else:
        ndvi = ee.Image(lsSR_masked.filterDate(start, end).map(
            lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_{}'.format(append_name))

    ndwi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.normalizedDifference(['B5', 'B6'])).max()).rename('nw_{}'.format(append_name))
    evi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: evi_(x)).max()).rename('evi_{}'.format(append_name))
    gi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: gi_(x)).max()).rename('gi_{}'.format(append_name))

    if composites_only:
        bands = ndvi.addBands([ndwi, evi, gi])
    else:
        bands = bands_means.addBands([ndvi, ndwi, evi, gi])

    return bands


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
