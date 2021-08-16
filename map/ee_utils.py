import ee

from datetime import datetime, timedelta


def add_doy(image):
    """ Add day-of-year image """
    mask = ee.Date(image.get('system:time_start'))
    day = ee.Image.constant(image.date().getRelative('day', 'year')).clip(image.geometry())
    i = image.addBands(day.rename('DOY')).int().updateMask(mask)
    return i


def get_world_climate(proj):
    n = list(range(1, 13))
    months = [str(x).zfill(2) for x in n]
    parameters = ['tavg', 'tmin', 'tmax', 'prec']
    combinations = [(m, p) for m in months for p in parameters]

    l = [ee.Image('WORLDCLIM/V1/MONTHLY/{}'.format(m)).select(p).resample('bilinear').reproject(crs=proj['crs'],
                                                                                                scale=30) for m, p in
         combinations]
    # not sure how to do this without initializing the image with a constant
    i = ee.Image(l)
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
    sr_bands = img.select('B1', 'B2', 'B3', 'B4', 'B5', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_sel = img_masked.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'], ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    mask_mult = mask_sel.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def ls8mask(img):
    sr_bands = img.select('B2', 'B3', 'B4', 'B5', 'B6', 'B7')
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


def landsat_composites(year, start, end, roi, append_name):
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

    lsSR_masked = landsat_masked(year, roi)
    bands_means = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                           ['B2_{}'.format(append_name),
                            'B3_{}'.format(append_name),
                            'B4_{}'.format(append_name),
                            'B5_{}'.format(append_name),
                            'B6_{}'.format(append_name),
                            'B7_{}'.format(append_name)]
                           )).mean())
    ndvi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.normalizedDifference(['B5', 'B4'])).max()).rename('nd_{}'.format(append_name))
    ndwi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.normalizedDifference(['B5', 'B6'])).max()).rename('nw_{}'.format(append_name))
    evi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: evi_(x)).max()).rename('evi_{}'.format(append_name))
    gi = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: gi_(x)).max()).rename('gi_{}'.format(append_name))
    bands = bands_means.addBands([ndvi, ndwi, evi, gi])

    return bands


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
