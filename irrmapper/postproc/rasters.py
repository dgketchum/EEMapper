"""Boolean/frequency GeoTIFF exports of the composited classifications to GCS."""

from datetime import datetime

import ee


def export_raster(irr_coll, roi=None, years=None, min_years=3, debug=False, state='WA',
                  export_freq=True):
    irr_min_yr_mask = None
    roi = ee.FeatureCollection(roi)
    geo = roi.geometry()

    irr_coll = ee.ImageCollection(irr_coll)

    start_yr, end_yr = years[0], years[-1]
    coll = irr_coll.filterDate(f'{start_yr}-01-01', f'{end_yr}-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))

    if min_years:
        irr_min_yr_mask = remap.sum().gte(min_years)

    if export_freq:
        if irr_min_yr_mask is not None:
            sum = remap.sum().mask(irr_min_yr_mask)
        else:
            sum = remap.sum()

        sum = sum.clip(geo).toInt()

        desc = 'irrmapper_{}_freq_{}_{}_{}'.format(state, start_yr, end_yr,
                                                   datetime.now().strftime('%d%b%Y').upper())
        task = ee.batch.Export.image.toCloudStorage(
            image=sum,
            description=desc,
            bucket='wudr',
            fileNamePrefix='irrmapper_{}/{}'.format(state, desc),
            region=geo,
            scale=30,
            maxPixels=1e13,
            crs='EPSG:5071',
            fileFormat='GeoTIFF')
        print(desc)
        task.start()

    for year in years:
        coll = irr_coll.filterDate(f'{year}-01-01', f'{year}-12-31').select('classification')
        if irr_min_yr_mask:
            remap = coll.map(lambda img: img.lt(1)).mosaic().mask(irr_min_yr_mask).toInt()
        else:
            remap = coll.map(lambda img: img.lt(1)).mosaic().toInt()
        remap = remap.clip(geo)

        if debug:
            pt = ee.FeatureCollection(ee.Geometry.Point([-120.5, 47.0]))
            data = remap.sampleRegions(collection=pt, scale=30)
            data = data.getInfo()

        desc = 'irrmapper_{}_{}'.format(state, year)
        task = ee.batch.Export.image.toCloudStorage(
            image=remap,
            description=desc,
            bucket='wudr',
            fileNamePrefix='irrmapper_{}/{}'.format(state, desc),
            region=geo,
            scale=30,
            maxPixels=1e13,
            crs='EPSG:5071',
            fileFormat='GeoTIFF')
        print(desc)
        task.start()
