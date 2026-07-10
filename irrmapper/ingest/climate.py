import ee


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
