import os
import json
import datetime

from pandas import read_csv, DataFrame
import numpy as np
import ee
import fiona
from shapely.geometry import shape

from state_county_codes import state_fips_code, county_acres, state_county_code

ee.Initialize()
BOUNDARIES = 'users/dgketchum/boundaries'

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
E_STATES = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']

BASIN = ['users/dgketchum/boundaries/umrb_ylstn_clip',
         'users/dgketchum/boundaries/CMB_RB_CLIP',
         'users/dgketchum/boundaries/CO_RB']


def confusion(irr_labels, unirr_labels, irr_image, unirr_image, state, clip=False):
    domain = 'users/dgketchum/boundaries/{}'.format(state)
    domain = ee.FeatureCollection(domain)
    domain = domain.toList(domain.size()).get(0)
    domain = ee.Feature(domain)

    true_positive = irr_image.eq(irr_labels)
    false_positive = irr_image.eq(unirr_labels)
    true_negative = unirr_image.eq(unirr_labels)
    false_negative = unirr_image.eq(irr_labels)

    if clip:
        basin = ee.FeatureCollection(BASIN[0]).merge(ee.FeatureCollection(BASIN[1])) \
            .merge(ee.FeatureCollection(BASIN[2]))

        true_positive = true_positive.clip(basin)
        false_positive = false_positive.clip(basin)
        true_negative = true_negative.clip(basin)
        false_negative = false_negative.clip(basin)

    TP = true_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FP = false_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FN = false_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    TN = true_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )

    out = {'TP': TP.getInfo(), 'FP': FP.getInfo(), 'FN': FN.getInfo(), 'TN': TN.getInfo()}
    return out


def create_lanid_labels(year, geo):
    begin = '{}-01-01'.format(year)
    end = '{}-12-31'.format(year)
    lanid = ee.ImageCollection('projects/openet/irrigated_area/LANID'
                               ).filterDate(begin, end).filterBounds(geo).first().select("irr_land")
    irr_mask = lanid.eq(1)
    unmasked = lanid.unmask(0)
    unirr_image = ee.Image(1).byte().updateMask(unmasked.Not())
    irr_image = ee.Image(1).byte().updateMask(irr_mask)
    return irr_image, unirr_image


def create_rf_labels(year, state_abv):
    rf = ee.Image('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp/{}_{}'.format(state_abv, year))
    irrMask = rf.lt(1)
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not())
    irrImage = ee.Image(1).byte().updateMask(irrMask)
    return irrImage, unirrImage


def create_irrigated_labels(all_data, year):
    if all_data:
        non_irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/dryland')
        fallow = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/fallow')
        irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))
    else:
        root = 'users/dgketchum/validation/'
        non_irrigated = ee.FeatureCollection(root + 'uncultivated')
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'dryland'))
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'wetlands'))

        fallow = ee.FeatureCollection(root + 'fallow')
        irrigated = ee.FeatureCollection(root + 'irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow).map(lambda x: x.buffer(-50))
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))

    irr_labels = ee.Image(1).byte().paint(irrigated, 0)
    irr_labels = irr_labels.updateMask(irr_labels.Not())
    unirr_labels = ee.Image(1).byte().paint(non_irrigated, 0)
    unirr_labels = unirr_labels.updateMask(unirr_labels.Not())

    return irr_labels, unirr_labels


def metrics(arr):
    recall = arr[0, 0] / (arr[0, 1] + arr[0, 0])
    precision = arr[0, 0] / (arr[1, 0] + arr[0, 0])
    return precision, recall


def calculate_accuracy_by_state(climate_json, csv_out):
    sdf = DataFrame(columns=['state', 'year', 'anomaly', 'normal',
                             'TP', 'FN', 'FP', 'TN', 'prec', 'rec'])
    with open(climate_json, 'r') as fp:
        stdct = json.load(fp)
    for k, v in stdct.items():
        for yr, tup_ in v.items():
            sdf = sdf.append({'state': k, 'year': yr, 'anomaly': tup_[1], 'normal': tup_[0]}, ignore_index=True)

    print(datetime.datetime.now())
    globe_conf = np.zeros((2, 2))

    for i, row in sdf.iterrows():
        if not row['state'] in ['MT', 'ID', 'WA', 'WY']:
            continue
        print('\n {} precip: {}'.format(row['year'], row['anomaly']))
        annual_conf = np.zeros((2, 2))
        try:
            irr_labels, unirr_labels = create_irrigated_labels(False, int(row['year']))
            irr_image, unirr_image = create_rf_labels(row['year'], state_abv=row['state'])
            cmt = confusion(irr_labels, unirr_labels, irr_image, unirr_image, row['state'], clip=True)

            print('{} {}'.format(row['state'], cmt))
            for pos, ct in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['TP', 'FN', 'FP', 'TN']):
                globe_conf[pos] += cmt[ct]['constant']
                annual_conf[pos] += cmt[ct]['constant']
                row[ct] = cmt[ct]['constant']

            p, r = metrics(annual_conf)
            row['prec'], row['rec'] = np.round(p, decimals=3), np.round(r, decimals=3)

            if np.isnan(p) or np.isnan(r):
                continue

            sdf.loc[i] = row
            print('prec {:.3f}, rec {:.3f}'.format(p, r))

        except Exception as e:
            print(e, row['state'], row['year'])
            pass

    print(globe_conf)
    p, r = metrics(globe_conf)
    sdf.to_csv(csv_out)
    print('prec {}, rec {}'.format(p, r))
    print(datetime.datetime.now())


def get_training_data_climate(training_data, state_boundaries, climate, o_json):
    clime_files = {os.path.basename(x)[:2]: os.path.join(climate, x) for x in os.listdir(climate)}

    st_geos = {}
    with fiona.open(state_boundaries, 'r') as st_src:
        for st in st_src:
            state = st['properties']['STUSPS']
            if state in ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']:
                poly = shape(st['geometry'])
                st_geos[state] = poly

    with fiona.open(training_data, 'r') as src:
        ct = 0
        st_years = {}
        for f in src:
            ct += 1
            y = f['properties']['YEAR']
            try:
                field = shape(f['geometry']).centroid
                for st, poly in st_geos.items():
                    if field.intersects(poly):
                        if st not in st_years.keys():
                            st_years[st] = [y]
                        elif y not in st_years[st]:
                            st_years[st].append(y)
                        else:
                            pass
            except Exception as e:
                print(e, y)

    st_clim = {}
    for st, yrs in st_years.items():
        df = read_csv(clime_files[st], header=4)
        df.index = [int(str(x)[:4]) for x in df['Date']]
        first = True
        for yr in yrs:
            try:
                if first:
                    st_clim[st] = {yr: (df.loc[yr]['Value'], df.loc[yr]['Anomaly'])}
                    first = False
                else:
                    st_clim[st].update({yr: (df.loc[yr]['Value'], df.loc[yr]['Anomaly'])})
            except KeyError:
                print(yr)

    with open(o_json, 'w') as fp:
        fp.write(json.dumps(st_clim, indent=4, sort_keys=True))


def get_nass_landcover(nass):
    state_codes = state_fips_code()
    df = read_csv(nass, index_col='GEOID')
    total, irr_area = 0, 0
    for s, c in state_codes.items():
        if s not in TARGET_STATES:
            continue
        if s not in state_county_code().keys():
            continue
        for k, v in state_county_code()[s].items():
            try:
                geoid = v['GEOID']
                tot_acres = county_acres()[geoid]
                water = tot_acres['water']
                land = tot_acres['land']
                total += land + water
                ia = df.loc[int(geoid)]['IRR_2017']
                if np.isnan(ia):
                    print('{} fails'.format(geoid))
                    continue
                irr_area += ia
            except KeyError:
                print('{} fails'.format(geoid))
                continue

    print('total {:.1f}:{:.1f} irrigated'.format(total, irr_area))
    print('{:.3f}'.format(irr_area / total))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    irr_shp = os.path.join(root, 'compiled_training_data/wgs/irrigated_26NOV2021.shp')
    state_shp = os.path.join(root, 'boundaries/states/western_states_11_wgs.shp')
    climate_ = os.path.join(root, 'climate')
    _json = os.path.join(climate_, 'irrmapper', 'training_climate.json')
    _csv = os.path.join(climate_, 'irrmapper', 'pixel_metric_climate_clip_buf_50.csv')
    # get_training_data_climate(irr_shp, state_shp, climate_, json_out)
    calculate_accuracy_by_state(_json, _csv)
    # nass_ = os.path.join(root, 'nass_data', 'nass_irr_crop_new.csv')
    # get_nass_landcover(nass_)

# ========================= EOF ====================================================================
