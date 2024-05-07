import os
from copy import deepcopy
import numpy as np
from pandas import read_table, read_csv, concat
import fiona
import matplotlib.pyplot as plt

from state_county_codes import state_fips_code

DROP = ['SOURCE_DESC', 'SECTOR_DESC', 'GROUP_DESC',
        'COMMODITY_DESC', 'CLASS_DESC', 'PRODN_PRACTICE_DESC',
        'UTIL_PRACTICE_DESC', 'STATISTICCAT_DESC', 'UNIT_DESC',
        'SHORT_DESC', 'DOMAIN_DESC', 'DOMAINCAT_DESC', 'STATE_FIPS_CODE',
        'ASD_CODE', 'ASD_DESC', 'COUNTY_ANSI',
        'REGION_DESC', 'ZIP_5', 'WATERSHED_CODE',
        'WATERSHED_DESC', 'CONGR_DISTRICT_CODE', 'COUNTRY_CODE',
        'COUNTRY_NAME', 'LOCATION_DESC', 'YEAR', 'FREQ_DESC',
        'BEGIN_CODE', 'END_CODE', 'REFERENCE_PERIOD_DESC',
        'WEEK_ENDING', 'LOAD_TIME', 'VALUE', 'AGG_LEVEL_DESC',
        'CV_%', 'STATE_ALPHA', 'STATE_NAME', 'COUNTY_NAME']

TSV = {1987: ('DS0041/35206-0041-Data.tsv', 'ITEM01018', 'FLAG01018'),
       1992: ('DS0042/35206-0042-Data.tsv', 'ITEM010018', 'FLAG010018'),
       1997: ('DS0043/35206-0043-Data.tsv', 'ITEM01019', 'FLAG01019')}

TARGET_STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
E_STATES = ['ND', 'SD', 'NE', 'KS', 'OK', 'TX']
STATES = TARGET_STATES + E_STATES

INT_COLS = ['STATE_ANSI', 'COUNTY_CODE']
FLOAT_COLS = ['VALUE_1987', 'VALUE_1992', 'VALUE_1997', 'VALUE_2002', 'VALUE_2007',
              'VALUE_2012', 'VALUE_2017', 'VALUE_2017', 'VALUE_2022']


def get_old_nass(_dir, out_file):
    master = None
    first = True
    for k, v in TSV.items():
        print(v)
        value = 'VALUE_{}'.format(k)
        _file, item, flag = v
        csv = os.path.join(_dir, _file)
        df = read_table(csv)
        df.columns = [str(x).upper() for x in df.columns]
        df.index = df['FIPS']
        try:
            df.drop('FIPS', inplace=True)
        except KeyError:
            pass
        df = df[['LEVEL', item, flag]]
        df = df[df['LEVEL'] == 1]
        if k != 1997:
            df = df[df[flag] == 0]
        df.dropna(axis=0, subset=[item], inplace=True, how='any')
        if first:
            first = False
            master = deepcopy(df)
            master[value] = df[item].astype(float)
            master.drop([flag, item, 'LEVEL'], inplace=True, axis=1)
        else:
            master = concat([master, df], axis=1)
            master[value] = df[item].astype(float)
            master.drop([flag, item, 'LEVEL'], inplace=True, axis=1)

    master.to_csv(out_file)


def get_nass(csv, out_file, old_nass=None):
    first = True
    if old_nass:
        old_df = read_csv(old_nass)
        old_df.dropna(axis=0, subset=['FIPS'], inplace=True)
        old_df.index = [int(f) for f in old_df['FIPS']]
        old_df = old_df[[c for c in old_df.columns if 'VALUE' in c]]
        old_df = old_df.loc[~old_df.duplicated()]
    for c in csv:
        print(c)
        try:
            df = read_table(c, sep='\t')
            assert len(list(df.columns)) > 2
        except AssertionError:
            df = read_csv(c)
        df.dropna(axis=0, subset=['COUNTY_CODE'], inplace=True, how='any')
        df = df[df['STATE_ALPHA'] == 'MT']
        cty_str = df['COUNTY_CODE'].map(lambda x: str(int(x)).zfill(3))
        idx_str = df['STATE_FIPS_CODE'].map(lambda x: str(int(x))) + cty_str
        idx = idx_str.map(int)
        df.index = idx
        df['COUNTY_NAME'] = [s.replace(' ', '_') for s in df['COUNTY_NAME']]
        df['ST_CNTY_STR'] = df['STATE_ALPHA'] + '_' + df['COUNTY_NAME']
        df = df[(df['SOURCE_DESC'] == 'CENSUS') &
                (df['SECTOR_DESC'] == 'ECONOMICS') &
                (df['GROUP_DESC'] == 'FARMS & LAND & ASSETS') &
                (df['COMMODITY_DESC'] == 'AG LAND') &
                (df['CLASS_DESC'] == 'ALL CLASSES') &
                (df['PRODN_PRACTICE_DESC'] == 'IRRIGATED') &
                (df['UTIL_PRACTICE_DESC'] == 'ALL UTILIZATION PRACTICES') &
                (df['STATISTICCAT_DESC'] == 'AREA') &
                (df['UNIT_DESC'] == 'ACRES') &
                (df['SHORT_DESC'] == 'AG LAND, IRRIGATED - ACRES') &
                (df['DOMAIN_DESC'] == 'TOTAL')]
        df['VALUE'] = df['VALUE'].map(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))
        if first:
            first = False
            new_nass = deepcopy(df)
            new_nass['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            new_nass.drop(columns=DROP, inplace=True)
        else:
            new_nass['VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']

    new_nass.to_csv(out_file.replace('.csv', '_new.csv'))

    if old_nass:
        match = [i for i in old_df.index if i in new_nass.index]
        for c in new_nass.columns:
            if 'VALUE' in c and c not in old_df.columns:
                old_df.loc[match, c] = new_nass.loc[match, c]

        df = old_df.copy()
    else:
        df = new_nass.copy()

    import geopandas as gpd
    county_select = [int('{}{}'.format(c['STATEFP'], c['COUNTYFP'])) for i, c in gpd.read_file(
        '/home/dgketchum/Downloads/west_monitoring_wells_sf/west_mt_counties_nass.shp').iterrows()]
    mt = df.loc[county_select, [c for c in df.columns if 'VALUE' in c]].sum(axis=0)
    mt.to_csv('/home/dgketchum/Downloads/nass_w_mt.csv')

    df.to_csv(out_file)


def merge_nass_irrmapper(nass, irrmapper, out_name):
    years = [1987, 1992, 1997, 2002, 2007, 2012, 2017]
    year_str = [str(x) for x in years]

    idf = read_csv(irrmapper, index_col=[2])
    cols = [x for x in idf.columns if x[-4:] in year_str]
    idf = idf[cols]
    idf.sort_index(axis=1, inplace=True)
    idf.columns = ['IM_{}'.format(x) for x in years]

    ndf = read_csv(nass, index_col=[0])
    ndf.drop(columns=['FIPS'], inplace=True)
    ndf.sort_index(axis=1, inplace=True)
    cols = [x for x in ndf.columns if 'VALUE' in x]
    ndf = ndf[cols]
    cols = ['NASS_{}'.format(y) for y in years]
    ndf.columns = cols

    df = concat([ndf, idf], axis=1)
    df.dropna(axis=0, thresh=8, inplace=True)
    df.to_csv(out_name)


def nass_statewide_summary(csv):
    df = read_csv(csv)
    df = df.groupby(['STATE_ANSI'])[FLOAT_COLS].sum()
    df.index = [str(int(x)).rjust(2, '0') for x in df.index]
    state_codes = state_fips_code()
    state_inv = {v: k for k, v in state_codes.items()}
    states = [state_inv[i] for i in df.index]
    years = [int(x[-4:]) for x in FLOAT_COLS]
    fig, ax = plt.subplots()
    df = df.apply(lambda x: x.div(x.mean(), x.values), axis=1)
    for i, r in df.iterrows():
        if state_inv[i] in TARGET_STATES:
            r.index = years
            r.name = state_inv[i]
            ax = r.plot(ax=ax, kind='line', x=years, y=r.values, alpha=0.6)

    plt.xlim(1984, 2020)
    plt.ylim(0.4, 1.6)
    plt.legend(loc='lower center', ncol=5, labelspacing=0.5)
    plt.show()

    pass


def nass_shapefile(counties, out_shp, nass_data, irr_data, states):
    with fiona.open(counties) as src:
        meta = src.meta
        features = []
        for f in src:
            f['properties']['FIPS'] = int(f['properties']['GEOID'])
            features.append(f)

    idf = read_csv(irr_data, index_col='GEOID')
    df = read_csv(nass_data, index_col='FIPS')
    idx = [i for i, x in enumerate(df.ST_CNTY_STR) if isinstance(x, str)]
    df = df.iloc[idx]
    df = df.iloc[[i for i, x in enumerate(df.index) if np.isfinite(x)]]
    df.index = [int(i) for i in df.index]
    df['FIPS'] = df.index
    for c in df.columns:
        if c in INT_COLS:
            df[c] = df[c].astype(int, copy=True)
        elif c in FLOAT_COLS:
            df[c] = df[c].astype(float, copy=True)
        else:
            df[c] = df[c].astype(str, copy=True)
    df = df.fillna(0)
    cols = list(df.columns) + list(idf.columns)
    [meta['schema']['properties'].update({col: 'int'}) for col in cols]
    in_feat = len(features)
    ct = 0
    with fiona.open(out_shp, 'w', **meta) as output:
        for feat in features:
            try:
                fips = feat['properties']['FIPS']
                _ = df.loc[int(fips)].to_dict()
                update = {}
                for k, v in _.items():
                    try:
                        update[k] = v.item()
                    except AttributeError:
                        update[k] = v
                feat['properties'].update(update)
                feat['properties']['sum'] = idf.loc[fips]['sum'].item()
                output.write(feat)
                ct += 1
            except Exception as e:
                print(feat['properties']['FIPS'], e)
    print('{} of {} features joined'.format(ct, in_feat))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nass_tables = os.path.join(root, 'nass_data')
    merged = os.path.join(nass_tables, 'nass_merged.csv')
    irr_extract = os.path.join(nass_tables, 'co_sw_23NOV2021_2017.csv')
    nass = os.path.join(nass_tables, 'nass_merged.csv')


    files = [os.path.join(nass_tables, 'qs.census{}.txt'.format(yr)) for yr in [2022]]
    out_nass = os.path.join(nass_tables, 'nass_irr_area_acres_MT_2002_2022_4MAR2024.csv')
    get_nass(files, out_file=out_nass, old_nass=nass)

    co_shp = os.path.join(root, 'boundaries', 'counties', 'western_17_states_counties_wgs.shp')
    o_shp = os.path.join(nass_tables, 'nass_counties.shp')

    # nass_shapefile(co_shp, o_shp, merged, irr_extract, STATES)
    # nass_statewide_summary(nass)
# ========================= EOF ====================================================================
