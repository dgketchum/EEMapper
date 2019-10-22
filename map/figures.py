# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import InsetPosition
from pandas import read_csv, Series
from numpy import array, vstack, append, isnan, any
from sklearn.metrics import r2_score


def state_sum(csv):
    cdf = read_csv(csv)
    df = cdf.groupby(['State', 'State_Code'])[['IM2002_ac', 'NASS_2002_ac', 'IM2007_ac',
                                               'NASS_2007_ac', 'IM2012_ac', 'NASS_2012_ac']].sum()
    fig, ax = plt.subplots(1, 1)
    s = Series(index=df.index)
    s.loc[0], s.loc[df.shape[0]] = 0, 1e8
    s.interpolate(axis=0, inplace=True)
    s.index = s.values
    s.plot(x=s.values, ax=ax, kind='line', lw=1, loglog=True, color='k', style='--', alpha=0.5)
    ax = df.plot(x='NASS_{}_ac'.format(2002), y='IM{}_ac'.format(2002), kind='scatter', s=4,
                 xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True, color='g')
    df.plot(x='NASS_{}_ac'.format(2007), y='IM{}_ac'.format(2007), kind='scatter', s=4,
            xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True, color='b')
    df.plot(x='NASS_{}_ac'.format(2012), y='IM{}_ac'.format(2012), kind='scatter', s=4,
            xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True, color='r')
    plt.xlabel('NASS FRIS Total Irrigated Acres'.format())
    plt.ylabel('IrrMapper Total Irrigated Acres'.format())
    # plt.show()
    # plt.savefig('figs/state_sum_{}.png'.format(yr))
    return plt


def compare_nass_irrmapper_scatter(csv, fig_name=None, print_correlation=False):
    df = read_csv(csv)
    fig, ax = plt.subplots(1, 1)
    s = Series(index=df.index)
    s.loc[0], s.loc[df.shape[0]] = 0, 1e6
    s.interpolate(axis=0, inplace=True)
    s.index = s.values
    s.plot(x=s.values, ax=ax, kind='line', lw=1, loglog=True, color='k', style='--', alpha=0.5, label='_nolegend_')

    df.plot(x='NASS_{}_ac'.format(2002), y='IM{}_ac'.format(2002), kind='scatter', s=4,
                 xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True, color='g')
    df.plot(x='NASS_{}_ac'.format(2007), y='IM{}_ac'.format(2007), kind='scatter', s=4,
            xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True, color='b')
    df.plot(x='NASS_{}_ac'.format(2012), y='IM{}_ac'.format(2012), kind='scatter', s=4,
            xlim=(1e2, 1e6), ylim=(1e2, 1e6), ax=ax, loglog=True, color='r')

    plt.xlabel('NASS AgStats Total Irrigated Acres, Counties'.format())
    plt.ylabel('IrrMapper Total Irrigated Acres'.format())
    legend =ax.legend(['2002', '2007', '2012'], loc='lower right')
    # fig.suptitle('Lorem ipsum dolor sit amet, consectetur adipiscing elit', ha='center')
    frame = legend.get_frame()
    frame.set_color('grey')
    df_state = df.groupby(['State', 'State_Code'])[['IM2002_ac', 'NASS_2002_ac', 'IM2007_ac',
                                                    'NASS_2007_ac', 'IM2012_ac', 'NASS_2012_ac']].sum()
    s = Series(index=df_state.index)
    s.loc[0], s.loc[df_state.shape[0]] = 1e6, 1e7
    s.interpolate(axis=0, inplace=True)
    s.index = s.values
    ax2 = fig.add_axes([0, 0, 1, 1])
    ax2.xaxis.label.set_visible(False)
    ax2.yaxis.label.set_visible(False)
    ax.text(0.26, 0.62, 'States', transform=ax.transAxes, ha="right")

    df_state.plot(x='NASS_{}_ac'.format(2002), y='IM{}_ac'.format(2002), kind='scatter', s=4,
                  xlim=(1e6, 1e7), ylim=(1e6, 1e7), ax=ax2, loglog=True, color='g')
    df_state.plot(x='NASS_{}_ac'.format(2007), y='IM{}_ac'.format(2007), kind='scatter', s=4,
                  xlim=(1e6, 1e7), ylim=(1e6, 1e7), ax=ax2, loglog=True, color='b')
    df_state.plot(x='NASS_{}_ac'.format(2012), y='IM{}_ac'.format(2012), kind='scatter', s=4,
                  xlim=(1e6, 1e7), ylim=(1e6, 1e7), ax=ax2, loglog=True, color='r')
    s.plot(x=s.values, kind='line', lw=1, loglog=True, color='k', style='--', alpha=0.5, label='_nolegend_')
    ip = InsetPosition(ax, [0.07, 0.67, 0.3, 0.3])
    ax2.set_axes_locator(ip)
    # plt.show()
    if fig_name:
        plt.savefig(fig_name)

    if print_correlation:
        irrmap_cols = [x for x in df_state.columns if 'IM' in x]
        nass_cols = [x for x in df_state.columns if 'NASS' in x]
        im_state_df = df_state[irrmap_cols]
        nass_state_df = df_state[nass_cols]
        im_state = array([im_state_df[x].values for x in im_state_df]).flatten()
        nass_state = array([nass_state_df[x].values for x in nass_state_df]).flatten()
        comp = vstack([im_state, nass_state]).transpose()
        comp = append(comp, (comp[:, 0] - comp[:, 1]).reshape(comp.shape[0], 1), axis=1)
        rmse = ((comp[2, :]) ** 2).mean() ** .5
        coeff_det = r2_score(comp[:, 0], comp[:, 1])
        print('state r squared: {}'.format(coeff_det))

        irrmap_cols = [x for x in df.columns if 'IM' in x]
        nass_cols = [x for x in df.columns if 'NASS' in x]
        im_county_df = df[irrmap_cols]
        nass_county_df = df[nass_cols]
        im_county = array([im_county_df[x].values for x in im_county_df]).flatten()
        nass_county = array([nass_county_df[x].values for x in nass_county_df]).flatten()
        comp = vstack([im_county, nass_county]).transpose()
        comp = append(comp, (comp[:, 0] - comp[:, 1]).reshape(comp.shape[0], 1), axis=1)
        comp = comp[~isnan(comp).any(axis=1)]
        rmse = ((comp[2, :]) ** 2).mean() ** .5
        coeff_det = r2_score(comp[:, 0], comp[:, 1])
        print('county r squared: {}'.format(coeff_det))
    return None


def irr_time_series(csv, fig_name=None):
    df = read_csv(csv)
    yrs = [x for x in df.columns if 'Ct_' in x]
    df = df.groupby(['STATEFP']).sum()
    df = df[yrs]
    df = df.div(df.mean(axis=1), axis=0)
    linear = [x for x in range(1986, 2019)]
    totals = df.sum(axis=0)
    z_totals = totals.div(totals.mean(), totals.values)
    z_totals.index = linear
    fig, ax = plt.subplots()
    for i, r in df.iterrows():
        r.index = linear
        r.name = state_fp_code()[r.name]
        ax = r.plot(ax=ax, kind='line', x=linear, y=r.values, alpha=0.6)

    z_totals.name = 'All'
    z_totals.plot(ax=ax, kind='line', x=linear, y=z_totals.values)

    plt.legend(loc='lower center', ncol=6)
    if fig_name:
        plt.savefig(fig_name.replace('.', '_state.'))
    plt.show()


def state_fp_code():
    return {4: 'AZ',
            6: 'CA',
            8: 'CO',
            16: 'ID',
            30: 'MT',
            32: 'NV',
            35: 'NM',
            41: 'OR',
            49: 'UT',
            53: 'WA',
            56: 'WY'}


if __name__ == '__main__':
    home = os.path.expanduser('~')

    tables = os.path.join(home, 'IrrigationGIS', 'time_series')
    huc_8 = os.path.join(tables, 'tables', 'concatenated_huc8.csv')
    # time_series_normalized(huc_8)

    irr_tables = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county', 'counties_v2', 'noCdlMask_minYr5')
    nass_tables = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')

    irr = os.path.join(irr_tables, 'irr_merged.csv')
    nass = os.path.join(nass_tables, 'nass_merged.csv')
    o = os.path.join(irr_tables, 'nass_irrMap.csv')

    irr_all = os.path.join(irr_tables, 'irr_v2_noCdlMask_minYr5_25SEPT2019.csv')
    irr_shp = irr_all.replace('.csv', '.shp')

    # figure = 'figs/z_annual_irr_byState_noCdl_5Yr_25SETP2019.png'
    figure = 'figs/comparison_noCdl_5Yr_22OCT2019.png'

    # compare_nass_irrmapper_scatter(o, print_correlation=True)
    irr_time_series(irr_all, fig_name=figure)
    # state_sum(o)
# ========================= EOF ====================================================================
