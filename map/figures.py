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
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from numpy import logical_not, isnan, array
from pandas import read_csv, Series
from sklearn import linear_model
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
    plt.show()

    return plt


def compare_nass_irrmapper_scatter(csv, fig_name):
    def get_axis_limits(ax, scale=.9):
        return ax.get_xlim()[1] * scale, ax.get_ylim()[1] * scale

    df = read_csv(csv)
    s = array([0, 1e6])
    fig, ax = plt.subplots(2, 4, figsize=(8, 6))
    rows, cols = [0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]

    for r, c, year in zip(rows, cols, range(1987, 2022, 5)):

        n, i = 'NASS_{}'.format(year), 'IM_{}'.format(year)
        a = ax[r, c]

        ydf = df[[n, i]]
        ydf = ydf / 247.105

        nass, irr = ydf[n].values, ydf[i].values
        nass, irr = nass[logical_not(isnan(nass))], irr[logical_not(isnan(nass))]
        nass, irr = nass.reshape(nass.shape[0], 1), irr.reshape(irr.shape[0], 1)

        r2, m, _int = get_correlations(nass, irr)

        a.plot(s, s, linewidth=1, linestyle='--', color='k', alpha=0.5, label='_nolegend_')

        ydf.plot(n, i, xlim=(1, 1e6), ylim=(1, 1e6), loglog=True, color='b', alpha=0.25,
                 kind='scatter', ax=a, marker='o', s=3)

        a.set(adjustable='box')
        a.set_title(str(year), size=10)

        a.text(0.05, 0.9, '$r^2$={0:.3f}'.format(r2), transform=a.transAxes,
               size=7)
        a.text(0.05, 0.85, '$m$={0:.2f}'.format(m), transform=a.transAxes,
               size=7)

        if c > 0:
            a.set_yticks([])
        if r == 0 and c < 3:
            a.set_xticks([])

        x_axis = a.xaxis
        x_axis.label.set_visible(False)
        a.set_xlim(1, 1e4)

        y_axis = a.yaxis
        y_axis.label.set_visible(False)
        a.set_ylim(1, 1e4)

    fig.delaxes(ax[1, 3])
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    x_txt = 'NASS Irrigated Area ($\mathregular{km^2}$)'
    y_txt = 'IrrMapper Irrigated Area ($\mathregular{km^2}$)'
    fig.text(0.5, 0.0, x_txt, ha='center')
    fig.text(0.0, 0.5, y_txt, va='center', rotation='vertical')
    # plt.show()
    plt.savefig(fig_name)
    # plt.close()


def get_correlations(a, b):
    coeff_det = r2_score(a, b)
    regr = linear_model.LinearRegression()
    regr.fit(a, b)
    pred = regr.predict(a)
    slope, intercept = regr.coef_[0][0], regr.intercept_
    print('regression coeff: {}'.format(slope))
    print('r squared: {}'.format(coeff_det))
    return coeff_det, slope, intercept


def irr_time_series_states(csv, fig_name=None):
    #  this uses pixel counts still
    df = read_csv(csv)
    yrs = [x for x in df.columns if 'noCdlMask_' in x]
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
    z_totals.plot(ax=ax, kind='line', color='k', alpha=0.7, x=linear, y=z_totals.values)
    # plt.title('Normalized Irrigated Area')
    ax.axvspan(2011.5, 2012.5, alpha=0.5, color='red')
    plt.xlim(1984, 2020)
    plt.ylim(0.4, 1.6)
    plt.legend(loc='lower center', ncol=5, labelspacing=0.5)
    if fig_name:
        plt.savefig(fig_name)
    # plt.show()


def irr_time_series_totals(irr, nass, fig_name):

    df = read_csv(irr)
    df.drop(['COUNTYFP', 'COUNTYNS', 'LSAD', 'GEOID'], inplace=True, axis=1)
    df = df.groupby(['STATEFP']).sum()
    totals = df.sum(axis=0)
    labels = [x for x in df.columns if 'noCdlMask' in x]
    irr_years = [x for x in range(1986, 2019)]
    totals = totals[labels]
    totals.sort_index(inplace=True)
    totals.index = irr_years
    totals = totals.values / 247.105

    nass = read_csv(nass, index_col=[0])
    nass.dropna(axis=0, subset=['STATE_ANSI'], inplace=True)
    nass['STATE_ANSI'] = nass['STATE_ANSI'].astype(int)
    nass = nass.loc[nass['STATE_ANSI'].isin(list(df.index))]
    cols = [x for x in nass.columns if 'VALUE' in x]
    nass = nass[cols]
    nass = nass / 247.105
    nass = nass.sum(axis=0)
    nass_years = [int(x[-4:]) for x in nass.index]
    nass.index = nass_years
    nass_values = nass.values

    plt.plot(irr_years, totals, label='IrrMapper', zorder=1)
    plt.scatter(x=nass_years, y=nass_values, marker='*', color='red', label='NASS', zorder=2)
    # plt.title('Total Irrigated Area, Western 11 States \n 1986 - 2018')
    plt.xlim(1985, 2019)
    # plt.ylim(20, 30)
    plt.ylabel('Thousand $\mathregular{km^2}$')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.legend()
    if fig_name:
        plt.savefig(fig_name.replace('.', '_totals.'))
    # plt.show()`


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

    county = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')
    nass_merged = os.path.join(county, 'nass_merged.csv')
    irr_tables = os.path.join(county, 'counties_v2', 'noCdlMask_minYr5')

    irrmapper_all = os.path.join(irr_tables, 'irr_merged.csv')
    totals_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
                          'figures', 'totals_time_series.png')

    nass_irrmapper = os.path.join(irr_tables, 'nass_irrMap.csv')
    scatter_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
                                  'figures', 'comparison_scatter_13NOV2019.png')

    state_irrmapper = os.path.join(irr_tables, 'irrmapper_annual_acres_state.csv')
    state_normalized_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
                                           'figures', 'states_normalized.png')

    # compare_nass_irrmapper_scatter(nass_irrmapper, scatter_figure)
    # irr_time_series_states(state_irrmapper, fig_name=state_normalized_figure)
    # state_sum(o)
    irr_time_series_totals(irrmapper_all, nass_merged, fig_name=totals_figure)
# ========================= EOF ====================================================================
