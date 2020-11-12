import os
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from numpy import logical_not, isnan, array, where, abs, max, min, vstack, hstack
from pandas import read_csv, Series, DataFrame
# from sklearn import linear_model
# from sklearn.metrics import r2_score

from map.variable_importance import variable_importance

# plt.style.use('presentation.mplstyle')


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


def compare_nass_irrmapper_scatter(csv, fig_name=None):
    df = read_csv(csv)
    s = array([0, 1e6])
    plt.rcParams['figure.constrained_layout.use'] = True
    fig, ax = plt.subplots(2, 4, figsize=(6, 5), tight_layout=True, sharex=True, sharey=True)
    rows, cols = [0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]
    all_comparison = array([]).reshape((0, 2))

    for r, c, year in zip(rows, cols, range(1987, 2022, 5)):

        n, i = 'NASS_{}'.format(year), 'IM_{}'.format(year)

        a = ax[r, c]

        ydf = df[[n, i]]
        ydf = ydf / 247.105

        nass, irr = ydf[n].values, ydf[i].values
        nass, irr = nass[logical_not(isnan(nass))], irr[logical_not(isnan(nass))]
        nass, irr = nass.reshape(nass.shape[0], 1), irr.reshape(irr.shape[0], 1)

        print('{} county correlations...'.format(year))
        r2, m, _int = get_correlations(nass, irr)

        a.plot(s, s, linewidth=1, linestyle='--', color='k', alpha=0.5, label='_nolegend_')

        ydf[n].name = ''
        ydf[i].name = ''
        ydf.plot(n, i, xlim=(1, 1e6), ylim=(1, 1e6), loglog=True, color='b', alpha=0.25,
                 kind='scatter', ax=a, marker='o', s=3)

        a.set(adjustable='box')
        a.set_title(str(year), size=10)

        a.text(0.05, 0.9, '$r^2$={0:.3f}'.format(r2), transform=a.transAxes,
               size=7)
        a.text(0.05, 0.85, '$m$={0:.2f}'.format(m), transform=a.transAxes,
               size=7)

        if r == 0 and c == 3:
            a.set_xticks([])
            a.tick_params(labelsize=10)

        a.tick_params(labelsize=10)

        x_axis = a.xaxis
        x_axis.label.set_visible(False)
        a.set_xlim(1, 1e4)

        y_axis = a.yaxis
        y_axis.label.set_visible(False)
        a.set_ylim(1, 1e4)
        comp = hstack([irr, nass])
        all_comparison = vstack([all_comparison, comp])

    print('overall county correlations...')
    i_c = all_comparison[:, 0].reshape((all_comparison[:, 0].shape[0], 1))
    n_c = all_comparison[:, 1].reshape((all_comparison[:, 1].shape[0], 1))
    get_correlations(i_c, n_c)
    fig.delaxes(ax[1, 3])
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    x_txt = 'NASS ($\mathregular{km^2}$)'
    y_txt = 'IrrMapper ($\mathregular{km^2}$)'

    # fig.subplots_adjust(bottom=0.2, left=-0.5)
    fig.text(0.5, 0.0, x_txt, ha='center', fontsize=10)
    fig.text(0.0, 0.5, y_txt, va='center', rotation='vertical', fontsize=10)
    if fig_name:
        plt.savefig(fig_name, pad_inches=0.25)
    else:
        # plt.show()
        plt.close()


def get_correlations(a, b):
    coeff_det = r2_score(a, b)
    regr = linear_model.LinearRegression()
    regr.fit(a, b)
    pred = regr.predict(a)
    slope, intercept = regr.coef_[0][0], regr.intercept_
    # print('regression coeff: {}'.format(slope))
    print('r squared: {}'.format(coeff_det))
    return coeff_det, slope, intercept


def irr_time_series_states(csv, fig_name=None):
    df = read_csv(csv)
    df = df.sort_index(axis=1)
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
        r.name = state_fp_code_abv()[r.name]
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
        return None
    plt.show()


def irr_time_series_totals(irr, nass, fig_name=None):
    df = read_csv(irr)
    df.drop(['COUNTYFP', 'COUNTYNS', 'LSAD', 'GEOID'], inplace=True, axis=1)
    df = df.groupby(['STATEFP']).sum()
    state_irr = deepcopy(df)
    totals = df.sum(axis=0)
    labels = [x for x in df.columns if 'noCdlMask' in x]
    irr_years = [x for x in range(1986, 2019)]
    totals_s = totals[labels]
    irr_nass_years = deepcopy(totals_s)
    totals_s.sort_index(inplace=True)
    totals_s.index = irr_years
    totals = totals_s.values / 247.105

    nass = read_csv(nass, index_col=[0])
    nass.dropna(axis=0, subset=['STATE_ANSI'], inplace=True)
    nass['STATE_ANSI'] = nass['STATE_ANSI'].astype(int)
    nass = nass.loc[nass['STATE_ANSI'].isin(list(df.index))]
    state_nass = deepcopy(nass)
    cols = [x for x in nass.columns if 'VALUE' in x]
    state_nass = state_nass[cols + ['STATE_ANSI']] / 247.105
    state_nass = state_nass.groupby(['STATE_ANSI']).sum()
    nass = nass[cols]
    nass = nass / 247.105
    nass_s = nass.sum(axis=0)
    nass_years = [int(x[-4:]) for x in nass_s.index]
    irr_labels = [x for x in labels if int(x[-4:]) in nass_years]
    state_irr = state_irr[irr_labels].sort_index() / 247.105
    nass_s.index = nass_years
    nass_values = nass_s.values

    print('state-by state correlations...')
    s_i = state_irr.values.reshape(state_irr.values.size, 1)
    s_n = state_nass.values.reshape(state_nass.values.size, 1)
    get_correlations(s_i, s_n)

    plt.plot(irr_years, totals / 1000., label='IrrMapper', zorder=1)
    plt.scatter(x=nass_years, y=nass_values / 1000., marker='.', color='red', label='NASS', zorder=2)
    # plt.title('Total Irrigated Area, Western 11 States \n 1986 - 2018')
    plt.xlim(1985, 2019)
    # plt.ylim(20, 30)
    plt.ylabel('Thousand $\mathregular{km^2}$')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.legend()
    if fig_name:
        plt.savefig(fig_name)
    else:
        pass
        # plt.show()


def state_fp_code_abv():
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


def state_fp_code_full_name():
    return {4: 'Arizona',
            6: 'California',
            8: 'Colorado',
            16: 'Idaho',
            30: 'Montana',
            32: 'Nevada',
            35: 'New Mexico',
            41: 'Oregon',
            49: 'Utah',
            53: 'Washington',
            56: 'Wyoming'}


def irrigated_years_precip_anomaly(csv, save_fig=None):
    df = read_csv(csv, skip_blank_lines=True).dropna()
    means = df.groupby(by=['State']).mean().drop(columns=['Year', 'Anomaly Inches', 'Anomaly mm'])

    n_cols, n_rows = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True,
                             sharey=False, figsize=(12, 6))

    pos = [(0, 0), (0, 1), (0, 2), (0, 3),
           (1, 0), (1, 1), (1, 2), (1, 3),
           (2, 0), (2, 1), (2, 2), (2, 3)]

    i = 0
    for p in pos:

        ax = axes[p[1], p[0]]

        if p == (0, 3):
            yrs_ = [_ for _ in range(1986, 2019)]
            d = [0 for _ in yrs_]
            ax.bar(d, height=d, bottom=d, width=0.75, align='center')
            ax.set(xlabel='Time')
            plt.xlim([1986, 2019])
            plt.ylim([0, 1])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.spines['left'].set_position(('data', 1986))
            ax.spines['right'].set_position(('data', 2018))
            ax.spines['left'].set_color('k')
            ax.spines['bottom'].set_position(('axes', 0.45))
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.tick_params(axis='y', which='major', length=0, labelleft=False, labelright=False)


        else:
            name = means.iloc[i].name
            print(name)
            d = df[df['State'] == name]
            a = d['Anomaly mm'].values
            mean_ = means.iloc[i]['Mean mm']
            bottoms = where(a < 0.0, mean_ + a, mean_)
            height = abs(a)
            x = d['Year'].values

            # data_color = [(a[i] - a.min()) / (a.max() - a.min()) for i, _ in enumerate(a)]
            data_color = [0.9 if x > 0.0 else 0.1 for x in a]
            cmap = cm.get_cmap('RdBu')
            color = cmap(data_color)
            ax.bar(x, height=height, bottom=bottoms, width=0.75, align='center', color=color)

            plt.xlim([1986, 2018])
            plt.ylim([min(bottoms) - mean_ * 0.1, max(a + mean_) + mean_ * 0.1])
            ax.set_title(name, size=12, y=0.9)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter(''))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(which='minor', length=1.5)

            ax.spines['left'].set_position(('data', 1986))
            ax.spines['right'].set_position(('data', 2018))
            ax.spines['bottom'].set_position(('data', mean_))
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=False)
            i += 1

    # fig.delaxes(axes[5, 1])
    tick_years = range(1990, 2020, 5)
    plt.setp(axes, xticks=tick_years, xticklabels=[str(x) for x in tick_years])

    if save_fig:
        plt.tight_layout()
        plt.savefig(save_fig)
        return None
    plt.show()


def state_bar_plots(csv, save_fig=None):
    df = read_csv(csv)
    df = df.sort_index(axis=1)
    year_sums = [x for x in df.columns if 'noCdlMask_' in x]
    df = df.groupby(['STATEFP']).sum()
    df = df[year_sums] / 247.105

    n_cols, n_rows = 2, 6
    fig, axes = plt.subplots(n_rows, n_cols, sharex=False,
                             sharey=False, figsize=(12, 10))

    pos = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
           (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]

    for p, (k, v) in zip(pos, state_fp_code_abv().items()):

        ax = axes[p[1], p[0]]

        if p == (0, 5):
            yrs = [_ for _ in range(1986, 2019)]
            d = [0 for _ in yrs]
            ax.bar(yrs, height=d, bottom=d, width=0.75, align='center')
            ax.set(xlabel='Year')
            plt.xlim([1986, 2019])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.spines['left'].set_position(('data', 1984))
            ax.spines['right'].set_position(('data', 2018))
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_position(('data', 0.06))
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.yaxis.set_major_formatter(FormatStrFormatter(''))
            ax.tick_params(axis='y', which='major', length=0, labelleft=False, labelright=False)
            ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True)
            # plt.xticks(plt.xticks()[0], [str(x) for x in plt.xticks()[0]])


        else:
            name = v
            d = df[df.index == k]
            yrs = [int(x.replace('noCdlMask_', '')) for x in d.columns]
            d.columns = yrs
            mean_ = d.values.mean()
            a = d.values - mean_
            bottoms = where(a < 0.0, mean_ + a, mean_)[0, :]
            height = abs(a)[0, :]

            data_color = [(a[i] - a.min()) / (a.max() - a.min()) for i, _ in enumerate(a)][0]
            cmap = cm.get_cmap('RdBu')
            color = cmap(data_color)
            ax.bar(yrs, height=height, bottom=bottoms, width=0.75, align='center', color=color)
            plt.xlim([1986, 2019])
            plt.ylim([min(bottoms) - mean_ * 0.1, max(a + mean_) + mean_ * 0.1])
            ax.set_title(name, size=10, y=0.9, x=0.1)
            ax.tick_params(which='minor', length=1.5, labelbottom=False, labeltop=False)
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter(''))
            ax.spines['left'].set_position(('data', 1984))
            ax.spines['right'].set_position(('data', 2018))
            ax.spines['bottom'].set_position(('data', mean_))
            ax.yaxis.set_ticks_position('left')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

    fig.delaxes(axes[5, 1])
    fig.text(0.37, 0.5, 'Annual Irrigated Area, $\mathregular{km^2}$', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.44, bottom=None, right=None,
                        top=None, wspace=None, hspace=None)
    if save_fig:
        plt.savefig(save_fig)
        return None
    plt.show()


def variable_importance_barh(savefig=None):
    # the top 20 variables add to 0.54
    vi = variable_importance()[:20]
    n = 'Variable'
    d = 'Importance'
    df = DataFrame.from_records(data=vi, columns=[n, d])
    df = df.sort_values(by='Importance', ascending=True)
    df.plot(n, d, kind='barh')
    if savefig:
        plt.tight_layout()
        plt.savefig(savefig)
        return None
    plt.show()


if __name__ == '__main__':
    # home = os.path.expanduser('~')
    home = '/media/research'
    county = os.path.join(home, 'IrrigationGIS', 'time_series', 'exports_county')
    nass_merged = os.path.join(county, 'nass_merged.csv')
    irr_tables = os.path.join(county, 'counties_v2', 'noCdlMask_minYr5')

    irrmapper_all = os.path.join(irr_tables, 'irr_merged_ac.csv')
    totals_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
                                 'figures', 'totals_time_series.pdf')
    irr_time_series_totals(irrmapper_all, nass_merged)

    # nass_irrmapper = os.path.join(irr_tables, 'nass_irrMap.csv')
    # scatter_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
    #                               'figures', 'comparison_scatter_6MAY2020.pdf')
    # compare_nass_irrmapper_scatter(nass_irrmapper)

    # state_irrmapper = os.path.join(irr_tables, 'irrmapper_annual_acres_state.csv')
    # state_normalized_figure = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
    #                                        'figures', 'states_normalized.pdf')
    # irr_time_series_states(state_irrmapper, state_normalized_figure)
    #
    # irr_precip = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
    #                           'IrrMapper_Irrigation_Years_PrecipAnom.csv')
    # precip_fig = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper', 'figures',
    #                           'IrrYears_precipAnomaly_7MAY2020.pdf')
    # irrigated_years_precip_anomaly(irr_precip, precip_fig)
    #
    # variable_import = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper',
    #                                'figures', 'variable_importance_6MAY2020.pdf')
    # variable_importance_barh(savefig=variable_import)

    # state_bars = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper', 'figures', 'state_bars.png')
    # state_bar_plots(state_irrmapper, save_fig=None)

    irr_precip = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper', 'tables', 'IrrMapper_Irrigation_Years_PrecipAnom.csv')
    precip_fig = os.path.join(home, 'IrrigationGIS', 'paper_irrmapper', 'figures', 'precip_training_years.pdf')
    irrigated_years_precip_anomaly(irr_precip, save_fig=precip_fig)
# ========================= EOF ====================================================================
