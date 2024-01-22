"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
import io
import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.lines import Line2D
from werkzeug.utils import secure_filename

from arch_opt_exp.experiments.experimenter import *

__all__ = ['plot_scatter', 'plot_problem_bars', 'plot_for_pub', 'analyze_perf_rank', 'plot_perf_rank', 'sb_theme',
           'plot_multi_idx_lines', 'plot_for_pub_sb']


_col_names = {
    'fail_rate_ratio': 'Fail rate ratio',
    'fail_rate': 'Fail rate % (end)',
    'fail_ratio': 'Fail rate ratio',
    'delta_hv_ratio': '$\Delta$HV ratio',
    'delta_hv_regret': '$\Delta$HV regret',
    'delta_hv_abs_regret': '$\Delta$HV regret',
    'iter_delta_hv_regret': '$\Delta$HV regret',
    'delta_hv_delta_hv': '$\Delta$HV',
    'hc_pred_acc': 'Predictor Accuracy',
    'hc_pred_max_acc': 'Predictor Max Accuracy',
    'hc_pred_max_acc_pov': 'Predictor POV @ Max Accuracy',
    'strategy': 'Strategy',
    'rep_strat': 'Replacement Strategy',
    'g_f_strat': 'Acquisition constraint (G) vs penalty (F)',
    'pw_strat': 'Predicted Worst Replacement variant',
    'doe_k': 'DOE K',
    'min_pov': '$PoV_{min}$',
    'delta_hv_pass_10': 'Delta HV pass 10%',
    'delta_hv_pass_20': 'Delta HV pass 20%',
    'delta_hv_pass_50': 'Delta HV pass 50%',
    'infill': 'Infill criterion',
    'cls': 'Class',
    'time_train': 'Training time [s]',
    'time_infill': 'Infill time [s]',
    'n_comp': 'Model comps',
    'is_ck_lt': 'Is light categorical kernel',
    'n_cat': 'Cat vars',
    'n_theta': 'Nr of hyperparameters',
    'corr_time_mean': 'Correction time [s]',
}


def plot_scatter(df_agg, folder, x_col, y_col, z_col=None, x_log=False, y_log=False, z_log=False, cmap='inferno'):
    x_name, y_name = _col_names[x_col], _col_names[y_col]
    z_name = _col_names[z_col] if z_col is not None else None
    plt.figure(figsize=(8, 4)), plt.title(f'{x_name} vs {y_name}')

    x_all, y_all, z_all = [], [], []
    x, y = df_agg[x_col].values, df_agg[y_col].values
    x_all += list(x)
    y_all += list(y)
    if z_col is not None:
        z = df_agg[z_col].values
        if z.dtype == object:
            values = {val for val in z if val is not None}
            z_num = np.zeros((len(z),))*np.nan
            for i_val, value in enumerate(sorted(values)):
                if value or value == 0.:
                    z_num[z == value] = i_val
            z = z_num
        z_all += list(np.log10(z) if z_log else z)
        size = 50
    else:
        z_all = 'k'
        size = 10

    c = plt.scatter(x_all, y_all, s=size, c=z_all, cmap=cmap)
    if z_col is not None:
        plt.colorbar(c).set_label((z_name + ' (log)') if z_log else z_name)

    if x_log:
        plt.gca().set_xscale('log')
    if y_log:
        plt.gca().set_yscale('log')
    plt.xlabel(x_name), plt.ylabel(y_name)

    plt.tight_layout()
    filename = f'{folder}/{x_col}_{y_col}{f"_{z_col}" if z_col is not None else ""}'
    plt.savefig(filename+'.png')


def plot_problem_bars(df_agg, folder, cat_col, y_col, y_log=False, prefix=None, prob_name_map=None, cat_colors=None,
                      label_i=None, label_rot=65, cat_names=None, rel=False):
    if prob_name_map is None:
        prob_name_map = {}
    col_name, y_col_name = _col_names[cat_col], _col_names[y_col]
    plt.figure(figsize=(8, 4))
    # plt.title(col_name)

    categories = df_agg[cat_col].unique()
    categories = np.array([val for val in categories if val and val != 0.])
    i_mid_cat = label_i if label_i is not None else len(categories) // 2
    if cat_colors is not None:
        assert len(cat_colors) == len(categories)
    else:
        if len(categories) <= 3:
            cat_colors = ['r', 'b'] if len(categories) == 2 else ['r', 'g', 'b']
        else:
            cat_colors = plt.cm.plasma(np.linspace(0, 1, len(categories)))
    x, w0 = 0, .8
    w = w0/len(categories)
    x_bars, y_bars, y_lower, y_upper, colors, labels = [], [], [], [], [], []
    for prob_name, df_group in df_agg.groupby(level=0):
        median_factor = None
        for i_cat, cat_value in enumerate(categories):
            cat_mask = df_group[cat_col] == cat_value
            medians = list(df_group[y_col].values[cat_mask])
            for q_post in ['q25', 'q75']:
                if f'{y_col}_{q_post}' in df_group.columns:
                    medians += list(df_group[f'{y_col}_{q_post}'].values[cat_mask])
            if len(medians) == 0:
                continue

            if median_factor is None:
                if rel:
                    median_factor = 1/np.nanmedian(medians)
                else:
                    median_factor = 1.

            x_bars.append(x-.5*w0+w*i_cat)
            labels.append(prob_name_map.get(prob_name, prob_name) if i_cat == i_mid_cat else '')
            medians = sorted(medians)
            y_bars.append(np.nanmedian(medians)*median_factor)
            y_lower.append(y_bars[-1]-np.nanquantile(medians, .25)*median_factor)
            y_upper.append(np.nanquantile(medians, .75)*median_factor-y_bars[-1])
            colors.append(cat_colors[i_cat])
        x += 1

    plt.bar(x_bars, y_bars, w, color=colors, tick_label=labels, yerr=np.array([y_lower, y_upper]), capsize=2,
            edgecolor='#424242', linewidth=.5, error_kw={'linewidth': .5})
    plt.xticks(rotation=label_rot, fontsize=8)
    if y_log:
        plt.yscale('log')
    plt.gca().legend([Line2D([0], [0], color=cat_colors[i_cat], lw=4) for i_cat, _ in enumerate(categories)],
                     cat_names if cat_names is not None else categories, loc='center left', bbox_to_anchor=(1, .5),
                     frameon=False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.ylabel(f'{y_col_name} (relative)' if rel else y_col_name)
    plt.tight_layout()

    prefix = f'{prefix}_' if prefix is not None else ''
    filename = f'{folder}/{prefix}{cat_col}_{y_col}'
    # plt.show()
    plt.savefig(filename+'.png')
    plt.savefig(filename+'.svg')


def plot_for_pub(exps, met_plot_map, algo_name_map=None, colors=None, styles=None, prefix='pub'):

    metric_names = {
        ('delta_hv', 'ratio'): '$\\Delta$HV Ratio',
    }
    if algo_name_map is None:
        algo_name_map = {}

    def _plot_callback(fig, metric_objs, metric_name, value_name, handles, line_titles):
        font = 'Times New Roman'
        fig.set_size_inches(4, 3)
        ax = plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title('')

        ax.set_xlabel('Infills', fontname=font)
        ax.set_ylabel(metric_names.get((metric_name, value_name), f'{metric_name}.{value_name}'), fontname=font)
        ax.tick_params(axis='both', labelsize=7)
        plt.xticks(fontname=font)
        plt.yticks(fontname=font)

        labels = [algo_name_map.get(title, title) for title in line_titles]
        plt.legend(loc='lower center', bbox_to_anchor=(.5, 1), frameon=False, ncol=len(line_titles), handles=handles,
                   labels=labels, prop={'family': font})
        plt.tight_layout()

    results = [exp.get_aggregate_effectiveness_results() for exp in exps]
    base_path = exps[0].get_problem_results_path()
    for metric, metric_values in met_plot_map.items():
        save_filename = f'{base_path}/{secure_filename(prefix+"_"+ metric)}'
        ExperimenterResult.plot_compare_metrics(
            results, metric, plot_value_names=metric_values, plot_evaluations=True, save_filename=save_filename,
            plot_callback=_plot_callback, save_svg=True, colors=colors, styles=styles, show=False)


def plot_for_pub_sb(exps, met_plot_map, algo_name_map=None, prefix='pub_sb', y_log=False, palette=None, zoom=False):
    if algo_name_map is None:
        algo_name_map = {}

    results = [(exp.get_aggregate_effectiveness_results(), exp.plot_name or exp.algorithm_name) for exp in exps]
    base_path = exps[0].get_problem_results_path()
    for metric_base, metric_values in met_plot_map.items():
        for value_col in metric_values:
            for do_zoom in [False, True]:
                if do_zoom and not zoom:
                    continue
                metric_name = _col_names[f'{metric_base}_{value_col}']

                n_eval = [np.array(res.n_eval)-res.n_eval[0] for res, _ in results]
                metrics = [(res.metrics[metric_base], algo_name) for res, algo_name in results]
                data = []
                algo_names = []
                n_evals = []
                y_end = []
                cols = []
                for k, (metric, algo_name) in enumerate(metrics):
                    if metric.values_agg is None:
                        raise ValueError('No aggregate values!')
                    y = np.atleast_1d(metric.values_agg[value_col]['median'])
                    y_q25 = np.atleast_1d(metric.values_agg[value_col]['q25'])
                    y_q75 = np.atleast_1d(metric.values_agg[value_col]['q75'])
                    y_concat = np.concatenate([y, y_q25, y_q75])
                    data.append(y_concat)
                    y_end += [y[-1], y_q25[-1], y_q75[-1]]

                    a_name = algo_name_map.get(algo_name, algo_name)
                    cols.append(a_name)
                    algo_names += [a_name for _ in range(len(y_concat))]
                    n_evals += np.tile(n_eval[k], 3).tolist()
                y_end = np.array(y_end)

                # df = pd.DataFrame(index=np.tile(n_eval[0], 3), data=np.column_stack(data), columns=cols)
                df = pd.DataFrame(data={'y': np.concatenate(data), 'algo': algo_names, 'n_eval': n_evals})
                with sb_theme():
                    if palette is None:
                        palette = sns.color_palette('mako', n_colors=len(cols))
                    plt.figure(figsize=(5, 3))
                    ax = sns.lineplot(data=df, estimator=lambda s: s.iloc[0], errorbar=lambda s: (s.iloc[1], s.iloc[2]),
                                      palette=palette, sort=False, x='n_eval', y='y', hue='algo')
                    ax.set(xlabel='Infill points', ylabel=metric_name)
                    if y_log:
                        ax.set(yscale='log')
                    if do_zoom:
                        min_val, mean_val, max_val = np.min(y_end), np.mean(y_end), np.max(y_end)
                        range_val = (max_val - min_val)*1.2
                        min_val, max_val = mean_val-.5*range_val, mean_val+.5*range_val
                        ax.set_ylim(min_val, max_val)
                        min_x, max_x = np.min(n_evals), np.max(n_evals)
                        ax.set_xlim(np.mean([min_x, max_x]), max_x)
                    sns.despine()
                    sns.move_legend(ax, 'center left', bbox_to_anchor=(1, .5), frameon=False)
                    plt.tight_layout()

                # plt.show()
                postfix = '_zoom' if do_zoom else ''
                save_filename = f'{base_path}/{secure_filename(f"{prefix}_{metric_base}_{value_col}{postfix}")}'
                plt.savefig(save_filename+postfix+'.png')
                plt.savefig(save_filename+postfix+'.svg')


@contextlib.contextmanager
def sb_theme():
    sns.set_theme(context='paper', style='ticks', font='serif')
    yield


def plot_perf_rank(df: pd.DataFrame, cat_col: str, cat_name_map=None, idx_name_map=None, prefix=None, save_path=None,
                   add_counts=True, hide_ranks=False, n_col_split=None, n_col_idx=1, h_factor=.3, quant_perf_col=None):
    prefix = '' if prefix is None else f'{prefix}_'
    rank_col = prefix+'perf_rank'
    if rank_col not in df.columns:
        raise RuntimeError('First run analyze_perf_rank!')

    if idx_name_map is None:
        idx_name_map = {}
    if cat_name_map is None:
        cat_name_map = {}
    df['idx'] = [idx_name_map.get(v[0], v[0]) for v in df.index]
    df['cat_names'] = [cat_name_map.get(v, v) for v in df[cat_col]]
    df_rank = df.pivot(index='cat_names', columns='idx', values=rank_col)
    df_perf = df.pivot(index='cat_names', columns='idx', values=quant_perf_col) if quant_perf_col is not None else None
    qpc_name = 'Penalty'

    if len(cat_name_map) > 0:
        cat_names = [cat_name for cat_name in cat_name_map.values() if cat_name in df_rank.index]
        df_rank = df_rank.loc[cat_names]
        df_perf = df_perf.loc[cat_names] if df_perf is not None else None
    if len(idx_name_map) > 0:
        idx_names = [idx_name for idx_name in idx_name_map.values() if idx_name in df_rank.columns]
        df_rank = df_rank[idx_names]
        df_perf = df_perf[idx_names] if df_perf is not None else None

    df_cnts = df_counts_an = None
    df_rank_latex = df_rank
    df_rank_orig = df_rank.copy()
    i_best = i_best_idx = None
    n_col_counts = 2
    vmax_perf = 100
    if add_counts:
        n_idx = len(df_rank.columns)
        df_rank['Rank 1'] = (df_rank_orig == 1).sum(axis=1)
        df_rank['Rank $\leq$ 2'] = (df_rank_orig <= 2).sum(axis=1)

        df_cnts = df_rank.copy()
        df_cnts.iloc[:, :-2] = np.nan
        df_cnts *= 100/n_idx

        np_counts = df_cnts.iloc[:, -2:].values
        i_good = np.where(np_counts[:, 1] == np.max(np_counts[:, 1]))[0]
        i_best = i_good[np_counts[i_good, 0] == np.max(np_counts[i_good, 0])]
        print(f'Best perf ({rank_col}): {df_rank.index[i_best].values}')
        i_best_idx = df_rank.index[i_best].values

        if df_perf is not None:
            ref_perf = df_perf.iloc[i_best, :].mean(axis=0)
            mean_rel_perf = ((df_perf / ref_perf).mean(axis=1) - 1)*100
            vmax_perf = min(100, max(20, np.max(mean_rel_perf)))
            df_cnts[qpc_name] = mean_rel_perf
            df_rank[qpc_name] = mean_rel_perf*np.nan
            n_col_counts += 1
        df_counts_an = df_cnts.applymap(lambda v: f'{v:.0f}%').to_numpy()

        df_rank.iloc[:, -n_col_counts:] = np.nan
        df_rank_latex = pd.concat([df_rank.iloc[:, :-n_col_counts], df_cnts.iloc[:, -n_col_counts:]], axis=1)
        df_rank_latex.index = [val.replace(' & ', '} & \\underline{') if i in i_best else val
                               for i, val in enumerate(df_rank_latex.index)]

    if save_path:
        rank_columns = df_rank_latex.columns
        count_columns = None
        perf_columns = None
        if add_counts:
            rank_columns = df_rank_latex.columns[:-n_col_counts]
            count_columns = df_rank_latex.columns[-n_col_counts:]
            if df_perf is not None:
                perf_columns = count_columns[2:]
                count_columns = count_columns[:2]
            if hide_ranks:
                rank_columns = []
                df_rank_latex = df_rank_latex.iloc[:, -n_col_counts:]
        count_perf_columns = (list(count_columns) or []) + (list(perf_columns) or [])

        has_idx_name = False
        if not isinstance(n_col_idx, int):
            df_rank_latex.index.name = ' & '.join(n_col_idx)
            has_idx_name = True
            n_col_idx = len(n_col_idx)
        if n_col_split is None:
            n_col_split = len(df_rank_latex.columns)+n_col_idx

        col_fmt = 'l'*n_col_idx+'c'*len(df_rank_latex.columns)
        buffer = io.StringIO()
        for i_start_col in range(-n_col_idx, len(df_rank_latex.columns), n_col_split):
            if i_start_col >= 0:
                buffer.write('\\vspace{15pt}\n')

            df_sub = df_rank_latex.iloc[:, max(0, i_start_col):i_start_col+n_col_split]
            s = df_sub.style
            if not has_idx_name:
                s.hide(names=True)
            s.hide(names=True, axis=1)
            if i_start_col >= 0:
                s.hide(axis='index')
            else:
                s.format_index(lambda s: s.replace('%', '\\%'), axis=0)

            if count_columns is not None:
                columns = [col for col in df_sub.columns if col in count_columns]
                if len(columns) > 0:
                    s.format('{:.0f}\%', subset=columns)
                    s.background_gradient(cmap='Blues', subset=columns, vmin=0, vmax=100)
            if perf_columns is not None:
                columns = [col for col in df_sub.columns if col in perf_columns]
                if len(columns) > 0:
                    s.format('{:.0f}\%', subset=columns)
                    s.background_gradient(cmap='Greens_r', subset=columns, vmin=0, vmax=vmax_perf)

            sub_rank_columns = [col for col in df_sub.columns if col in rank_columns]
            if len(sub_rank_columns) > 0:
                s.background_gradient(cmap='Greens_r', subset=sub_rank_columns, vmin=1, vmax=max(df_rank_orig.max()))
                s.format('{:.0f}', subset=sub_rank_columns)

            if i_best is not None:
                if add_counts:
                    s.set_properties(subset=pd.IndexSlice[df_sub.index[i_best], df_sub.columns[-n_col_counts:]],
                                     **{'underline': '--rwrap--latex'})

                def style_idx_(s):
                    styles = np.array(['']*len(s), dtype=object)
                    styles[i_best] = 'underline: --rwrap--latex;'
                    return styles
                s.apply_index(style_idx_)

            s.to_latex(buffer, hrules=True, convert_css=True, column_format=col_fmt[i_start_col+n_col_idx:][:n_col_split])

        with open(save_path+'.tex', 'w') as fp:
            buffer.seek(0)
            fp.write(buffer.read())

    h = .5+len(df_rank)*h_factor
    w = 1+len(df_rank.columns)*1.2

    if i_best is not None:
        df_rank.index = idx = [f'>>>>> {val} <<<<<' if i in i_best else val for i, val in enumerate(df_rank.index)]
        if df_cnts is not None:
            df_cnts.index = idx

    with sb_theme():
        cmap = sns.light_palette('seagreen', reverse=True, as_cmap=True)
        cmap_cnt = sns.light_palette('b', as_cmap=True)
        cmap_perf = sns.light_palette('seagreen', as_cmap=True, reverse=True)
        plt.figure(figsize=(w, h))

        df_rank.index = [val.replace('$', '') for val in df_rank.index]
        ax = sns.heatmap(df_rank, annot=True, fmt='.0f', linewidths=0, cmap=cmap, cbar=False)
        if df_cnts is not None:
            if df_perf is not None:
                df_perf_heatmap = df_cnts.copy()
                df_perf_heatmap.iloc[:, :-1] = np.nan
                sns.heatmap(df_perf_heatmap, annot=df_counts_an, fmt='s', linewidths=0, cmap=cmap_perf,
                            cbar=False, vmin=0, vmax=vmax_perf)
                df_cnts.iloc[:, -1:] = np.nan

            df_cnts.index = [val.replace('$', '') for val in df_cnts.index]
            sns.heatmap(df_cnts, annot=df_counts_an, fmt='s', linewidths=0, cmap=cmap_cnt,
                        cbar=False, vmin=0, vmax=100)

        ax.set(xlabel="", ylabel="")
        sns.despine()
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path+'.png')
        plt.savefig(save_path+'.svg')
    else:
        plt.show()
    return i_best_idx


def plot_multi_idx_lines(df, folder, y_col, sort_by=None, multi_col=None, multi_col_titles=None, prob_names=None,
                         x_ticks=None, save_prefix=None, x_label='', y_log=False, y_fmt=None, legend_title=None,
                         height=2, aspect=1.5, cat_colors=False, y_lims=None, y_names=None):
    if sort_by is not None:
        df = df.sort_values(sort_by)
        if len(df.index.levels[0]) > 1:
            df = df.sort_index(level=0)

    if prob_names is None:
        prob_names = {}
    df['idx0'] = [prob_names.get(val[0], val[0]) for val in df.index]
    df['idx1'] = [val[1] for val in df.index]

    if multi_col is not None and multi_col_titles is not None:
        df[multi_col] = [multi_col_titles.get(val, val) for val in df[multi_col]]

    grp = df.groupby('idx0', group_keys=False)
    if multi_col is not None:
        grp = df.groupby(['idx0', multi_col], group_keys=False)
    grouped = grp.apply(lambda df_: pd.Series(index=df_.index, data=np.arange(len(df_))))
    df['x'] = grouped.iloc[0, :] if isinstance(grouped, pd.DataFrame) else grouped

    if x_ticks is None:
        x_ticks = {}
    cat_unique = df['idx1'].unique()
    if multi_col is not None:
        cat_unique = df[df[multi_col] == df[multi_col].unique()[0]]['idx1'].unique()
    x_ticks = [x_ticks.get(val, val) for val in cat_unique]

    y_cols_list = y_col if isinstance(y_col, list) else [y_col]
    if y_names is None:
        y_names = [_col_names[y_col_] for y_col_ in y_cols_list]
    y_log_list = y_log if isinstance(y_log, list) else [y_log]*len(y_cols_list)

    df_q25 = df.copy()
    df_q75 = df.copy()
    for y_col_ in y_cols_list:
        df_q25[y_col_] = df_q25[y_col_+'_q25'] if y_col_+'_q25' in df.columns else df_q25[y_col_]
        df_q75[y_col_] = df_q75[y_col_+'_q75'] if y_col_+'_q75' in df.columns else df_q75[y_col_]
    df = pd.concat([df, df_q25, df_q75], axis=0)

    n_colors = len(df.index.levels[0])

    kwargs = {}
    y_col_plot = y_col
    row_var = None
    if isinstance(y_col, list):
        id_vars = ['x', 'idx0']
        if multi_col is not None:
            id_vars.append(multi_col)
        df = pd.melt(df, id_vars=id_vars, value_vars=y_col, var_name='var', value_name='value')
        y_col_plot = 'value'
        row_var = 'var'
        kwargs['facet_kws'] = dict(sharey='row')

    with sb_theme():
        if n_colors > 1:
            palette = sns.color_palette('cubehelix' if cat_colors else 'mako_r', n_colors=n_colors)
        else:
            palette = sns.cubehelix_palette(light=0, n_colors=n_colors)
        g = sns.relplot(data=df, kind='line', x='x', y=y_col_plot, hue='idx0', style='idx0', legend=False if legend_title is False else 'brief',
                        estimator=lambda s: s.iloc[0], errorbar=lambda s: (s.iloc[1], s.iloc[2]),
                        sort=False, col=multi_col, row=row_var, palette=palette, height=height, aspect=aspect, **kwargs)

        g.set(xlabel=x_label)
        if multi_col is not None and multi_col_titles is not None:
            g.set_titles(col_template='{col_name}', template='{col_name}')
        elif len(y_cols_list) > 1:
            g.set_titles(template='')

        for i_row, row in enumerate(g.axes):
            for ax in row:
                if y_log_list[i_row]:
                    ax.set(yscale='log')
                if i_row > 0:
                    ax.set_title('')
                if y_lims is not None:
                    ax.set_ylim(*y_lims[i_row])
            row[0].set_ylabel(y_names[i_row])

        if legend_title is not False:
            g._legend.set_title(legend_title or '')
        if y_fmt is not None:
            for ax in g.axes.flat:
                ax.yaxis.set_major_formatter(tkr.StrMethodFormatter(y_fmt))
                ax.yaxis.set_minor_formatter(tkr.NullFormatter())

        plt.xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks)
        sns.despine()

    # plt.show()
    save_prefix = f'_{save_prefix}' if save_prefix is not None else ''
    save_path = f'{folder}/line{save_prefix}_{"_".join(y_cols_list)}'
    plt.savefig(save_path+'.png')
    plt.savefig(save_path+'.svg')


def analyze_perf_rank(df: pd.DataFrame, perf_col: str, n_repeat: int, perf_min=True, prefix=None, df_subset=None):
    df_sub = df[df_subset] if df_subset is not None else df

    def _expand(ser_grp_res, is_bool=False):
        if df_subset is None:
            return ser_grp_res, ser_grp_res
        expanded_series = pd.Series(
            index=df.index, data=np.zeros((len(df),), dtype=bool) if is_bool else np.zeros((len(df),))*np.nan)
        expanded_series[df_subset] = ser_grp_res.iloc[0, :] if isinstance(ser_grp_res, pd.DataFrame) else ser_grp_res
        return expanded_series, ser_grp_res

    prefix = '' if prefix is None else f'{prefix}_'
    df[prefix+'perf_rank'], df_sub[prefix+'perf_rank'] = _expand(df_sub.groupby(level=0, axis=0, group_keys=False).apply(
        lambda x: get_ranks(x, perf_col, n_repeat, perf_min=perf_min)))
    df[prefix+'is_best'], df_sub[prefix+'is_best'] = _expand(df_sub[prefix+'perf_rank'] == 1)
    df[prefix+'n_is_best'], _ = _expand(df_sub.groupby(level=1, axis=0, group_keys=False).apply(
        lambda x: count_bool(x, prefix+'is_best')))

    df[prefix+'is_good'], df_sub[prefix+'is_good'] = _expand(df_sub[prefix+'perf_rank'] <= 2)
    df[prefix+'n_is_good'], _ = _expand(df_sub.groupby(level=1, axis=0, group_keys=False).apply(
        lambda x: count_bool(x, prefix+'is_good')))

    df[prefix+'is_bad'], df_sub[prefix+'is_bad'] = _expand(df_sub[prefix+'perf_rank'] >= 4)
    df[prefix+'n_is_bad'], _ = _expand(df_sub.groupby(level=1, axis=0, group_keys=False).apply(
        lambda x: count_bool(x, prefix+'is_bad')))
    return df


def get_ranks(df: pd.DataFrame, col: str, n_samples: int, perf_min=True):
    from scipy.stats.distributions import norm
    from scipy.stats import ttest_ind_from_stats

    mean_val = df[col].values
    n = norm(0, 1)
    std_val = (df[col+'_q75'].values-df[col+'_q25'].values)/(n.ppf(.75)-n.ppf(.25))

    ranks = np.zeros((len(df),), dtype=int)
    i_compare = np.argmin(mean_val) if perf_min else np.argmax(mean_val)
    ranks[i_compare] = 1

    while np.any(ranks == 0):
        not_compared = np.where(ranks == 0)[0]
        j_compare = not_compared[np.argmin(mean_val[not_compared]) if perf_min else np.argmax(mean_val[not_compared])]

        p = ttest_ind_from_stats(mean_val[i_compare], std_val[i_compare], n_samples,
                                 mean_val[j_compare], std_val[j_compare], n_samples,
                                 equal_var=False).pvalue

        if p <= .10:  # Means are not the same: increase rank count
            ranks[j_compare] = ranks[i_compare]+1
            i_compare = j_compare
        else:
            ranks[j_compare] = ranks[i_compare]

    return pd.Series(index=df.index, data=ranks)


def count_bool(df: pd.DataFrame, col: str):
    n = df[col].sum()
    return pd.Series(index=df.index, data=[n]*len(df))
