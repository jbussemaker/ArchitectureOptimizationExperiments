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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from werkzeug.utils import secure_filename

from arch_opt_exp.experiments.experimenter import *

__all__ = ['plot_scatter', 'plot_problem_bars', 'plot_for_pub', 'analyze_perf_rank']


_col_names = {
    'fail_rate_ratio': 'Fail rate ratio',
    'fail_ratio': 'Fail rate ratio',
    'delta_hv_ratio': 'Delta HV ratio',
    'delta_hv_regret': 'Delta HV regret',
    'delta_hv_abs_regret': 'Delta HV regret',
    'iter_delta_hv_regret': 'Delta HV regret',
    'delta_hv_delta_hv': 'Delta HV',
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
    'time_train': 'Training time',
    'time_infill': 'Infill time',
    'n_comp': 'Model comps',
    'is_ck_lt': 'Is light categorical kernel',
    'n_cat': 'Cat vars',
    'n_theta': 'Hyperparams',
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
        if z.dtype == np.object:
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


def plot_for_pub(exps, met_plot_map, algo_name_map=None, colors=None, styles=None):

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
        save_filename = f'{base_path}/{secure_filename("pub_"+ metric)}'
        ExperimenterResult.plot_compare_metrics(
            results, metric, plot_value_names=metric_values, plot_evaluations=True, save_filename=save_filename,
            plot_callback=_plot_callback, save_svg=True, colors=colors, styles=styles, show=False)


def analyze_perf_rank(df: pd.DataFrame, perf_col: str, n_repeat: int, perf_min=True, prefix=None):
    prefix = '' if prefix is None else f'{prefix}_'
    df[prefix+'perf_rank'] = df.groupby(level=0, axis=0, group_keys=False).apply(
        lambda x: get_ranks(x, perf_col, n_repeat, perf_min=perf_min))
    df[prefix+'is_best'] = df[prefix+'perf_rank'] == 1
    df[prefix+'n_is_best'] = df.groupby(level=1, axis=0, group_keys=False).apply(
        lambda x: count_bool(x, prefix+'is_best'))

    df[prefix+'is_good'] = df[prefix+'perf_rank'] <= 2
    df[prefix+'n_is_good'] = df.groupby(level=1, axis=0, group_keys=False).apply(
        lambda x: count_bool(x, prefix+'is_good'))

    df[prefix+'is_bad'] = df[prefix+'perf_rank'] >= 4
    df[prefix+'n_is_bad'] = df.groupby(level=1, axis=0, group_keys=False).apply(
        lambda x: count_bool(x, prefix+'is_bad'))
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
