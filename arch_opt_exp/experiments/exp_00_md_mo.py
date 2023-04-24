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
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Union, Dict
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from pymoo.core.population import Population

from sb_arch_opt.problem import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.constrained import *

from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.infill import MinVariancePFInfill, get_default_infill

from arch_opt_exp.md_mo_hier.infill import *
from arch_opt_exp.experiments.runner import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.experiments.experimenter import *
from arch_opt_exp.experiments.metrics import get_exp_metrics

log = logging.getLogger('arch_opt_exp.01_md_mo')
capture_log()

_exp_00_01_folder = '00_md_mo_01_md_gp'
_exp_00_02_folder = '00_md_mo_02_infill'
_exp_00_03_folder = '00_md_mo_03_constraints'


_test_problems = lambda: [
    (Branin(), '00_C_SO'),
    (Rosenbrock(), '00_C_SO'),
    (MDBranin(), '01_MD_SO'),
    (AugmentedMDBranin(), '01_MD_SO'),
    (MunozZunigaToy(), '01_MD_SO'),
    (MOGoldstein(), '02_C_MO'),
    (MORosenbrock(), '02_C_MO'),
    (MDMOGoldstein(), '03_MD_MO'),
    (MDMORosenbrock(), '03_MD_MO'),
    (ArchCantileveredBeam(), '04_C_SO_G'),
    (MDCantileveredBeam(), '05_MD_SO_G'),
    (ArchWeldedBeam(), '06_C_MO_G'),
    (ArchCarside(), '06_C_MO_G'),
    (MDWeldedBeam(), '07_MD_MO_G'),
    (MDCarside(), '07_MD_MO_G'),
]


def _get_metrics(problem):
    metrics = get_exp_metrics(problem, including_convergence=False)+[SBOTimesMetric()]
    additional_plot = {
        'time': ['train', 'infill'],
    }
    return metrics, additional_plot


def exp_00_01_md_gp(post_process=False):
    """
    Test different Gaussian Process models on mixed-discrete problems.
    """
    folder = set_results_folder(_exp_00_01_folder)
    n_infill = 30
    n_repeat = 20

    problems = [(prob, category) for prob, category in _test_problems() if '_SO' in category and '_G' not in category]
    problem_paths = []
    problem_names = []
    problem: Union[ArchOptProblemBase]
    for i, (problem, category) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)

        n_init = int(np.ceil(2*problem.n_var))

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe, doe_delta_hvs = _create_does(problem, n_init, n_repeat)
        log.info(f'DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                 f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')

        metrics, additional_plot = _get_metrics(problem)
        infill = ExpectedImprovementInfill() if problem.n_obj == 1 else MinVariancePFInfill()

        algorithms = []
        algo_names = []
        for (model, norm), model_name in [
            ((ModelFactory.get_kriging_model(), None), 'BO'),
            (ModelFactory(problem).get_md_kriging_model(), 'MD-BO'),
        ]:
            sbo = SBOInfill(model, infill, pop_size=100, termination=100, normalization=norm, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=1, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(model_name)

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                   return_exp=post_process)

        _plot_for_pub(exps, met_plot_map={
            'delta_hv': ['ratio'],
        }, algo_name_map={'BO': 'Continuous GP', 'MD-BO': 'Mixed-discrete GP'})
        plt.close('all')


def _plot_for_pub(exps, met_plot_map, algo_name_map=None, colors=None, styles=None):

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


def _create_does(problem: ArchOptProblemBase, n_doe, n_repeat):
    doe: Dict[int, Population] = {}
    doe_delta_hvs = []
    for i_rep in range(n_repeat):
        for _ in range(10):
            doe_algo = get_doe_algo(n_doe)
            doe_algo.setup(problem)
            doe_algo.run()

            doe_pf = DeltaHVMetric.get_pareto_front(doe_algo.pop.get('F'))
            doe_f = DeltaHVMetric.get_valid_pop(doe_algo.pop).get('F')
            delta_hv = DeltaHVMetric(problem.pareto_front()).calculate_delta_hv(doe_pf, doe_f)[0]
            if delta_hv < 1e-6:
                continue
            break

        doe[i_rep] = doe_algo.pop
        doe_delta_hvs.append(delta_hv)
    return doe, doe_delta_hvs


def exp_00_02_infill(post_process=False):
    """
    Test different infill criteria on single- and multi-objective (mixed-discrete) problems.

    Conclusions:
    - If evaluations can be run in parallel:
      - Single-objective: Ensemble of EI, LCB, PoI with n_batch = n_parallel
      - Multi-objective:  Ensemble of MPoI, MEPoI  with n_batch = n_parallel
    - If no parallelization possible:
      - Single-objective
        - Continuous: Mean function estimate
        - Mixed-discrete: Ensemble of EI, LCB, PoI with n_batch = 1
      - Multi-objective:  Ensemble of MPoI, MEPoI  with n_batch = 1
    """
    folder = set_results_folder(_exp_00_02_folder)
    n_infill = 20
    n_iter_compare_at = 4  # * max(n_batch) = n_infill
    n_repeat = 20

    so_ensemble = [ExpectedImprovementInfill(), LowerConfidenceBoundInfill(alpha=2.), ProbabilityOfImprovementInfill()]
    so_infills = [
        (FunctionEstimateConstrainedInfill(), 'y', 1),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 1),
        (ExpectedImprovementInfill(), 'EI', 1),
        # (ProbabilityOfImprovementInfill(), 'PoI', 1),
        (EnsembleInfill(infills=so_ensemble), 'Ensemble', 1),
        (EnsembleInfill(infills=so_ensemble), 'Ensemble', 2),
        (EnsembleInfill(infills=so_ensemble), 'Ensemble', 5),
    ]

    mo_ensemble = [MinimumPoIInfill(), MinimumPoIInfill(euclidean=True)]  # , LowerConfidenceBoundInfill(alpha=2.)]
    mo_infills = [
        (MinVariancePFInfill(), 'MVPF', 1),
        (MinVariancePFInfill(), 'MVPF', 2),
        (MinVariancePFInfill(), 'MVPF', 5),
        (MinimumPoIInfill(), 'MPoI', 1),
        (MinimumPoIInfill(euclidean=True), 'EMPoI', 1),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 1),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 2),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 5),
        (EnsembleInfill(infills=mo_ensemble), 'Ensemble', 1),
        (EnsembleInfill(infills=mo_ensemble), 'Ensemble', 2),
        (EnsembleInfill(infills=mo_ensemble), 'Ensemble', 5),
    ]

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['n_batch'] = n_batch_ = int(algo_name.split('_')[1])

        row_compare = df_strat.iloc[n_iter_compare_at, :]
        for col_eval_compare, factor in [('delta_hv_ratio', 1), ('delta_hv_regret', n_batch_)]:
            for col in [col_eval_compare, col_eval_compare+'_q25', col_eval_compare+'_q75']:
                strat_data_[f'iter_{col}'] = row_compare[col]/factor

    problems = [(prob, category) for prob, category in _test_problems() if '_G' not in category]
    problem_paths = []
    problem_names = []
    problem: Union[ArchOptProblemBase]
    for i, (problem, category) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)
        if post_process:
            continue

        n_init = int(np.ceil(2*problem.n_var))

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe, doe_delta_hvs = _create_does(problem, n_init, n_repeat)
        log.info(f'DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                 f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')

        metrics, additional_plot = _get_metrics(problem)

        algorithms = []
        algo_names = []
        for infill, infill_name, n_batch in (so_infills if problem.n_obj == 1 else mo_infills):
            model, norm = ModelFactory(problem).get_md_kriging_model()
            sbo = SBOInfill(model, infill, pop_size=100, termination=100, normalization=norm, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=n_batch, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(f'{infill_name}_{n_batch}')

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run)

        _agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)

        _plot_for_pub(exps, met_plot_map={
            'delta_hv': ['ratio'],
        }, algo_name_map={})
        plt.close('all')

    def _add_cols(df_agg_):
        df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        return df_agg_

    df_agg = _agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    strategy_map = {}
    prob_map = {}
    _make_comparison_df(df_agg[~df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='so', strategy_map=strategy_map, prob_map=prob_map)
    _make_comparison_df(df_agg[df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='mo', strategy_map=strategy_map, prob_map=prob_map)
    _make_comparison_df(df_agg[~df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='so', strategy_map=strategy_map, prob_map=prob_map)
    _make_comparison_df(df_agg[df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='mo', strategy_map=strategy_map, prob_map=prob_map)


def exp_00_03_constraints(post_process=False):
    """
    Test different constraint handling strategies on single- and multi-objective (mixed-discrete) problems.
    """
    folder = set_results_folder(_exp_00_03_folder)
    n_infill = 20
    n_iter_compare_at = 4  # * max(n_batch) = n_infill
    n_repeat = 20

    strategies = [
        (MeanConstraintPrediction(), 'g'),
        (ProbabilityOfFeasibility(min_pof=.25), 'PoF_25'),
        (ProbabilityOfFeasibility(min_pof=.5),  'PoF_50'),
        (ProbabilityOfFeasibility(min_pof=.75), 'PoF_75'),
        (UpperTrustBound(tau=1.), 'UTB_10'),
        (UpperTrustBound(tau=2.), 'UTB_20'),
    ]

    def prob_add_cols(strat_data_, df_strat, algo_name):
        return
        # strat_data_['n_batch'] = n_batch_ = int(algo_name.split('_')[1])
        # row_compare = df_strat.iloc[n_iter_compare_at, :]
        # for col_eval_compare, factor in [('delta_hv_ratio', 1), ('delta_hv_regret', n_batch_)]:
        #     for col in [col_eval_compare, col_eval_compare+'_q25', col_eval_compare+'_q75']:
        #         strat_data_[f'iter_{col}'] = row_compare[col]/factor

    problems = [(prob, category) for prob, category in _test_problems() if '_G' in category]
    problem_paths = []
    problem_names = []
    problem: Union[ArchOptProblemBase]
    for i, (problem, category) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)
        if post_process:
            continue

        n_init = int(np.ceil(2*problem.n_var))

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe, doe_delta_hvs = _create_does(problem, n_init, n_repeat)
        log.info(f'DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                 f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')

        metrics, additional_plot = _get_metrics(problem)

        algorithms = []
        algo_names = []
        for strategy, strategy_name in strategies:
            infill, n_batch = get_default_infill(problem, n_parallel=1)
            infill.constraint_strategy = strategy
            model, norm = ModelFactory(problem).get_md_kriging_model()
            sbo = SBOInfill(model, infill, pop_size=100, termination=100, normalization=norm, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=n_batch, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(strategy_name)

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run)

        _agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)

        _plot_for_pub(exps, met_plot_map={
            'delta_hv': ['ratio'],
        }, algo_name_map={})
        plt.close('all')

    def _add_cols(df_agg_):
        df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        return df_agg_

    df_agg = _agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    strategy_map = {}
    prob_map = {}
    _make_comparison_df(df_agg[~df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='so', strategy_map=strategy_map, prob_map=prob_map)
    _make_comparison_df(df_agg[df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='mo', strategy_map=strategy_map, prob_map=prob_map)
    # _make_comparison_df(df_agg[~df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='so', strategy_map=strategy_map, prob_map=prob_map)
    # _make_comparison_df(df_agg[df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='mo', strategy_map=strategy_map, prob_map=prob_map)


def _make_comparison_df(df_agg, column, title, folder, key=None, strategy_map=None, prob_map=None):
    strategy_map = strategy_map or {}
    prob_map = prob_map or {}

    post = f'_{key}' if key is not None else ''
    with pd.ExcelWriter(f'{folder}/compare_{column+post}.xlsx') as writer:
        col_map = {column: title, f'{column}_q25': 'Q1', f'{column}_q75': 'Q3'}
        df_compare = df_agg[[column, f'{column}_q25', f'{column}_q75']].rename(columns=col_map)
        df_compare.to_excel(writer, sheet_name='Compare')

        for col in df_compare.columns:
            df_compare_val = df_compare[col].unstack(level=0)
            col_rename = {prob: prob.split(' ')[1] for prob in df_compare_val.columns}
            col_rename = {key: prob_map.get(value, value) for key, value in col_rename.items()}
            df_compare_val = df_compare_val.rename(columns=col_rename)

            df_compare_val.insert(0, '$n_{batch}$', [int(idx.split('_')[1]) for idx in df_compare_val.index])
            idx_replace = {idx: idx.split('_')[0] for idx in df_compare_val.index}
            idx_replace = {key: strategy_map.get(value, value) for key, value in idx_replace.items()}
            df_compare_val = df_compare_val.rename(index=idx_replace)

            df_compare_val.to_excel(writer, sheet_name=col)


def _agg_prob_exp(problem, problem_path, exps, add_cols_callback=None):
    df_data = []
    for exp in exps:
        with open(exp.get_problem_algo_results_path('result_agg_df.pkl'), 'rb') as fp:
            df_strat = pickle.load(fp)
            strat_data = pd.Series(df_strat.iloc[-1, :], name=exp.algorithm_name)
            if add_cols_callback is not None:
                strat_data_ = add_cols_callback(strat_data, df_strat, exp.algorithm_name)
                if strat_data_ is not None:
                    strat_data = strat_data_
            df_data.append(strat_data)

    df_prob = pd.concat(df_data, axis=1).T
    df_prob.to_pickle(f'{problem_path}/df_problem.pkl')
    with pd.ExcelWriter(f'{problem_path}/df_problem.xlsx') as writer:
        df_prob.to_excel(writer)


def _agg_opt_exp(problem_names, problem_paths, folder, add_cols_callback):
    df_probs = []
    for i, problem_name in enumerate(problem_names):
        problem_path = problem_paths[i]
        try:
            with open(f'{problem_path}/df_problem.pkl', 'rb') as fp:
                df_prob = pickle.load(fp)
        except FileNotFoundError:
            continue

        df_prob = df_prob.set_index(pd.MultiIndex.from_tuples([(problem_name, val) for val in df_prob.index]))
        df_probs.append(df_prob)

    df_agg = pd.concat(df_probs, axis=0)
    df_agg_ = add_cols_callback(df_agg)
    if df_agg_ is not None:
        df_agg = df_agg_

    try:
        with pd.ExcelWriter(f'{folder}/results.xlsx') as writer:
            df_agg.to_excel(writer)
    except PermissionError:
        pass
    return df_agg


if __name__ == '__main__':
    # exp_00_01_md_gp()
    # exp_00_02_infill()
    exp_00_03_constraints()
