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
import os
import timeit
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Union, Dict
import matplotlib
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from pymoo.problems.multi.zdt import ZDT1
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization

from sb_arch_opt.problem import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.constrained import *
from sb_arch_opt.problems.problems_base import *

from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.infill import MinVariancePFInfill, get_default_infill

from arch_opt_exp.md_mo_hier.infill import *
from arch_opt_exp.experiments.runner import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.experiments.plotting import *
from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.hc_strategies.rejection import *
from arch_opt_exp.hc_strategies.sbo_with_hc import *
from arch_opt_exp.experiments.metrics import get_exp_metrics

from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
from smt.surrogate_models.kpls import KPLS
from smt.surrogate_models.kplsk import KPLSK

log = logging.getLogger('arch_opt_exp.01_md_mo')
capture_log()

_exp_00_01_folder = '00_md_mo_01_md_gp'
_exp_00_02_folder = '00_md_mo_02_infill'
_exp_00_03a_folder = '00_md_mo_03a_plot_g'
_exp_00_03b_folder = '00_md_mo_03b_multi_y'
_exp_00_03_folder = '00_md_mo_03_constraints'
_exp_00_04_folder = '00_md_mo_04_high_dim'


_test_problems = lambda: [
    (Branin(), '00_C_SO', 'Branin'),
    (Rosenbrock(), '00_C_SO', 'Rosenbrock'),
    (MDBranin(), '01_MD_SO', 'MD Branin'),
    (AugmentedMDBranin(), '01_MD_SO', 'Aug. MD Branin'),
    (MunozZunigaToy(), '01_MD_SO', 'MZ-Toy'),
    (MOGoldstein(), '02_C_MO', 'MO Goldstein'),
    (MORosenbrock(), '02_C_MO', 'MO Rosenbrock'),
    (MDMOGoldstein(), '03_MD_MO', 'MD-MO Golds.'),
    (MDMORosenbrock(), '03_MD_MO', 'MD-MO Rosenb.'),
    (ConBraninProd(), '04_C_SO_G', 'C Branin'),  # 1 constr
    (ConBraninGomez(), '04_C_SO_G', 'C Branin (Gomez)'),  # 1 constr
    (ArchCantileveredBeam(), '04_C_SO_G', 'Cant. Beam'),  # 2 constr
    (MDCantileveredBeam(), '05_MD_SO_G', 'MD Cant. Beam'),  # 2 constr
    (ArchOSY(), '06_C_MO_G', 'Osy.-Kundu'),  # 6 constr
    (MDOSY(), '07_MD_MO_G', 'MD Osy.-Kundu'),  # 6 constr
    # (ArchWeldedBeam(), '06_C_MO_G', 'Welded Beam'),
    # (ArchCarside(), '06_C_MO_G', 'Carside'),
    # (MDWeldedBeam(), '07_MD_MO_G', 'MD Welded Beam'),
    # (MDCarside(), '07_MD_MO_G', 'MD Carside'),
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

    Conclusions:
    - MD GP works better than continuous GP
    - Increasing the nr of restarts for training does not necessarily improve results,
      however linearly increases training time
    """
    folder = set_results_folder(_exp_00_01_folder)
    n_infill = 30
    n_repeat = 20

    problems = [(prob, category, title) for prob, category, title in _test_problems()
                if '_SO' in category and '_G' not in category]
    problem_paths = []
    problem_names = []
    problem_name_map = {}
    problem: Union[ArchOptProblemBase]
    for i, (problem, category, title) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_name_map[name] = title
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
            (ModelFactory(problem).get_md_kriging_model(n_start=5), 'MD-BO-5'),
        ]:
            sbo = SBOInfill(model, infill, pop_size=100, termination=100, normalization=norm, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=1, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(model_name)

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                   return_exp=post_process)

        # plot_for_pub(exps, met_plot_map={
        #     'delta_hv': ['ratio'],
        # }, algo_name_map={'BO': 'Continuous GP', 'MD-BO': 'Mixed-discrete GP'})
        plt.close('all')


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
    n_infill = 40
    n_iter_compare_at = 4  # * max(n_batch) = n_infill
    n_repeat = 20

    so_ensemble = [ExpectedImprovementInfill(), LowerConfidenceBoundInfill(alpha=2.), ProbabilityOfImprovementInfill()]
    so_infills = [
        (FunctionEstimateConstrainedInfill(), 'y', 1, 'y-mean'),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 1, 'LCB),
        (ExpectedImprovementInfill(), 'EI', 1, 'EI'),
        # (ProbabilityOfImprovementInfill(), 'PoI', 1, 'PoI'),
        (EnsembleInfill(infills=so_ensemble), 'Ensemble', 1, 'Ensemble'),
        (EnsembleInfill(infills=so_ensemble), 'Ensemble', 2, 'Ensemble (2)'),
        (EnsembleInfill(infills=so_ensemble), 'Ensemble', 5, 'Ensemble (5)'),
    ]

    mo_ensemble = [MinimumPoIInfill(), MinimumPoIInfill(euclidean=True)]  # , LowerConfidenceBoundInfill(alpha=2.)]
    mo_infills = [
        (MinVariancePFInfill(), 'MVPF', 1, 'MVPF'),
        (MinVariancePFInfill(), 'MVPF', 2, 'MVPF (2)'),
        (MinVariancePFInfill(), 'MVPF', 5, 'MVPF (5)'),
        (MinimumPoIInfill(), 'MPoI', 1, 'MPoI'),
        (MinimumPoIInfill(euclidean=True), 'EMPoI', 1, 'EMPoI'),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 1, 'LCB'),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 2, 'LCB (2)'),
        # (LowerConfidenceBoundInfill(alpha=2.), 'LCB', 5, 'LCB (5)'),
        (EnsembleInfill(infills=mo_ensemble), 'Ensemble', 1, 'Ensemble'),
        (EnsembleInfill(infills=mo_ensemble), 'Ensemble', 2, 'Ensemble (2)'),
        (EnsembleInfill(infills=mo_ensemble), 'Ensemble', 5, 'Ensemble (5)'),
    ]

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['n_batch'] = n_batch_ = int(algo_name.split('_')[1])

        row_compare = df_strat.iloc[n_iter_compare_at, :]
        for col_eval_compare, factor in [('delta_hv_ratio', 1), ('delta_hv_regret', n_batch_)]:
            for col in [col_eval_compare, col_eval_compare+'_q25', col_eval_compare+'_q75']:
                strat_data_[f'iter_{col}'] = row_compare[col]/factor

    problems = [(prob, category, title) for prob, category, title in _test_problems() if '_G' not in category]
    problem_paths = []
    problem_names = []
    p_name_map = {}
    problem: Union[ArchOptProblemBase]
    for i, (problem, category, title) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        p_name_map[name] = title
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
        for infill, infill_name, n_batch, _ in (so_infills if problem.n_obj == 1 else mo_infills):
            model, norm = ModelFactory(problem).get_md_kriging_model(multi=True)
            sbo = SBOInfill(model, infill, pop_size=100, termination=100, normalization=norm, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=n_batch, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(f'{infill_name}_{n_batch}')

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                   run_if_exists=False)

        _agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)

        plot_for_pub(exps, met_plot_map={
            'delta_hv': ['ratio'],
        }, algo_name_map={})
        plt.close('all')

    def _add_cols(df_agg_):
        df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        df_agg_['is_md'] = ['_MD_' in val[0] for val in df_agg_.index]
        df_agg_['infill'] = [val[1] for val in df_agg_.index]

        df_agg_so = analyze_perf_rank(df_agg_[~df_agg_.is_mo].copy(), 'delta_hv_regret', n_repeat)
        df_agg_mo = analyze_perf_rank(df_agg_[df_agg_.is_mo].copy(), 'delta_hv_regret', n_repeat)

        df_agg_so = analyze_perf_rank(df_agg_so, 'iter_delta_hv_regret', n_repeat, prefix='iter')
        df_agg_mo = analyze_perf_rank(df_agg_mo, 'iter_delta_hv_regret', n_repeat, prefix='iter')

        df_agg_ = pd.concat([df_agg_so, df_agg_mo])
        return df_agg_

    df_agg = _agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    def _mod_compare(df_compare_val, col):
        df_compare_val.insert(0, '$n_{batch}$', [int(idx.split('_')[1]) for idx in df_compare_val.index])
        idx_replace = {idx: idx.split('_')[0] for idx in df_compare_val.index}
        idx_replace = {key: strategy_map.get(value, value) for key, value in idx_replace.items()}
        return df_compare_val.rename(index=idx_replace)

    strategy_map = {}
    prob_map = {}
    kwargs = dict(strategy_map=strategy_map, prob_map=prob_map, mod_compare=_mod_compare)
    _make_comparison_df(df_agg[~df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='so', **kwargs)
    _make_comparison_df(df_agg[df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='mo', **kwargs)
    _make_comparison_df(df_agg[~df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='so', **kwargs)
    _make_comparison_df(df_agg[df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='mo', **kwargs)

    so_cat_map = {f'{infill_name}_{n_batch}': title for _, infill_name, n_batch, title in so_infills}
    mo_cat_map = {f'{infill_name}_{n_batch}': title for _, infill_name, n_batch, title in mo_infills}
    for df_rank, rank_cat, cat_map in [
        (df_agg[~df_agg.is_mo], 'so', so_cat_map),
        (df_agg[~df_agg.is_mo & ~df_agg.is_md], 'so_c', so_cat_map),
        (df_agg[~df_agg.is_mo & df_agg.is_md], 'so_md', so_cat_map),
        (df_agg[df_agg.is_mo], 'mo', mo_cat_map),
        (df_agg[df_agg.is_mo & ~df_agg.is_md], 'mo_c', mo_cat_map),
        (df_agg[df_agg.is_mo & df_agg.is_md], 'mo_md', mo_cat_map),
    ]:
        plot_perf_rank(df_rank, 'infill', cat_name_map=cat_map, idx_name_map=p_name_map,
                       save_path=f'{folder}/rank_{rank_cat}')
        plot_perf_rank(df_rank, 'infill', cat_name_map=cat_map, idx_name_map=p_name_map,
                       prefix='iter', save_path=f'{folder}/rank_{rank_cat}_iter')

    green = matplotlib.cm.get_cmap('Greens')
    blue = matplotlib.cm.get_cmap('Blues')
    orange = matplotlib.cm.get_cmap('Oranges')

    so_cat_names = [inf_data[3] for inf_data in so_infills]
    so_cat_colors = [
        green(.5), orange(.5),
        blue(.25), blue(.5), blue(.75),
    ]
    plot_problem_bars(df_agg[~df_agg.is_mo], folder, 'infill', 'delta_hv_regret', prefix='so', prob_name_map=p_name_map,
                      cat_colors=so_cat_colors, label_rot=0, label_i=2, cat_names=so_cat_names, y_log=True)
    plot_problem_bars(df_agg[~df_agg.is_mo], folder, 'infill', 'iter_delta_hv_regret', prefix='so_iter', prob_name_map=p_name_map,
                      cat_colors=so_cat_colors, label_rot=0, label_i=2, cat_names=so_cat_names, y_log=True)

    mo_cat_names = [inf_data[3] for inf_data in mo_infills]
    mo_cat_colors = [
        green(.25), green(.5), green(.75),
        orange(.33), orange(.66),
        blue(.25), blue(.5), blue(.75),
    ]
    plot_problem_bars(df_agg[df_agg.is_mo], folder, 'infill', 'delta_hv_regret', prefix='mo', prob_name_map=p_name_map,
                      cat_colors=mo_cat_colors, label_rot=0, label_i=3, cat_names=mo_cat_names, y_log=True)
    plot_problem_bars(df_agg[df_agg.is_mo], folder, 'infill', 'iter_delta_hv_regret', prefix='mo_iter', prob_name_map=p_name_map,
                      cat_colors=mo_cat_colors, label_rot=0, label_i=3, cat_names=mo_cat_names, y_log=True)

    plt.close('all')


def exp_00_03a_plot_constraints():
    folder = set_results_folder(_exp_00_03a_folder)
    n_infill = 10

    strategies = [
        (AdaptiveProbabilityOfFeasibility(min_pof_bounds=(.1, .5)), 'APoF'),
        (MeanConstraintPrediction(), 'g'),
        (ProbabilityOfFeasibility(min_pof=.25), 'PoF_25'),
        (ProbabilityOfFeasibility(min_pof=.5),  'PoF_50'),
        (ProbabilityOfFeasibility(min_pof=.75), 'PoF_75'),
        (UpperTrustBound(tau=1.), 'UTB_10'),
        (UpperTrustBound(tau=2.), 'UTB_20'),
    ]

    for problem in [
        ConBraninProd(),
        ConBraninGomez(),
    ]:
        prob_name = problem.__class__.__name__
        n_init = int(2*problem.n_var)
        doe = get_doe_algo(n_init)
        doe.setup(problem)
        doe.run()
        doe_pop = doe.pop
        f_doe = doe_pop.get('F')
        f_doe[doe_pop.get('CV')[:, 0] > 0, :] = np.nan

        pf = problem.pareto_front()
        f_known_best = pf[0, 0]
        log.info(f'Best of initial DOE ({n_init}): {np.nanmin(f_doe[:, 0])} (best: {f_known_best})')

        for i, (strategy, strat_name) in enumerate(strategies):
            log.info(f'Strategy {i+1}/{len(strategies)}: {strat_name}')
            strategy_folder = f'{folder}/{prob_name}_{i:02d}_{secure_filename(strat_name)}'
            os.makedirs(strategy_folder, exist_ok=True)

            infill, n_batch = get_default_infill(problem, n_parallel=1)
            infill.constraint_strategy = strategy
            model, norm = ModelFactory(problem).get_md_kriging_model()
            sbo_infill = HiddenConstraintsSBO(model, infill, pop_size=100, termination=100, normalization=norm,
                                              verbose=False)
            sbo_infill.hc_strategy = RejectionHCStrategy()

            sbo = sbo_infill.algorithm(infill_size=n_batch, init_size=n_init)
            sbo.initialization = Initialization(doe_pop)
            sbo.setup(problem)
            doe_pop = sbo.ask()  # Once to initialize the infill search using the DOE
            sbo.evaluator.eval(problem, doe_pop)
            sbo.tell(doe_pop)

            n_pop, n_fail = [len(doe_pop)], [np.sum(ArchOptProblemBase.get_failed_points(doe_pop))]
            f_best = [np.nanmin(doe_pop.get('F')[:, 0])]
            for i_infill in range(n_infill):
                # # Do the last infill using the mean prediction
                # if i_infill == n_infill-1:
                #     sbo_infill.infill = inf = FunctionEstimateConstrainedInfill()
                #     inf.initialize(sbo_infill.problem, sbo_infill.surrogate_model, sbo_infill.normalization)

                log.info(f'Infill {i_infill+1}/{n_infill}')
                infills = sbo.ask()
                assert len(infills) == 1
                sbo.evaluator.eval(problem, infills)

                if i_infill == 0:
                    sbo_infill.plot_state(save_path=f'{strategy_folder}/doe', plot_g=True, show=False)
                sbo_infill.plot_state(x_infill=infills.get('X')[0, :], plot_std=False, plot_g=True,
                                      save_path=f'{strategy_folder}/infill_{i_infill}', show=False)

                sbo.tell(infills=infills)

                n_pop.append(len(sbo.pop))
                n_fail.append(np.sum(ArchOptProblemBase.get_failed_points(sbo.pop)))
                f_best.append(sbo.opt.get('F')[0, 0])

            plt.figure(), plt.title(f'{strat_name} on {prob_name}')
            plt.plot(f_best, 'k', linewidth=2)
            plt.plot([f_known_best]*len(f_best), '--k', linewidth=.5)
            plt.xlabel('Iteration'), plt.ylabel('Best $f$')
            plt.tight_layout()
            plt.savefig(f'{strategy_folder}/f.png')


def exp_00_03b_multi_y():
    """
    Confirm that supplying multiple columns in the training outputs is the same as training multiple independent GP's.
    """
    folder = set_results_folder(_exp_00_03b_folder)
    n_repeat = 4
    n_doe = 40
    n_test = 10

    problem = MOZDT1()
    doe_algo = get_doe_algo(doe_size=n_doe)
    doe_algo.setup(problem)
    doe_algo.run()
    x = doe_algo.pop.get('X')
    y = doe_algo.pop.get('F')
    y = np.column_stack([y, -y, y*1.5, y*.25, y+10, y-100])
    x_test, y_test = x[-n_test:, :], y[-n_test:, :]
    x, y = x[:-n_test, :], y[:-n_test, :]

    t_train_indep = []
    t_train_agg = []
    rmse_indep = []
    rmse_agg = []
    for i in range(1, y.shape[1]+1):
        t_indep = []
        t_agg = []
        rmse_i = []
        rmse_a = []
        for j in range(n_repeat):
            log.info(f'Training {j+1}/{n_repeat} for n_y = {i}/{y.shape[1]}')

            model = ModelFactory.get_kriging_model()
            s = timeit.default_timer()
            model.set_training_values(x, y[:, :i])
            model.train()
            t_agg.append(timeit.default_timer()-s)

            y_pred = model.predict_values(x_test)
            rmse_a.append(np.sqrt(np.mean((y_pred-y_test[:, :i])**2)))

            models = [ModelFactory.get_kriging_model() for _ in range(i)]
            s = timeit.default_timer()
            for k in range(i):
                models[k].set_training_values(x, y[:, [k]])
                models[k].train()
            t_indep.append(timeit.default_timer()-s)

            y_pred = []
            for k in range(i):
                y_pred.append(models[k].predict_values(x_test))
            rmse_i.append(np.sqrt(np.mean((np.column_stack(y_pred)-y_test[:, :i])**2)))

        t_train_indep.append(np.mean(t_indep))
        t_train_agg.append(np.mean(t_agg))
        rmse_indep.append(np.mean(rmse_i))
        rmse_agg.append(np.mean(rmse_a))

    x_plot = np.arange(1, y.shape[1]+1)
    plt.figure(), plt.title('Training time')
    plt.plot(x_plot, t_train_indep, '-r', linewidth=1, label='Independent')
    plt.plot(x_plot, t_train_agg, '-b', linewidth=1, label='Aggregated')
    plt.xlabel('$n_y$'), plt.ylabel('Training time [s]')
    plt.legend(), plt.tight_layout()
    plt.savefig(f'{folder}/t.png')

    plt.figure(), plt.title('Accuracy')
    plt.plot(x_plot, rmse_indep, '-r', linewidth=1, label='Independent')
    plt.plot(x_plot, rmse_agg, '-b', linewidth=1, label='Aggregated')
    plt.xlabel('$n_y$'), plt.ylabel('RMSE')
    plt.legend(), plt.tight_layout()
    plt.savefig(f'{folder}/rmse.png')


def exp_00_03_constraints(post_process=False):
    """
    Test different constraint handling strategies on single- and multi-objective (mixed-discrete) problems.

    Conclusions:
    - UTB (or PoF < 50%) works well for highly-constrained problems
    - Non-aggregated PoF 75% works best; PoF 50% and g-mean work are close
    - g-mean aggregation reduces performance greatly, however at a training time reduction of up to 50%
    - Recommendation: use PoF 75%, except if more or less conservatism is needed; aggregate if training time should be
      reduced
    """
    folder = set_results_folder(_exp_00_03_folder)
    n_infill = 30
    n_iter_compare_at = 4  # * max(n_batch) = n_infill
    n_repeat = 20

    strategies = [
        # strategy, name, aggregate, eliminate, title
        (MeanConstraintPrediction(), 'g', False, False, 'g-mean'),
        (MeanConstraintPrediction(), 'g_agg', True, False, 'g-mean (Agg.)'),
        (MeanConstraintPrediction(), 'g_elim', False, True, 'g-mean (Elim.)'),
        (ProbabilityOfFeasibility(min_pof=.25), 'PoF_25', False, False, 'PoF (25%)'),
        (ProbabilityOfFeasibility(min_pof=.75), 'PoF_75', False, False, 'PoF (75%)'),
        (UpperTrustBound(tau=1.), 'UTB_10', False, False, 'UTB ($\\tau$ = 1)'),
        (UpperTrustBound(tau=2.), 'UTB_20', False, False, 'UTB ($\\tau$ = 2)'),
    ]

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['n_doe'] = n_init
        strat_data_['n_dim'] = problem.n_var
        strat_data_['n_con'] = problem.n_ieq_constr
        strat_data_['n_obj'] = problem.n_obj
        strat_data_['is_mo'] = problem.n_obj > 1
        # strat_data_['n_batch'] = n_batch_ = int(algo_name.split('_')[1])
        # row_compare = df_strat.iloc[n_iter_compare_at, :]
        # for col_eval_compare, factor in [('delta_hv_ratio', 1), ('delta_hv_regret', n_batch_)]:
        #     for col in [col_eval_compare, col_eval_compare+'_q25', col_eval_compare+'_q75']:
        #         strat_data_[f'iter_{col}'] = row_compare[col]/factor

    problems = [(prob, category, title) for prob, category, title in _test_problems() if '_G' in category]
    problem_paths = []
    problem_names = []
    p_name_map = {}
    problem: Union[ArchOptProblemBase]
    for i, (problem, category, title) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        p_name_map[name] = title
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)
        if post_process:
            continue

        n_init = int(np.ceil(2*problem.n_var))
        n_infill_prob = n_infill
        if isinstance(problem, ConBraninGomez):
            n_infill_prob *= 2

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe, doe_delta_hvs = _create_does(problem, n_init, n_repeat)
        log.info(f'DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                 f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')

        metrics, additional_plot = _get_metrics(problem)

        algorithms = []
        algo_names = []
        for strategy, strategy_name, aggregate, eliminate, _ in strategies:
            infill, n_batch = get_default_infill(problem, n_parallel=1)
            infill.constraint_strategy = strategy
            model, norm = ModelFactory(problem).get_md_kriging_model(multi=True)
            sbo = ConstraintAggSBOInfill(
                model, infill, pop_size=100, termination=100, normalization=norm, aggregate_g=aggregate,
                eliminate_g=eliminate, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=n_batch, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(strategy_name)

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill_prob,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                   run_if_exists=True)

        _agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)

        plot_for_pub(exps, met_plot_map={
            'delta_hv': ['ratio'],
        }, algo_name_map={})
        plt.close('all')

    def _add_cols(df_agg_):
        df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        df_agg_['strategy'] = [val[1] for val in df_agg_.index]
        analyze_perf_rank(df_agg_, 'delta_hv_regret', n_repeat)

        df_agg_ = df_agg_[df_agg_.strategy != 'PoF_50']
        return df_agg_

    df_agg = _agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    strategy_map = {}
    prob_map = {}
    _make_comparison_df(df_agg[~df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='so', strategy_map=strategy_map, prob_map=prob_map)
    _make_comparison_df(df_agg[df_agg.is_mo], 'delta_hv_regret', 'Regret', folder, key='mo', strategy_map=strategy_map, prob_map=prob_map)
    # _make_comparison_df(df_agg[~df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='so', strategy_map=strategy_map, prob_map=prob_map)
    # _make_comparison_df(df_agg[df_agg.is_mo], 'iter_delta_hv_regret', 'Regret', folder, key='mo', strategy_map=strategy_map, prob_map=prob_map)

    cat_map = {strategy_name: title for _, strategy_name, _, _, title in strategies}
    plot_perf_rank(df_agg[~df_agg.is_mo], 'strategy', cat_name_map=cat_map, idx_name_map=p_name_map,
                   save_path=f'{folder}/rank_so')
    plot_perf_rank(df_agg[df_agg.is_mo], 'strategy', cat_name_map=cat_map, idx_name_map=p_name_map,
                   save_path=f'{folder}/rank_mo')

    green = matplotlib.cm.get_cmap('Greens')
    blue = matplotlib.cm.get_cmap('Blues')
    orange = matplotlib.cm.get_cmap('Oranges')

    cat_names = [inf_data[4] for inf_data in strategies]
    cat_colors = [
        green(.25), green(.5), green(.75),
        orange(.33), orange(.66),
        blue(.33), blue(.66),
    ]
    plot_problem_bars(df_agg, folder, 'strategy', 'delta_hv_regret', prob_name_map=p_name_map,
                      cat_colors=cat_colors, label_rot=15, label_i=3, cat_names=cat_names, y_log=True)
    # plot_problem_bars(df_agg, folder, 'strategy', 'iter_delta_hv_regret', prefix='iter', prob_name_map=p_name_map,
    #                   cat_colors=cat_colors, label_rot=15, label_i=3, cat_names=cat_names, y_log=True)

    plt.close('all')


def plot_pof_utb():
    sigma_limits = [-3, 3]
    plot_pof = [.25, .5, .75]
    plot_utb = [0, 1, 2]

    from scipy.stats import norm
    n = norm(0, 1)
    sigma = np.linspace(sigma_limits[0], sigma_limits[1], 100)

    plt.figure(), plt.title('PoF vs UTB')
    plt.plot(sigma, n.cdf(sigma), '-k', linewidth=1, label='CDF')

    for pof in plot_pof:
        sigma = n.ppf(pof)
        plt.plot([sigma_limits[0], sigma, sigma], [pof, pof, 0], '--', linewidth=1,
                 label=f'PoF {pof*100:.0f} = UTB {-sigma:.1f}')

    for utb in plot_utb:
        cdf_utb = n.cdf(-utb)
        plt.plot([sigma_limits[0], -utb, -utb], [cdf_utb, cdf_utb, 0], '-.', linewidth=1,
                 label=f'UTB {utb:.1f} = PoF {cdf_utb*100:.0f}')

    plt.xlabel('$\sigma$'), plt.ylabel('CDF'), plt.ylim([0, 1]), plt.xlim(sigma_limits), plt.legend()
    plt.show()


def _make_comparison_df(df_agg, column, title, folder, key=None, strategy_map=None, prob_map=None, mod_compare=None):
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

            if mod_compare is not None:
                df_compare_val_ = mod_compare(df_compare_val, col)
                if df_compare_val_ is not None:
                    df_compare_val = df_compare_val_
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


def _build_md_kriging(problem: ArchOptProblemBase, surrogate_factory):
    from sb_arch_opt.algo.arch_sbo.models import MultiSurrogateModel
    factory = ModelFactory(problem)
    normalization = factory.get_md_normalization()
    norm_ds_spec = factory.create_smt_design_space_spec(problem.design_space, md_normalize=True)

    surrogate = surrogate_factory(norm_ds_spec.design_space)
    surrogate = MultiSurrogateModel(surrogate)
    # if norm_ds_spec.is_mixed_discrete:
    #     surrogate = MixedIntegerKrigingModel(surrogate=surrogate)
    return surrogate, normalization


def _patch_kpls():
    # https://github.com/jbussemaker/smt/tree/kpls-fix
    estimate_func = KPLS._estimate_number_of_components
    predict_func = KPLS._predict_values

    def _wrapped_predict(self, x):
        y_out = predict_func(self, x)
        return y_out.reshape(x.shape[0], self.ny)

    def _fixed_estimate(self):
        self._predict_values = lambda x: _wrapped_predict(self, x)
        estimate_func(self)
        self._predict_values = lambda x: predict_func(self, x)

    KPLS._estimate_number_of_components = _fixed_estimate


_patch_kpls()


class MDTestZDT1(MixedDiscretizerProblemBase):

    def __init__(self, n_cat: int, n_var=30):
        self._zdt1_n_var = n_var
        self._n_cat = n_cat
        super().__init__(ZDT1(n_var=n_var), n_opts=5, n_vars_int=n_cat, cat=True)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_cat={self._n_cat}, n_var={self._zdt1_n_var})'


def exp_00_04_high_dim(post_process=False):
    """
    Test different time reduction strategies on high-dimensional (mixed-discrete) problems.

    Conclusions:
    - Training time varies linearly with the nr of hyperparameters (n_theta) to tune
      - n_theta is greatly increased by using EHH/HH kernels
      - n_theta increases linearly with the nr of surrogates to train
      - n_theta is reduced linearly for the nr of PLS components (on vars where it is applied)
      - PLS is not applied to categorical vars if EHH/HH kernel is used
    - Gower/CR shows better optimization results than EHH/HH kernels
      - Gower slightly better than CR
    - Applying PLS
      - Reduces optimizer performance slightly
      - Reduces training time for non-categorical variables

    Recommendation:
    - Use Gower Distance (GD) or Continuous Relaxation (CR) by default
    - Apply PLS to reduce training time above ~10 components
    """
    # post_process = True
    folder = set_results_folder(_exp_00_04_folder)
    n_infill = 20
    n_repeat = 12

    # def _get_kpls_factory(is_md_, n_comp: int = None, kplsk=False):
    #     kwargs = {
    #         'print_global': False,
    #         'categorical_kernel': MixIntKernelType.HOMO_HSPHERE,
    #         'hierarchical_kernel': MixHrcKernelType.ALG_KERNEL,
    #     }
    #     if is_md_:
    #         kwargs['n_start'] = 5
    #
    #     def _factory(ds):
    #         if kplsk:
    #             return KPLSK(design_space=ds, **kwargs)
    #
    #         if n_comp is None:
    #             kwargs['eval_n_comp'] = True
    #         else:
    #             kwargs['n_comp'] = n_comp
    #         return KPLS(design_space=ds, **kwargs)
    #
    #     return _factory

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['nx'] = problem.n_var
        strat_data_['n_cat'] = int(category.split('_')[1])
        strat_data_['n_theta'] = model_n_theta[algo_name]

    kw_ehh = dict(categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE)
    kw_cr = dict(categorical_kernel=MixIntKernelType.CONT_RELAX)

    n_cat = [0, 2, 4, 8]
    n_var = 10
    problems = [(
        MDTestZDT1(n_cat=n, n_var=n_var),
        f'ZDT1_{n:02.0f}',
        f'{n:.0f}/{n_var}',
    ) for n in n_cat]
    # problems = [
    #     (MOZDT1(), '02_C_MO'),
    #     (MDZDT1(), '03_MD_MO'),
    #     (DZDT1(), '03_MD_MO'),
    # ]
    problem_paths = []
    problem_names = []
    p_names_map = {}
    problem: Union[ArchOptProblemBase]
    for i, (problem, category, title) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        p_names_map[name] = title
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)
        if post_process:
            continue

        n_init = int(np.ceil(5*problem.n_var))

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe, doe_delta_hvs = _create_does(problem, n_init, n_repeat)
        log.info(f'DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                 f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')

        metrics, additional_plot = _get_metrics(problem)
        infill = ExpectedImprovementInfill() if problem.n_obj == 1 else MinVariancePFInfill()

        # is_md = not np.all(problem.is_cont_mask)

        algorithms = []
        algo_names = []
        model_n_theta = {}
        for (model, norm), model_name in [
            (ModelFactory(problem).get_md_kriging_model(**kw_ehh), 'Krg'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=1, **kw_ehh), 'KPLS-1'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=2, **kw_ehh), 'KPLS-2'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=4, **kw_ehh), 'KPLS-4'),
            # (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=10, **kw_ehh), 'KPLS-10'),
            (ModelFactory(problem).get_md_kriging_model(), 'Krg-Gow'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=1), 'KPLS-1-Gow'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=2), 'KPLS-2-Gow'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=4), 'KPLS-4-Gow'),
            # (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=10), 'KPLS-10-Gow'),
            (ModelFactory(problem).get_md_kriging_model(**kw_cr), 'Krg-CR'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=1, **kw_cr), 'KPLS-1-CR'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=2, **kw_cr), 'KPLS-2-CR'),
            (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=4, **kw_cr), 'KPLS-4-CR'),
            # (ModelFactory(problem).get_md_kriging_model(kpls_n_comp=10), 'KPLS-10-CR'),

            # (_build_md_kriging(problem, _get_kpls_factory(is_md, n_comp=1)), 'KPLS-1'),
            # (_build_md_kriging(problem, _get_kpls_factory(is_md, n_comp=2)), 'KPLS-2'),
            # (_build_md_kriging(problem, _get_kpls_factory(is_md, n_comp=4)), 'KPLS-4'),
            # (_build_md_kriging(problem, _get_kpls_factory(is_md, n_comp=10)), 'KPLS-10'),
            # (_build_md_kriging(problem, _get_kpls_factory(is_md, n_comp=None)), 'KPLS-auto'),
            # (_build_md_kriging(problem, _get_kpls_factory(is_md, kplsk=True)), 'KPLSK'),
        ]:
            sbo = SBOInfill(model, infill, normalization=norm, verbose=False)
            sbo_algo = sbo.algorithm(infill_size=1, init_size=n_init)
            algorithms.append(sbo_algo)
            algo_names.append(model_name)
            model_n_theta[model_name] = ModelFactory.get_n_theta(problem, model)

        do_run = not post_process
        exps = run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                   run_if_exists=False)

        _agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)

        # plot_for_pub(exps, met_plot_map={
        #     'delta_hv': ['ratio'],
        # }, algo_name_map={'BO': 'Continuous GP', 'MD-BO': 'Mixed-discrete GP'})
        plt.close('all')

    def _add_cols(df_agg_):
        df_agg_['surr'] = [val[1].split('-')[0] for val in df_agg_.index]
        df_agg_['is_pls'] = ['KPLS' in val[1] for val in df_agg_.index]
        df_agg_['cat_ker'] = ['Gower' if 'Gow' in val[1] else 'CR' if 'CR' in val[1] else 'EHH'
                              for val in df_agg_.index]
        df_agg_['is_ck_lt'] = [1 if 'Gow' in val[1] else .5 if 'CR' in val[1] else 0 for val in df_agg_.index]

        nx = df_agg_['nx'].values
        df_agg_['n_comp'] = [int(val[1].split('-')[1]) if 'KPLS' in val[1] else nx[ii]
                             for ii, val in enumerate(df_agg_.index)]

        return df_agg_

    df_agg = _agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    # plot_scatter(df_agg, folder, 'delta_hv_ratio', 'time_train', z_col='n_comp', x_log=True, y_log=True)
    # plot_scatter(df_agg, folder, 'delta_hv_ratio', 'time_infill', z_col='n_comp', x_log=True, y_log=True)
    plot_scatter(df_agg, folder, 'delta_hv_regret', 'time_train', z_col='n_comp', y_log=True)
    plot_scatter(df_agg, folder, 'delta_hv_regret', 'time_infill', z_col='n_comp', y_log=True)
    plot_scatter(df_agg, folder, 'delta_hv_regret', 'time_train', z_col='is_ck_lt', y_log=True)
    plot_scatter(df_agg, folder, 'delta_hv_regret', 'time_infill', z_col='is_ck_lt', y_log=True)
    plot_scatter(df_agg, folder, 'delta_hv_regret', 'time_train', z_col='n_cat', y_log=True)
    plot_scatter(df_agg, folder, 'delta_hv_regret', 'time_infill', z_col='n_cat', y_log=True)
    plot_scatter(df_agg, folder, 'n_theta', 'time_train', z_col='n_cat', x_log=True, y_log=True)
    plot_scatter(df_agg, folder, 'n_theta', 'time_infill', z_col='n_cat', x_log=True, y_log=True)

    x_ticks_map = {val: val.split("-")[1] if 'KPLS' in val else 'No KPLS'
                   for val in df_agg.index.levels[1]}
    x_label = '$n_{kpls}$'
    mc_titles = {'Gower': 'Gower Distance', 'CR': 'Continuous Relaxation'}

    df_agg_multi = df_agg[df_agg.cat_ker != 'EHH']
    kwargs = dict(sort_by='n_comp', multi_col='cat_ker', multi_col_titles=mc_titles,
                  prob_names=p_names_map, x_ticks=x_ticks_map, x_label=x_label, legend_title='Cat. $x$')
    plot_multi_idx_lines(df_agg_multi, folder, 'delta_hv_regret', y_fmt='{x:.0f}', **kwargs)
    plot_multi_idx_lines(df_agg_multi, folder, 'time_train', y_log=True, y_fmt='{x:.0f}', **kwargs)
    plot_multi_idx_lines(df_agg_multi, folder, 'time_infill', y_log=True, y_fmt='{x:.0f}', **kwargs)
    plot_multi_idx_lines(df_agg_multi, folder, ['delta_hv_regret', 'time_train', 'time_infill'],
                         y_log=[False, True, True], y_fmt='{x:.0f}', **kwargs)

    # for df_agg_sub, prefix in [
    #     (df_agg[df_agg.cat_ker == 'Gower'], 'gower'),
    #     (df_agg[df_agg.cat_ker == 'CR'], 'cr'),
    # ]:
    #     plot_multi_idx_lines(df_agg_sub, folder, 'delta_hv_regret', sort_by='n_comp', prob_names=p_names_map,
    #                          x_ticks=x_ticks_map, save_prefix=prefix, x_label=x_label)


if __name__ == '__main__':
    # exp_00_01_md_gp()
    # exp_00_02_infill()
    # exp_00_03a_plot_constraints()
    # exp_00_03b_multi_y()
    # exp_00_03_constraints()
    exp_00_04_high_dim()
