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
import copy
import pickle
import timeit
import logging
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization

from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.turbofan_arch import *

from sb_arch_opt.algo.arch_sbo import *
from sb_arch_opt.algo.tpe_interface import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.pymoo_interface.random_search import *

from arch_opt_exp.metrics.performance import *
from arch_opt_exp.experiments.runner import *
from arch_opt_exp.experiments.metrics import *
from arch_opt_exp.experiments.plotting import *
from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.md_mo_hier.sampling import *
from arch_opt_exp.md_mo_hier.hier_problems import *
from arch_opt_exp.md_mo_hier.infill import *
from arch_opt_exp.md_mo_hier.naive import *
from arch_opt_exp.experiments.exp_01_sampling import agg_opt_exp, agg_prob_exp
from arch_opt_exp.experiments.experimenter import Experimenter

log = logging.getLogger('arch_opt_exp.02_hier')
capture_log()

_exp_02_01_folder = '02_hier_01_tpe'
_exp_02_02_folder = '02_hier_02_strategies'
_exp_02_02a_folder = '02_hier_02a_model_fit'
_exp_02_03_folder = '02_hier_03_sensitivities'
_exp_02_04_folder = '02_hier_04_dv_examples'


def _create_does(problem: ArchOptProblemBase, n_doe, n_repeat, sampler=None, repair=None, evaluator=None, seed=None):
    doe: Dict[int, Population] = {}
    doe_delta_hvs = []
    for i_rep in range(n_repeat):
        for _ in range(10):
            doe_algo = get_doe_algo(n_doe)
            if sampler is not None or repair is not None:
                doe_algo.initialization = Initialization(
                    sampler if sampler is not None else doe_algo.initialization.sampling,
                    repair=repair if repair is not None else doe_algo.repair,
                    eliminate_duplicates=LargeDuplicateElimination(),
                )
            if seed is not None:  # Works with the HierarchicalSampler
                np.random.seed(seed+i_rep)
            if evaluator is not None:
                doe_algo.evaluator = evaluator
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


def _get_metrics(problem):
    metrics = get_exp_metrics(problem, including_convergence=False) +\
              [SBOTimesMetric()]
    additional_plot = {
        'time': ['train', 'infill'],
    }
    return metrics, additional_plot


def exp_02_01_tpe():
    """
    Check if Tree Parzen Estimators (TPE) might be a good alternative for hierarchical optimization.

    Conclusion: GP is better on the test problems.
    """
    post_process = False
    folder = set_results_folder(_exp_02_01_folder)
    n_infill = 10
    n_repeat = 8
    problems = [Branin(), Rosenbrock(), MDBranin(), AugmentedMDBranin(), Jenatton(), HierBranin()]
    for i, problem in enumerate(problems):
        name = f'{problem.__class__.__name__}'
        # problem_path = f'{folder}/{secure_filename(name)}'

        # Rule of thumb: k*n_dim --> corrected for expected fail rate (unknown before running a problem, of course)
        n_init = int(np.ceil(2*problem.n_var))

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe, doe_delta_hvs = _create_does(problem, n_init, n_repeat)
        log.info(f'DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                 f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')

        metrics, additional_plot = _get_metrics(problem)

        algorithms = []
        algo_names = []

        algorithms.append(RandomSearchAlgorithm(n_init=n_init))
        algo_names.append('RS')

        algorithms.append(TPEAlgorithm(n_init=n_init))
        algo_names.append('TPE')

        algorithms.append(get_arch_sbo_gp(problem, init_size=n_init))
        algo_names.append('SBO')

        do_run = not post_process
        run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
            metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
            return_exp=post_process)
        plt.close('all')


def exp_02_02_hier_strategies(sbo=False):
    """
    Demonstrate that at least some level of optimizer integration is needed, by comparing the following levels of
    hierarchy integration:
    - 0 naive: all hierarchy information is ignored, i.e. the problem is treated as a regular mixed-discrete problem
    - 1 x_out: corrected/imputed x is accepted as output of the evaluation calls
    - 2 repair: a repair operator is available (i.e. correction outside the evaluation call is also possible)
                --> e.g. correction can be used for sampling and infill
    - 3 activeness: also activeness information is returned, so the full integration is possible,
                    --> e.g. hierarchical sampling, hierarchical Kriging kernels

    Conclusions:
    - For the non-hierarchical problem, the integration level does not matter; activeness even performs worse
    - For hierarchical problems, a higher level of integration improves performance
      - Especially from level 2 (repair) performance greatly improves
      - For SBO, activeness (3) doesn't necessarily perform better than repair (2)
    """
    post_process = False
    folder_post = 'sbo' if sbo else 'nsga2'
    folder = set_results_folder(f'{_exp_02_02_folder}_{folder_post}')
    n_infill = 100
    n_gen = 50
    n_repeat = 16 if sbo else 100
    doe_k = 10
    # n_sub, n_opts = 8, 2
    n_sub, n_opts = 9, 3
    i_sub_opt = None
    prob_data = {}

    def prob_add_cols(strat_data_, df_strat, algo_name):
        data_key = name
        if data_key in prob_data:
            for key, value in prob_data[data_key].items():
                strat_data_[key] = value
            return

        discrete_rates = problem.get_discrete_rates(force=True)

        prob_data[data_key] = data = {
            'is_mo': problem.n_obj > 1,
            'imp_ratio': problem.get_imputation_ratio(),
            'imp_ratio_d': problem.get_discrete_imputation_ratio(),
            'imp_ratio_c': problem.get_continuous_imputation_ratio(),
            'n_discr': problem.get_n_valid_discrete(),
            'n_sub': n_sub,
            'n_doe': n_init,
            'max_dr': discrete_rates.loc['diversity'].max(),
            'max_adr': discrete_rates.loc['active-diversity'].max(),
        }
        for key, value in data.items():
            strat_data_[key] = value

    problems = [
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts, imp_ratio=1., diversity_range=0), '00_SO_NO_HIER', 'Branin'),
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts, diversity_range=0), '01_SO_LDR', 'Branin (H)'),
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts), '02_SO_HDR', 'Branin (H/MRD)'),  # High diversity range
        # # (lambda: HierarchicalGoldstein(), '02_SO_HDR', 'Goldstein (H/MRD)'),
        (lambda: SelectableTunableZDT1(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts), '03_MO_HDR', 'ZDT1 (H/MRD)'),
    ]
    # for i, (problem_factory, _, _) in enumerate(problems):
    #     problem_factory().print_stats()
    # exit()

    problem_paths = []
    problem_names = []
    p_name_map = {}
    i_prob = 0
    problem: ArchOptProblemBase
    for i, (problem_factory, category, title) in enumerate(problems):
        problem = problem_factory()
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        p_name_map[name] = title
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)
        if post_process:
            continue

        n_init = int(np.ceil(doe_k*problem.n_var))
        n_kpls = None
        # n_kpls = n_kpls if problem.n_var > n_kpls else None
        i_prob += 1
        log.info(f'Running optimizations for {i_prob}/{len(problems)}: {name} '
                 f'(n_init = {n_init}, n_kpls = {n_kpls})')
        problem.pareto_front()

        metrics, additional_plot = _get_metrics(problem)
        additional_plot['delta_hv'] = ['ratio', 'regret', 'delta_hv', 'abs_regret']

        algo_names, prob_and_settings = zip(*[
            ('00_naive', (NaiveProblem(problem), False)),
            ('01_x_out', (NaiveProblem(problem, return_mod_x=True), False)),
            ('02_repair', (NaiveProblem(problem, return_mod_x=True, correct=True), False)),
            ('03_act_md_gp', (NaiveProblem(problem, return_mod_x=True, correct=True, return_activeness=True), True)),
            ('03_activeness', (NaiveProblem(problem, return_mod_x=True, correct=True, return_activeness=True), False)),
        ])

        sampler = lambda: ActiveVarHierarchicalSampling(weight_by_nr_active=True)
        # sampler = lambda: ActiveVarHierarchicalSampling()
        algo_models = []
        if sbo:
            n_eval_max = n_infill
            infill, n_batch = get_default_infill(problem)
            # infill.select_improve_infills = False
            algorithms = []
            for problem_, ignore_hierarchy in prob_and_settings:
                from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
                kwargs = dict(
                    kpls_n_comp=n_kpls,
                    ignore_hierarchy=ignore_hierarchy,
                    # categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
                    # hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
                )
                model, norm = ModelFactory(problem_).get_md_kriging_model(**kwargs)
                algo_models.append((model, norm))
                algorithms.append(get_sbo(model, infill, infill_size=n_batch, init_size=n_init, normalization=norm,
                                          init_sampling=sampler()))
        else:
            pop_size = n_init
            n_eval_max = (n_gen-1)*pop_size
            algorithms = [ArchOptNSGA2(pop_size=pop_size, sampling=sampler()) for _ in range(len(prob_and_settings))]

        doe = {}
        problems = [entry[0] for entry in prob_and_settings]
        for j, problem_ in enumerate(problems):
            doe_prob, doe_delta_hvs = _create_does(problem_, n_init, n_repeat, sampler=sampler(), seed=42)
            log.info(f'Naive DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                     f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')
            doe[algo_names[j]] = doe_prob

        do_run = not post_process
        exps = run(folder, problems, algorithms, algo_names, n_repeat=n_repeat, n_eval_max=n_eval_max, doe=doe,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                   run_if_exists=False)
        agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)

        # # Investigate model fitting qualities
        # if sbo:
        #     algo_map = {name: i for i, name in enumerate(algo_names)}
        #     md_i, hier_i = algo_map.get('03_act_md_gp'), algo_map.get('03_activeness')
        #     if md_i is not None and hier_i is not None:
        #         doe_pops = []
        #         for i_prob, algo_name in [(md_i, '03_act_md_gp'), (hier_i, '03_activeness')]:
        #             doe_pops.append([Evaluator().eval(problems[i_prob], doe_i) for doe_i in doe[algo_name].values()])
        #         _compare_first_last_model_fit([exps[md_i], exps[hier_i]], algo_models, doe_pops, ['MD GP', 'Hier GP'])

        plt.close('all')

    strat_map = {'00_naive': 'Naive', '01_x_out': 'X out', '02_repair': 'Repair', '03_act_md_gp': 'Hier sampl.',
                 '03_activeness': 'Activeness'}

    def _add_cols(df_agg_):
        # df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        df_agg_['strategy'] = [strat_map.get(val[1], val[1]) for val in df_agg_.index]
        analyze_perf_rank(df_agg_, 'delta_hv_abs_regret', n_repeat)
        return df_agg_

    df_agg = agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    cat_name_map = {val: val for val in strat_map.values()}
    plot_perf_rank(df_agg, 'strategy', idx_name_map=p_name_map, cat_name_map=cat_name_map, save_path=f'{folder}/rank')


def _compare_first_last_model_fit(exps: List[Experimenter], algo_models, does: List[List[Population]],
                                  exp_names: List[str]):
    prob_folder = exps[0].get_problem_results_path()
    df_path = f'{prob_folder}/compare_models'
    n_train = 10

    if not os.path.exists(df_path):
        data = {'name': [], 'stage': [], 'loocv': [], 'time_train': []}
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for i, exp in enumerate(exps):
                model, norm = algo_models[i]
                futures = []
                for j, doe_pop in enumerate(does[i]):
                    log_str = f'Model fit: {exp.problem_name} {exp.algorithm_name}; DOE {j+1}/{len(does[i])}'
                    futures.append(executor.submit(_do_train_cv, exp.problem, log_str, model, norm, doe_pop, n_train))
                n_doe = len(futures)

                for j, eff_res in enumerate(exp.get_effectiveness_results()):
                    log_str = f'Model fit: {exp.problem_name} {exp.algorithm_name}; END {j+1}'
                    futures.append(executor.submit(_do_train_cv, exp.problem, log_str, model, norm, eff_res.pop, n_train))
                n_end = len(futures)

                other_exp = exps[i-1]
                for j, eff_res in enumerate(other_exp.get_effectiveness_results()):
                    log_str = f'Model fit: {other_exp.problem_name} {other_exp.algorithm_name}; END {j+1} (OTHER)'
                    futures.append(executor.submit(
                        _do_train_cv, other_exp.problem, log_str, model, norm, eff_res.pop, n_train))

                concurrent.futures.wait(futures)
                for j, fut in enumerate(futures):
                    loocv_max, time_train = fut.result()

                    stage = 'DoE' if j < n_doe else ('End' if j < n_end else 'End (other model)')
                    data['name'].append(exp_names[i])
                    data['stage'].append(stage)
                    data['loocv'].append(loocv_max)
                    data['time_train'].append(time_train)

        df = pd.DataFrame(data=data)
        df.to_pickle(df_path)
        df.to_csv(df_path+'.csv')
    else:
        with open(df_path, 'rb') as fp:
            df = pickle.load(fp)

    import seaborn as sns
    import matplotlib.pyplot as plt
    with sb_theme():
        sns.boxplot(x='stage', y='loocv', hue='name', data=df, palette=['b', 'y'], gap=.1)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()

    plt.savefig(f'{df_path}.png')
    plt.savefig(f'{df_path}.svg')
    # plt.show()


def _do_train_cv(problem, log_str, model, norm, pop: Population, n_train):
    # Following code should have same behavior as arch_sbo/algo.py::_build_model
    x = pop.get('X')
    y = pop.get('F')

    y_min, y_max = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
    y_norm = y_max-y_min
    y_norm[y_norm < 1e-6] = 1e-6
    y_train = (y-y_min)/y_norm

    x_train, is_active = problem.correct_x(x)
    x_train = norm.forward(x_train)

    return _do_cv(log_str, model, x_train, y_train, n_train=n_train, is_active=is_active)


def exp_02_02a_model_fit(post_process=False):
    """
    Test fitted models for the different hierarchy integration strategies.
    """
    # post_process = True
    from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
    Experimenter.capture_log()

    folder = set_results_folder(_exp_02_02a_folder)
    df_path = f'{folder}/results'

    k_doe = [4, 10]  # [1, 10, 20]
    n_sample = 8
    n_train = 5
    # n_sub, n_opts = 8, 2
    n_sub, n_opts = 9, 3
    i_sub_opt = None
    add_close_pts = False
    # i_sub_opt = n_sub-1
    problems = [
        # (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts, imp_ratio=1., diversity_range=0), '00_SO_NO_HIER', 'Branin'),
        # (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts, diversity_range=0), '01_SO_LDR', 'Branin (H)'),
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts), '02_SO_HDR', 'Branin (H/MRD)'),  # High diversity range
        (lambda: SelectableTunableZDT1(n_sub=n_sub, i_sub_opt=i_sub_opt, n_opts=n_opts), '03_MO_HDR', 'ZDT1 (H/MRD)'),
        # (lambda: HierarchicalGoldstein(), '02_SO_HDR', 'Goldstein (H/MRD)'),
        # (lambda: HierarchicalRosenbrock(), '02_SO_HDR', 'Rosenbrock (H/MRD)'),
        # (lambda: HierCantileveredBeam(), '02_SO_HDR', 'Cant. Beam (H/MRD)'),
        # (lambda: SimpleTurbofanArch(), '04_REAL_S', 'Simple Jet'),
        # (lambda: RealisticTurbofanArch(), '04_REAL_L', 'Real Jet'),
    ]
    # for problem_factory, _, _ in problems:
    #     problem_factory().print_stats()
    # exit()

    if post_process and not os.path.exists(df_path):
        post_process = False
    if not post_process:
        data = {
            'problem': [], 'prob_title': [], 'type': [], 'type_int': [],
            'k_doe': [], 'n_doe': [], 'cat_ker': [], 'hier_ker': [], 'loocv': [], 'time_train': [],
        }
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for base_problem_factory, problem_name, title in problems:
                base_problem = base_problem_factory()
                is_jet = isinstance(base_problem, (SimpleTurbofanArch, RealisticTurbofanArch))
                repair_problem = NaiveProblem(base_problem, return_mod_x=True, correct=True)
                act_problem = NaiveProblem(base_problem, return_mod_x=True, correct=True, return_activeness=True)
                for int_name, problem, cat_kernel, hier_kernel in [
                    # ('00_naive', NaiveProblem(base_problem), MixIntKernelType.GOWER, MixHrcKernelType.ALG_KERNEL),
                    # ('01_x_out', NaiveProblem(base_problem, return_mod_x=True), MixIntKernelType.GOWER, MixHrcKernelType.ALG_KERNEL),
                    ('02_repair_gd', repair_problem, MixIntKernelType.GOWER, MixHrcKernelType.ALG_KERNEL),
                    # ('02_repair_cr', repair_problem, MixIntKernelType.CONT_RELAX, MixHrcKernelType.ALG_KERNEL),
                    # ('02_repair_ehh', repair_problem, MixIntKernelType.EXP_HOMO_HSPHERE, MixHrcKernelType.ALG_KERNEL),
                    ('03_activeness_gd_alg', act_problem, MixIntKernelType.GOWER, MixHrcKernelType.ALG_KERNEL),
                    # ('03_activeness_gd_arc', act_problem, MixIntKernelType.GOWER, MixHrcKernelType.ARC_KERNEL),
                    ('03_activeness_cr_alg', act_problem, MixIntKernelType.CONT_RELAX, MixHrcKernelType.ALG_KERNEL),
                    # ('03_activeness_cr_arc', act_problem, MixIntKernelType.CONT_RELAX, MixHrcKernelType.ARC_KERNEL),
                    ('03_activeness_ehh_alg', act_problem, MixIntKernelType.EXP_HOMO_HSPHERE, MixHrcKernelType.ALG_KERNEL),
                    # ('03_activeness_ehh_arc', act_problem, MixIntKernelType.EXP_HOMO_HSPHERE, MixHrcKernelType.ARC_KERNEL),
                ]:
                    if is_jet and not (int_name.startswith('03_') or int_name.startswith('02_')):
                        continue
                    base_model, norm = ModelFactory(problem).get_md_kriging_model(
                        categorical_kernel=cat_kernel, hierarchical_kernel=hier_kernel, multi=False,
                        ignore_hierarchy=False,
                    )
                    for k in k_doe:
                        futures = []
                        for i_sample in range(n_sample):
                            log_str = f'{problem_name} {title} {int_name} k={k} {i_sample+1}/{n_sample}'

                            n_doe = int(k*problem.n_var)
                            if isinstance(base_problem, (SimpleTurbofanArch, RealisticTurbofanArch)):
                                x, f, g = base_problem._load_evaluated()
                                is_failed = base_problem.get_failed_points({'F': f, 'G': g})
                                pop = Population.new(X=x[~is_failed, :], F=f[~is_failed, :], G=g[~is_failed, :])
                                if len(pop) > n_doe:
                                    i_sel = np.random.choice(range(len(pop)), n_doe, replace=False)
                                    pop = pop[i_sel]
                            else:
                                sampler = HierarchicalSampling()
                                # sampler = ActiveVarHierarchicalSampling()
                                pop = sampler.do(problem, n_samples=n_doe)
                                if add_close_pts:
                                    i_pop_ref = np.random.choice(range(len(pop)))
                                    x_ref = pop.get('X')[i_pop_ref, :]
                                    i_cont, = np.where(problem.is_cont_mask)
                                    if len(i_cont) > 0:
                                        i_cont = i_cont[:2]
                                        scl, n_add = .005, 6
                                        dxx = np.meshgrid(*[np.linspace(-scl, scl, n_add) for _ in range(len(i_cont))])
                                        dx = np.column_stack([dxx_.ravel() for dxx_ in dxx])
                                        dx *= problem.xu[i_cont]-problem.xl[i_cont]
                                        x_add = np.repeat(np.array([x_ref]), dx.shape[0], axis=0)
                                        x_add[:, i_cont] += dx
                                        pop = Population.new(X=np.row_stack([pop.get('X'), x_add]))
                                        pop = LargeDuplicateElimination().do(pop)

                                pop = Evaluator().eval(problem, pop)

                            # Following code should have same behavior as arch_sbo/algo.py::_build_model
                            x = pop.get('X')
                            y = pop.get('F')

                            y_min, y_max = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
                            y_norm = y_max-y_min
                            y_norm[y_norm < 1e-6] = 1e-6
                            y_train = (y-y_min)/y_norm

                            x_train, is_active = problem.correct_x(x)
                            x_train = norm.forward(x_train)
                            futures.append(executor.submit(
                                _do_cv, log_str, base_model, x_train, y_train, n_train=n_train, is_active=is_active))

                        concurrent.futures.wait(futures)
                        for fut in futures:
                            loocv_max, time_train = fut.result()

                            data['problem'].append(problem_name)
                            data['prob_title'].append(title)
                            data['type'].append(int_name)
                            data['type_int'].append('_'.join(int_name.split('_')[:2]))
                            data['k_doe'].append(k)
                            data['n_doe'].append(n_doe)
                            data['cat_ker'].append(cat_kernel.name[:3])
                            data['hier_ker'].append(hier_kernel.name.split('_')[0])
                            data['loocv'].append(loocv_max)
                            data['time_train'].append(time_train)

        df = pd.DataFrame(data=data)
        df.to_pickle(df_path)
        df.to_csv(df_path+'.csv')
    else:
        with open(df_path, 'rb') as fp:
            df = pickle.load(fp)

    df['ker'] = [f'{val} {df.hier_ker.values[i]}' for i, val in enumerate(df['cat_ker'])]

    import seaborn as sns
    import matplotlib.pyplot as plt
    with sb_theme():
        df_plot = df
        # df_plot = df_plot[(df_plot.type_int == '02_repair') | (df_plot.type_int == '03_activeness')]
        palette = sns.color_palette("hls", len(df_plot['type_int'].unique()))
        g = sns.relplot(kind='line', data=df_plot, x='k_doe', y='loocv',
                        hue='type_int', style='ker', palette=palette,
                        col='prob_title', col_wrap=2, estimator='median', errorbar=('pi', 50))
        g.set(xscale='log', yscale='log')
        sns.despine()

    plt.savefig(f'{folder}/compare.png')
    plt.savefig(f'{folder}/compare.svg')
    plt.show()


def _do_cv(log_str, surrogate_model, xt: np.ndarray, yt: np.ndarray, n_train: int = None, is_active: np.ndarray = None):
    log.info(log_str)
    s = timeit.default_timer()
    loocv = cross_validate(surrogate_model, xt, yt, n_train=n_train, is_active=is_active)
    time_train = (timeit.default_timer()-s)/n_train
    loocv_max = np.max(loocv)
    return loocv_max, time_train


def cross_validate(surrogate_model, xt: np.ndarray, yt: np.ndarray, n_train: int = None,
                   is_active: np.ndarray = None) -> np.ndarray:
    if n_train is None:
        n_train = xt.shape[0]
    if n_train > xt.shape[0]:
        n_train = xt.shape[0]

    i_leave_out = np.random.choice(xt.shape[0], n_train, replace=False)
    errors = np.empty((n_train, yt.shape[1]))
    for i, i_lo in enumerate(i_leave_out):
        errors[i, :] = _get_error(surrogate_model, xt, yt, i_lo, is_active=is_active)

    rmse = np.sqrt(np.mean(errors**2, axis=0))
    return rmse


def _get_error(surrogate_model, xt: np.ndarray, yt: np.ndarray, i_leave_out, is_active: np.ndarray = None) -> np.ndarray:
    x_lo = xt[i_leave_out, :]
    y_lo = yt[i_leave_out, :]
    is_active_lo = is_active[[i_leave_out], :] if is_active is not None else None
    xt = np.delete(xt, i_leave_out, axis=0)
    yt = np.delete(yt, i_leave_out, axis=0)
    is_active = np.delete(is_active, i_leave_out, axis=0)

    surrogate_model_copy = copy.deepcopy(surrogate_model)
    surrogate_model_copy.set_training_values(xt, yt, is_acting=is_active)
    surrogate_model_copy.train()

    y_lo_predict = surrogate_model_copy.predict_values(np.atleast_2d(x_lo), is_acting=is_active_lo)
    return y_lo_predict-y_lo


def exp_02_03_sensitivities(sbo=False, mrd=False):
    """
    Investigate sensitivity of imputation ratio and max rate diversity on optimizer performance.

    Conclusions:
    - More integration is better, especially for higher imputation ratios or rate diversities
    - For NSGA2 more integration is always better, for SBO it is more apparent for higher values
    - Adding activeness does not seem to affect the results much
    """
    post_process = False
    folder_post = 'rate_div' if mrd else 'imp_ratio'
    folder_post += '_sbo' if sbo else '_nsga2'
    folder = set_results_folder(f'{_exp_02_03_folder}_{folder_post}')
    n_infill = 100
    n_gen = 50
    n_repeat = 12 if sbo else 100
    doe_k = 5
    n_sub = 16
    i_sub_opt = n_sub-1

    if mrd:
        rate_divs = [0, .25, .5, .75, .95]
        imp_ratios = [60]*len(rate_divs)
    else:
        imp_ratios = [1, 2, 10, 20, 60, 200]
        rate_divs = [.8]*len(imp_ratios)
        rate_divs[0] = 0
        rate_divs[1] = .4
        # Actual imp ratios: 1, 2, 9, 18, 54, 108

    prob_data = {}
    problems = [
        (SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, imp_ratio=imp_ratio, diversity_range=rate_divs[j], n_opts=4),
         f'MRD_{int(rate_divs[j]*100):03d}' if mrd else f'IR_{imp_ratio:03d}')
        for j, imp_ratio in enumerate(imp_ratios)]
    # for i, (problem, category) in enumerate(problems):
    #     problem.print_stats()
    # exit()

    def prob_add_cols(strat_data_, df_strat, algo_name):
        data_key = name
        if data_key in prob_data:
            for key, value in prob_data[data_key].items():
                strat_data_[key] = value
            return

        discrete_rates = problem.get_discrete_rates(force=True)

        prob_data[data_key] = data = {
            'is_mo': problem.n_obj > 1,
            'imp_ratio': problem.get_imputation_ratio(),
            'imp_ratio_d': problem.get_discrete_imputation_ratio(),
            'imp_ratio_c': problem.get_continuous_imputation_ratio(),
            'n_discr': problem.get_n_valid_discrete(),
            'n_sub': n_sub, 'n_doe': n_init,
            'mrd': discrete_rates.loc['diversity'].max(),
            'mrd_act': discrete_rates.loc['active-diversity'].max(),
        }
        for key, value in data.items():
            strat_data_[key] = value

    problem_paths = []
    problem_names = []
    i_prob = 0
    problem: ArchOptProblemBase
    for i, (problem, category) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)
        # if post_process:
        #     continue

        n_init = int(np.ceil(doe_k*problem.n_var))
        n_kpls = None
        # n_kpls = n_kpls if problem.n_var > n_kpls else None
        i_prob += 1
        log.info(f'Running optimizations for {i_prob}/{len(problems)}: {name} '
                 f'(n_init = {n_init}, n_kpls = {n_kpls})')
        problem.pareto_front()

        metrics, additional_plot = _get_metrics(problem)
        additional_plot['delta_hv'] = ['ratio', 'regret', 'delta_hv', 'abs_regret']

        algo_names, hier_sampling, problems = zip(*[
            ('00_naive', False, NaiveProblem(problem)),
            ('01_x_out', False, NaiveProblem(problem, return_mod_x=True)),
            ('02_repair', False, NaiveProblem(problem, return_mod_x=True, correct=True)),
            ('03_activeness', True, NaiveProblem(problem, return_mod_x=True, correct=True, return_activeness=True)),
        ])

        if sbo:
            n_eval_max = n_infill
            infill, n_batch = get_default_infill(problem)
            algorithms = []
            for problem_ in problems:
                model, norm = ModelFactory(problem_).get_md_kriging_model(kpls_n_comp=n_kpls, ignore_hierarchy=False)
                algorithms.append(get_sbo(model, infill, infill_size=n_batch, init_size=n_init, normalization=norm))
        else:
            pop_size = n_init
            n_eval_max = (n_gen-1)*pop_size
            algorithms = [ArchOptNSGA2(pop_size=pop_size) for _ in range(len(problems))]

        doe = {}
        for j, problem_ in enumerate(problems):
            doe_prob, doe_delta_hvs = _create_does(problem_, n_init, n_repeat)
            log.info(f'Naive DOE Delta HV for {name}: {np.median(doe_delta_hvs):.3g} '
                     f'(Q25 {np.quantile(doe_delta_hvs, .25):.3g}, Q75 {np.quantile(doe_delta_hvs, .75):.3g})')
            doe[algo_names[j]] = doe_prob

        do_run = not post_process
        exps = run(folder, problems, algorithms, algo_names, n_repeat=n_repeat, n_eval_max=n_eval_max, doe=doe,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run)
        agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)
        plt.close('all')

    def _add_cols(df_agg_):
        # df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        return df_agg_

    df_agg = agg_opt_exp(problem_names, problem_paths, folder, _add_cols)
    if mrd:
        _plot_comparison_df(df_agg, 'delta_hv_abs_regret', 'Regret', folder, 'mrd_act', 'Max rate diversity')
    else:
        _plot_comparison_df(df_agg, 'delta_hv_abs_regret', 'Regret', folder, 'imp_ratio', 'Imputation ratio', x_log=True)
    plt.close('all')


def _plot_comparison_df(df_agg, column, title, folder, x_column, x_label, x_log=False):
    plt.figure(figsize=(8, 6))

    x_plot = df_agg[x_column].unstack(level=1).iloc[:, 0].astype(float).values
    df_compare = df_agg[column].unstack(level=1)
    df_compare_q25 = df_agg[column+'_q25'].unstack(level=1)
    df_compare_q75 = df_agg[column+'_q75'].unstack(level=1)
    for label in df_compare.columns:
        l, = plt.plot(x_plot, df_compare[label].values, linewidth=1, label=label)
        plt.fill_between(x_plot, df_compare_q25[label].astype(float).values, df_compare_q75[label].astype(float).values,
                         alpha=.05, color=l.get_color(), linewidth=0)
    if x_log:
        plt.gca().set_xscale('log')
    plt.legend()
    plt.xlabel(x_label), plt.ylabel(title)
    plt.savefig(f'{folder}/compare.png')


def exp_02_04_tunable_hier_dv_examples():
    folder = set_results_folder(_exp_02_04_folder)

    n_opts = 3
    p = lambda n: Branin()
    problems = [
        (TunableHierarchicalMetaProblem(p, imp_ratio=1, n_subproblem=9, diversity_range=0, n_opts=n_opts), 'IR_01_MRD_00'),
        (TunableHierarchicalMetaProblem(p, imp_ratio=10, n_subproblem=9, diversity_range=0, n_opts=n_opts), 'IR_10_MRD_00'),
        (TunableHierarchicalMetaProblem(p, imp_ratio=10, n_subproblem=9, diversity_range=.5, n_opts=n_opts), 'IR_10_MRD_05'),
        (TunableHierarchicalMetaProblem(p, imp_ratio=10, n_subproblem=9, diversity_range=1, n_opts=n_opts), 'IR_10_MRD_MAX'),
        (TunableHierarchicalMetaProblem(p, imp_ratio=1000, n_subproblem=9, diversity_range=1, n_opts=n_opts), 'IR_MAX_MRD_MAX'),
    ]

    x_cart = np.array(list(itertools.product(*[range(2) for _ in range(4)])))
    is_act_cart = (x_cart*0+1).astype(bool)
    is_cont_mask = np.array([False]*4, dtype=bool)

    x_corr = x_cart.copy()
    x_corr[x_corr[:, 0] == 0, 2:] = 0
    is_act_corr = is_act_cart.copy()
    is_act_corr[x_corr[:, 1] == 0, 3] = False

    x_imp = x_corr.copy()
    x_imp[~is_act_corr] = 0
    _, i_unique = np.unique(x_imp, axis=0, return_index=True)

    additional = [
        # x, is_active, is_cont_mask, name
        (x_cart, is_act_cart, is_cont_mask, 'example_0_cartesian'),
        (x_corr, is_act_corr, is_cont_mask, 'example_1_corrected'),
        (x_imp, is_act_cart, is_cont_mask, 'example_2_imputed'),
        (x_imp[i_unique, :], is_act_cart[i_unique, :], is_cont_mask, 'example_3_valid'),
    ]

    def _store_dvs(problem: ArchOptProblemBase, name: str, incl_cont=False):
        x, is_active = problem.all_discrete_x
        assert x is not None
        is_cont_mask = problem.is_cont_mask
        _store_x(x, is_active, is_cont_mask, name, incl_cont=incl_cont)

    def _store_x(x, is_active, is_cont_mask, name: str, incl_cont=False, i_col_name=None):
        data = [[i+1]+[('cont' if is_cont_mask[j] else xij) if is_active[i, j] else ' ' for j, xij in enumerate(xi)
                       if incl_cont or (not incl_cont and not is_cont_mask[j])]
                for i, xi in enumerate(x)]
        x_cols = [f'x{i}' for i in range(len(is_cont_mask) if incl_cont else int(np.sum(~is_cont_mask)))]
        df = pd.DataFrame(data=data, columns=[i_col_name or '$i_{sub}$']+x_cols)
        df.iloc[:, 1:].to_excel(writer, sheet_name=name)

        # https://stackoverflow.com/a/54110153
        worksheet = writer.sheets[name]
        for fmt in formats:
            worksheet.conditional_format(1, 1, len(df), len(df.columns)-1, fmt)

        # Output to Latex
        col_rename_map = {col: f'$x_{{{col[1:]}}}$' for col in df.columns[1:]}
        x_cols = [col_rename_map.get(col, col) for col in x_cols]
        df = df.rename(columns=col_rename_map)
        styler = df.style
        if np.any(df.values == ' '):
            styler.apply(lambda df_: np.where(
                df_.values == ' ', 'background-color:#FFC7CE;color:#9C0006;', ''), axis=None, subset=x_cols)
        if np.any(df.values == 'cont'):
            styler.apply(lambda df_: np.where(
                df_.values == 'cont', f'background-color:{hx(blue(.25))};color:{hx(blue(.75))};', ''),
                         axis=None, subset=x_cols)
        styler.format(lambda v: v if isinstance(v, str) else int(v), subset=x_cols)

        i_max = np.max(x)
        for col in x_cols:
            styler.background_gradient(cmap='Greens', low=.25, high=.5, axis=0, vmin=0, vmax=i_max,
                                       subset=pd.IndexSlice[(df[col] != ' ') & (df[col] != 'cont'), col])

        styler.hide()
        styler.to_latex(f'{folder}/{name}.tex', hrules=True, convert_css=True, column_format='l'*len(df.columns))

    with pd.ExcelWriter(f'{folder}/hierarchical_design_vectors.xlsx', engine='xlsxwriter') as writer:
        hx = matplotlib.colors.rgb2hex
        green = matplotlib.cm.get_cmap('Greens')
        blue = matplotlib.cm.get_cmap('Blues')
        workbook = writer.book
        formats = [
            {'type': 'text', 'criteria': 'containing', 'value': 'inactive',
             'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})},
            {'type': 'text', 'criteria': 'containing', 'value': 'cont',
             'format': workbook.add_format({'bg_color': hx(blue(.25)), 'font_color': hx(blue(.75))})},
        ]
        for ic, frac in enumerate(np.linspace(.25, .5, n_opts)):
            formats.append({'type': 'cell', 'criteria': '=', 'value': ic,
                            'format': workbook.add_format({'bg_color': hx(green(frac))})})

        for problem_, name_ in problems:
            problem_.print_stats()
            _store_dvs(problem_, name_)
            # problem_.get_discrete_rates().to_excel(writer, sheet_name=f'{name_}_rates')

        for args in additional:
            _store_x(*args, i_col_name='$i_{dv}$')


if __name__ == '__main__':
    # from exp_01_sampling import exp_01_06_opt
    # exp_01_06_opt()

    # exp_02_01_tpe()
    # exp_02_02a_model_fit()
    # exp_02_02_hier_strategies()
    # exp_02_02_hier_strategies(sbo=True)
    # exp_02_03_sensitivities()
    # exp_02_03_sensitivities(mrd=True)
    # exp_02_03_sensitivities(sbo=True)
    # exp_02_03_sensitivities(sbo=True, mrd=True)
    exp_02_04_tunable_hier_dv_examples()
