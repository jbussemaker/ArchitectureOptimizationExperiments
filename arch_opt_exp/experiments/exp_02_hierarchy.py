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
import logging
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization

from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.hierarchical import *

from sb_arch_opt.algo.arch_sbo import *
from sb_arch_opt.algo.tpe_interface import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.pymoo_interface.random_search import *

from arch_opt_exp.metrics.performance import *
from arch_opt_exp.experiments.runner import *
from arch_opt_exp.experiments.metrics import *
from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.md_mo_hier.hier_problems import *
from arch_opt_exp.md_mo_hier.naive import *
from arch_opt_exp.experiments.exp_01_sampling import agg_opt_exp, agg_prob_exp

log = logging.getLogger('arch_opt_exp.02_hier')
capture_log()

_exp_02_01_folder = '02_hier_01_tpe'
_exp_02_02_folder = '02_hier_02_strategies'


def _create_does(problem: ArchOptProblemBase, n_doe, n_repeat, sampler=None, repair=None, evaluator=None):
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

        algorithms.append(get_arch_sbo_krg(init_size=n_init, use_ei=True))
        algo_names.append('SBO')

        do_run = not post_process
        run(folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
            metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
            return_exp=post_process)
        plt.close('all')


def exp_02_02_hier_strategies(sbo=False):
    """
    Demonstrate that at least some level of optimizer integration is needed, by comparing naive, imputation, and full
    integration strategies.
    """
    post_process = False
    folder_post = 'sbo' if sbo else 'nsga2'
    folder = set_results_folder(f'{_exp_02_02_folder}_{folder_post}')
    n_infill = 100
    n_gen = 50
    n_repeat = 8 if sbo else 100
    doe_k = 5
    n_sub = 8
    i_sub_opt = n_sub-1
    prob_data = {}

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['sampler_hier'] = is_hier_sampler = algo_name.startswith('Hier')
        strat_data_['sampler_strat'] = algo_name[4:].replace('Wt', '') if is_hier_sampler else 'NoHier'
        strat_data_['sampler_grp_wt'] = 'Wt' in algo_name

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
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, imp_ratio=1., diversity_range=0), '00_SO_NO_HIER'),
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt, diversity_range=0), '01_SO_LDR'),
        (lambda: SelectableTunableBranin(n_sub=n_sub, i_sub_opt=i_sub_opt), '02_SO_HDR'),  # High diversity range
        (lambda: SelectableTunableZDT1(n_sub=n_sub, i_sub_opt=i_sub_opt), '03_MO_HDR'),
    ]
    # for i, (problem_factory, category) in enumerate(problems):
    #     problem_factory().print_stats()
    # exit()

    problem_paths = []
    problem_names = []
    i_prob = 0
    problem: ArchOptProblemBase
    for i, (problem_factory, category) in enumerate(problems):
        problem = problem_factory()
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)

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
            infill, n_batch, agg_g = get_default_infill(problem)
            algorithms = []
            for problem_ in problems:
                model, norm = ModelFactory(problem_).get_md_kriging_model(kpls_n_comp=n_kpls)
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


if __name__ == '__main__':
    # exp_02_01_tpe()
    exp_02_02_hier_strategies()
    exp_02_02_hier_strategies(sbo=True)
