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
from pymoo.core.population import Population

from sb_arch_opt.problem import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.hierarchical import *

from sb_arch_opt.algo.arch_sbo import *
from sb_arch_opt.algo.tpe_interface import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.pymoo_interface.random_search import *

from arch_opt_exp.experiments.runner import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.experiments.metrics import get_exp_metrics

log = logging.getLogger('arch_opt_exp.02_hier')
capture_log()

_exp_02_01_folder = '02_hier_01_tpe'


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


def _get_metrics(problem):
    metrics = get_exp_metrics(problem, including_convergence=False) +\
              [SBOTimesMetric()]
    additional_plot = {
        'time': ['train', 'infill'],
    }
    return metrics, additional_plot


def exp_02_01_tpe():
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


if __name__ == '__main__':
    exp_02_01_tpe()
