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
from typing import Union, Dict
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from pymoo.core.population import Population

from sb_arch_opt.problem import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *

from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.infill import *

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
    """
    folder = set_results_folder(_exp_00_02_folder)
    n_infill = 20
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

    problems = [(prob, category) for prob, category in _test_problems() if '_G' not in category]
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

        _plot_for_pub(exps, met_plot_map={
            'delta_hv': ['ratio'],
        }, algo_name_map={})
        plt.close('all')


if __name__ == '__main__':
    # exp_00_01_md_gp()
    exp_00_02_infill()
