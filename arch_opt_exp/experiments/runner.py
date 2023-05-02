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
import shutil
import logging
import warnings
import arch_opt_exp
from typing import *
import concurrent.futures
import matplotlib.pyplot as plt
from sb_arch_opt.sampling import *
from sb_arch_opt.problem import ArchOptProblemBase
from arch_opt_exp.experiments.metrics_base import *
from arch_opt_exp.experiments.experimenter import *
from arch_opt_exp.metrics.performance import *
from werkzeug.utils import secure_filename

from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.initialization import Initialization

__all__ = ['run', 'set_results_folder', 'get_experimenters', 'run_effectiveness_multi',
           'plot_effectiveness_results', 'calc_initial_hv', 'capture_log']

log = logging.getLogger('arch_opt_exp.runner')
warnings.filterwarnings("ignore")


def calc_initial_hv(problem: ArchOptProblemBase):
    pop = Initialization(HierarchicalSampling(repair=problem.get_repair())).do(problem, 1000)
    Evaluator().eval(problem, pop)
    f = pop.get('F')

    hv = DeltaHVMetric(problem.pareto_front()).calculate_delta_hv(f)[2]
    pre, post = '', ''
    if hv == 0:  # https://stackoverflow.com/a/287944
        pre, post = '\033[91m', '\033[0m'
        post = f' ({os.path.basename(problem._pf_cache_path())}){post}'
    print(f'{pre}Initial hypervolume: {hv:.2f}{post}')


def capture_log(level='INFO'):
    Experimenter.capture_log(level)


def run(results_key, problems, algorithms, algo_names, doe=None, plot_names=None, problem_name=None, n_repeat=8,
        n_eval_max=300, do_run=True, return_exp=False, do_plot=True, additional_plot=None, metrics=None):
    set_results_folder(results_key)
    exp = get_experimenters(problems, algorithms, doe=doe, n_eval_max=n_eval_max, algorithm_names=algo_names,
                            plot_names=plot_names, problem_name=problem_name, metrics=metrics)
    if return_exp:
        return exp

    import matplotlib
    matplotlib.use('Agg')
    if do_run:
        run_effectiveness_multi(exp, n_repeat=n_repeat)
    if do_plot:
        plot_metric_values = {
            'delta_hv': ['ratio', 'regret'],
            # 'IGD': None,
            # 'spread': ['delta'],
            'max_cv': None,
            # 'sm_quality': ['rmse', 'loo_cv'] if include_loo_cv else ['rmse'],
            # 'training': ['n_train', 'n_samples', 'time_train'],
            # 'infill': ['time_infill'],
        }
        if additional_plot is not None:
            plot_metric_values.update(additional_plot)
        plot_effectiveness_results(exp, plot_metric_values=plot_metric_values, save=True, show=False)
    return exp


def set_results_folder(key: str, sub_key: str = None):
    folder = os.path.join(os.path.dirname(arch_opt_exp.__file__), '..', 'results', key)
    if sub_key is not None:
        folder = os.path.join(folder, sub_key)
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    Experimenter.results_folder = folder

    return folder


def reset_results():
    """Use after using set_results_folder!"""
    folder = Experimenter.results_folder
    if folder is not None:
        shutil.rmtree(folder)


def get_experimenters(problems: Union[List[Problem], Problem], algorithms: Union[List[Algorithm], Algorithm],
                      doe: dict = None, metrics: List[Metric] = None, n_eval_max: Union[int, List[int]] = 300,
                      algorithm_names: Union[List[str], str] = None,
                      plot_names: List[str] = None, problem_name=None) -> List[Experimenter]:
    """Result Experimenter instances corresponding to the algorithms."""
    if isinstance(problems, Problem):
        if not isinstance(algorithms, list):
            raise ValueError('Algorithms must be list!')
        if algorithm_names is None:
            algorithm_names = [None for _ in range(len(algorithms))]
        if plot_names is None:
            plot_names = [None for _ in range(len(algorithms))]

        if not isinstance(n_eval_max, list):
            n_eval_max = [n_eval_max]*len(algorithms)

        return [Experimenter(problems, algorithm, n_eval_max=n_eval_max[i], doe=doe, metrics=metrics,
                             algorithm_name=algorithm_names[i], plot_name=plot_names[i], problem_name=problem_name)
                for i, algorithm in enumerate(algorithms)]

    elif isinstance(algorithms, Algorithm):
        raise ValueError('Algorithms must be list!')

    else:
        if len(algorithms) != len(problems):
            raise ValueError('Algorithms and problem must be same length!')
        if algorithm_names is None:
            algorithm_names = [None for _ in range(len(algorithms))]
        if plot_names is None:
            plot_names = [None for _ in range(len(algorithms))]

        if not isinstance(n_eval_max, list):
            n_eval_max = [n_eval_max]*len(problems)

        return [Experimenter(problem, algorithms[i], n_eval_max=n_eval_max[i], doe=doe, metrics=metrics,
                             algorithm_name=algorithm_names[i], plot_name=plot_names[i], problem_name=problem_name)
                for i, problem in enumerate(problems)]


def run_effectiveness_multi(experimenters: List[Experimenter], n_repeat=12, reset=False, agg_align_end=False):
    """Runs the effectiveness experiment using multiple algorithms, repeated a number of time for each algorithm."""
    Experimenter.capture_log()
    log.info('Running effectiveness experiments: %d algorithms @ %d repetitions (%d total runs)' %
             (len(experimenters), n_repeat, len(experimenters)*n_repeat))

    if reset:
        reset_results()
    for exp in experimenters:
        exp.run_effectiveness_parallel(n_repeat=n_repeat)
        agg_res = exp.get_aggregate_effectiveness_results(force=True, align_end=agg_align_end)

        agg_res.export_pandas().to_pickle(exp.get_problem_algo_results_path('result_agg_df.pkl'))
        agg_res.save_csv(exp.get_problem_algo_results_path('result_agg.csv'))
        agg_res.save_excel(exp.get_problem_algo_results_path('result_agg.xlsx'))


def plot_effectiveness_results(experimenters: List[Experimenter], plot_metric_values: Dict[str, List[str]] = None,
                               save=False, show=True):
    """Plot metrics results generated using run_effectiveness_multi."""
    Experimenter.capture_log()
    results = []
    for exp in experimenters:
        agg_res = exp.get_aggregate_effectiveness_results()
        results.append(agg_res)

    metrics = sorted(results[0].metrics.values(), key=lambda m: m.name)
    if plot_metric_values is None:
        plot_metric_values = {met.name: None for met in metrics}

    base_plot_path = experimenters[0].get_problem_results_path()
    for ii, metric in enumerate(metrics):
        if metric.name not in plot_metric_values:
            continue
        log.info('Plotting metric: %s -> %r' % (metric.name, plot_metric_values.get(metric.name)))

        save_filename = f'{base_plot_path}/{secure_filename("results_"+ metric.name)}'
        ExperimenterResult.plot_compare_metrics(
            results, metric.name, plot_value_names=plot_metric_values.get(metric.name), plot_evaluations=True,
            save_filename=save_filename, show=False)

        # ExperimenterResult.plot_compare_metrics(
        #     results, metric.name, plot_value_names=plot_metric_values.get(metric.name), plot_evaluations=True,
        #     save_filename=os.path.join(experimenters[0].results_folder,
        #                                secure_filename('ns_results_%s' % metric.name)),
        #     std_sigma=0., show=False)

    # for exp in experimenters:
    #     results = exp.get_effectiveness_results()
    #     for i, result in enumerate(results):
    #         save_filename = exp.get_problem_algo_results_path(f'result_{i}_pareto')
    #         result.plot_obj_progress(save_filename=save_filename, show=False)

    if show:
        plt.show()
    elif save:
        plt.close('all')


def run_efficiency_multi(experimenters: List[Experimenter], metric_terminations: List[MetricTermination]):
    Experimenter.capture_log()
    log.info('Running efficiency experiments: %d algorithms' % (len(experimenters),))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        fut = [executor.submit(_run, exp, mt) for exp in experimenters for mt in metric_terminations]
        concurrent.futures.wait(fut)


def _run(exp, mt):
    Experimenter.capture_log()
    exp.run_efficiency_repeated(mt)
    agg_res = exp.get_aggregate_efficiency_results(mt, force=True)

    agg_res.export_pandas().to_pickle(exp.get_problem_algo_metric_results_path(mt, 'result_agg_df.pkl'))
    agg_res.save_csv(exp.get_problem_algo_metric_results_path(mt, 'result_agg.csv'))


def plot_efficiency_results(experimenters: List[Experimenter], metric_terminations: List[MetricTermination],
                            plot_metric_values: Dict[str, List[str]] = None, save=False, show=True):
    """Plot metrics results generated using run_effectiveness_multi."""
    Experimenter.capture_log()
    results = [exp.get_aggregate_effectiveness_results() for exp in experimenters]
    metrics = sorted(results[0].metrics.values(), key=lambda m: m.name)
    if plot_metric_values is None:
        plot_metric_values = {met.name: None for met in metrics}

    mt_names = [mt.metric_name.replace('exp_moving_average', 'ema').replace('steady_performance', 'spi')
                for mt in metric_terminations]

    for j, exp in enumerate(experimenters):
        folder = os.path.join(exp.results_folder, 'eff_'+secure_filename(exp.algorithm_name))
        os.makedirs(folder, exist_ok=True)

        for mt in metric_terminations:
            log.info('Plotting termination: %s / %s' % (exp.algorithm_name, mt.metric_name))

            mt_results = exp.get_list_efficiency_results(mt)
            n_eval = [res.termination.n_eval for res in mt_results]
            save_filename = os.path.join(folder, secure_filename('term_%s' % secure_filename(mt.metric_name)))
            Metric.plot_multiple(
                [res.termination.metric for res in mt_results], n_eval=n_eval, plot_value_names=[mt.value_name],
                save_filename=save_filename, show=False)

            for ii, res in enumerate(mt_results):
                save_filename = os.path.join(
                    folder, secure_filename('details_%s_%d' % (secure_filename(mt.metric_name), ii)))
                res.termination.plot(save_filename=save_filename, show=False)

        eff_results = [exp.get_aggregate_efficiency_results(mt) for mt in metric_terminations]
        for ii, metric in enumerate(metrics):
            if metric.name not in plot_metric_values:
                continue
            log.info('Plotting metric (Pareto): %s / %s -> %r' %
                     (exp.algorithm_name, metric.name, plot_metric_values.get(metric.name)))

            save_filename = os.path.join(folder, secure_filename('pareto_%s' % metric.name))
            plot_value_names = plot_metric_values.get(metric.name)
            if plot_value_names is None:
                plot_value_names = metric.value_names
            for value_name in plot_value_names:
                ExperimenterResult.plot_metrics_pareto(
                    [results[j]]+eff_results,
                    names=['eff']+mt_names,
                    metric1_name_value=('n_eval', ''),
                    metric2_name_value=(metric.name, value_name),
                    save_filename=save_filename, show=False,
                )

    if show:
        plt.show()
    elif save:
        plt.close('all')
