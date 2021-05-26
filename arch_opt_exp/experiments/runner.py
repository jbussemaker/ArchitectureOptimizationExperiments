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

Copyright: (c) 2021, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
from arch_opt_exp.experimenter import *
from arch_opt_exp.metrics_base import *
from werkzeug.utils import secure_filename

from pymoo.model.problem import Problem
from pymoo.model.algorithm import Algorithm

__all__ = ['set_results_folder', 'get_experimenters', 'run_effectiveness_multi', 'plot_effectiveness_results']

log = logging.getLogger('arch_opt_exp.runner')
warnings.filterwarnings("ignore")


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


def get_experimenters(problem: Problem, algorithms: List[Algorithm], metrics: List[Metric],
                      n_eval_max: Union[int, List[int]]=500, algorithm_names: List[str] = None) -> List[Experimenter]:
    """Result Experimenter instances corresponding to the algorithms."""
    if algorithm_names is None:
        algorithm_names = [None for _ in range(len(algorithms))]

    if not isinstance(n_eval_max, list):
        n_eval_max = [n_eval_max]*len(algorithms)

    return [Experimenter(problem, algorithm, n_eval_max=n_eval_max[i], metrics=metrics,
                         algorithm_name=algorithm_names[i]) for i, algorithm in enumerate(algorithms)]


def run_effectiveness_multi(experimenters: List[Experimenter], n_repeat=12, reset=False):
    """Runs the effectiveness experiment using multiple algorithms, repeated a number of time for each algorithm."""
    Experimenter.capture_log()
    log.info('Running effectiveness experiments: %d algorithms @ %d repetitions (%d total runs)' %
             (len(experimenters), n_repeat, len(experimenters)*n_repeat))

    if reset:
        reset_results()
    for exp in experimenters:
        exp.run_effectiveness_parallel(n_repeat=n_repeat)
        agg_res = exp.get_aggregate_effectiveness_results()

        agg_res.export_pandas().to_pickle(exp.get_problem_algo_results_path('result_agg_df.pkl'))
        agg_res.save_csv(exp.get_problem_algo_results_path('result_agg.csv'))


def plot_effectiveness_results(experimenters: List[Experimenter], plot_metric_values: Dict[str, List[str]] = None,
                               save=False, show=True):
    """Plot metrics results generated using run_effectiveness_multi."""
    Experimenter.capture_log()
    results = [exp.get_aggregate_effectiveness_results() for exp in experimenters]
    metrics = sorted(results[0].metrics.values(), key=lambda m: m.name)
    if plot_metric_values is None:
        plot_metric_values = {met.name: None for met in metrics}

    for ii, metric in enumerate(metrics):
        if metric.name not in plot_metric_values:
            continue
        log.info('Plotting metric: %s -> %r' % (metric.name, plot_metric_values.get(metric.name)))
        save_filename = os.path.join(experimenters[0].results_folder, secure_filename('results_%s' % metric.name))

        ExperimenterResult.plot_compare_metrics(
            results, metric.name, plot_value_names=plot_metric_values.get(metric.name), plot_evaluations=True,
            save_filename=save_filename, show=False)

        ExperimenterResult.plot_compare_metrics(
            results, metric.name, plot_value_names=plot_metric_values.get(metric.name), plot_evaluations=True,
            save_filename=os.path.join(experimenters[0].results_folder, secure_filename('ns_results_%s' % metric.name)),
            std_sigma=0., show=False)

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
    agg_res = exp.get_aggregate_efficiency_results(mt)

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
            results = exp.get_list_efficiency_results(mt)
            save_filename = os.path.join(folder, secure_filename('term_%s' % secure_filename(mt.metric_name)))
            ExperimenterResult.plot_compare_metrics(
                results, mt.metric_name, plot_value_names=[mt.value_name], plot_evaluations=True,
                save_filename=save_filename, show=False)

        eff_results = [exp.get_aggregate_efficiency_results(mt) for mt in metric_terminations]
        for ii, metric in enumerate(metrics):
            if metric.name not in plot_metric_values:
                continue
            log.info('Plotting metrics: %s, %s -> %r' %
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
