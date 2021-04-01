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


def get_experimenters(problem: Problem, algorithms: List[Algorithm], metrics: List[Metric], n_eval_max=500,
                      algorithm_names: List[str] = None) -> List[Experimenter]:
    """Result Experimenter instances corresponding to the algorithms."""
    if algorithm_names is None:
        algorithm_names = [None for _ in range(len(algorithms))]

    return [Experimenter(problem, algorithm, n_eval_max=n_eval_max, metrics=metrics, algorithm_name=algorithm_names[i])
            for i, algorithm in enumerate(algorithms)]


def run_effectiveness_multi(experimenters: List[Experimenter], n_repeat=12, reset=False):
    """Runs the effectiveness experiment using multiple algorithms, repeated a number of time for each algorithm."""
    Experimenter.capture_log()
    log.info('Running effectiveness experiments: %d algorithms @ %d repetitions (%d total runs)' %
             (len(experimenters), n_repeat, len(experimenters)*n_repeat))

    if reset:
        reset_results()
    for exp in experimenters:
        exp.run_effectiveness_parallel(n_repeat=n_repeat)
        exp.get_aggregate_effectiveness_results()


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

    if show:
        plt.show()
    elif save:
        plt.close('all')
