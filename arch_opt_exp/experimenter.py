"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2020, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""

import os
import copy
import pickle
import logging
import tempfile
import contextlib
import numpy as np
from typing import *
import logging.config
import concurrent.futures
from arch_opt_exp.metrics_base import *
from werkzeug.utils import secure_filename

from pymoo.optimize import minimize
from pymoo.model.result import Result
from pymoo.model.problem import Problem
from pymoo.model.algorithm import Algorithm
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination

__all__ = ['Experimenter', 'EffectivenessTerminator', 'ExperimenterResult']

log = logging.getLogger('arch_opt_exp.exp')


class EffectivenessTerminator(MaximumFunctionCallTermination):
    """Terminator that terminates after a maximum number of function evaluations, and also calculates indicator values
    at every algorithm step."""

    def __init__(self, n_eval_max: int, metrics: List[Metric] = None):
        super(EffectivenessTerminator, self).__init__(n_max_evals=n_eval_max)

        self.metrics = metrics or []

    def _do_continue(self, algorithm: Algorithm):
        for metric in self.metrics:
            metric.calculate_step(algorithm)

        return super(EffectivenessTerminator, self)._do_continue(algorithm)


class ExperimenterResult(Result):
    """Result class with some extra data compared to the pymoo Result class."""

    def __init__(self):
        super(ExperimenterResult, self).__init__()

        self.metrics: Dict[str, Metric] = {}
        self.metric_converged = None
        self.termination: Optional[MetricTermination] = None

        self.n_steps = None
        self.n_steps_std = None

        self.n_eval = None
        self.n_eval_std = None

        self.exec_time_std = None

    @classmethod
    def from_result(cls, result: Result) -> 'ExperimenterResult':
        """Create from a Result class as output by an Algorithm."""
        exp_result = cls()
        for key in result.__dict__.keys():
            setattr(exp_result, key, getattr(result, key))

        exp_result.n_steps = len(exp_result.history) if exp_result.history is not None else None

        exp_result.n_eval = [algo.evaluator.n_eval for algo in exp_result.history]

        return exp_result

    @classmethod
    def aggregate_results(cls, results: List['ExperimenterResult']) -> 'ExperimenterResult':
        """Aggregate results from multiple ExperimenterResult instances, replacing metrics values with the mean and
        adding standard deviations."""
        result = cls()

        result.exec_time, result.exec_time_std = cls._get_mean_std(results, lambda r: r.exec_time)
        result.n_steps, result.n_steps_std = cls._get_mean_std(results, lambda r: r.n_steps)
        result.n_eval, result.n_eval_std = cls._get_mean_std(results, lambda r: r.n_eval)

        for name, metric in results[0].metrics.items():
            result.metrics[name] = metric = copy.deepcopy(metric)

            metric.values, metric.values_std = {}, {}
            for key in metric.value_names:
                metric.values[key], metric.values_std[key] = \
                    cls._get_mean_std(results, lambda r: r.metrics[name].values[key])

        return result

    @staticmethod
    def _get_mean_std(results: List['ExperimenterResult'],
                      getter: Callable[['ExperimenterResult'], Optional[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and standard deviation for several repeated experimenter results."""

        results_data = None
        for result in results:
            res_data = getter(result)
            if res_data is None:
                continue
            res_data = np.atleast_3d(res_data)
            if results_data is None:
                results_data = res_data
            else:

                # Make sure results shapes are the same
                # Align data points at the end, so that mean values can be compared for steps from end
                # (e.g. upon termination)
                if results_data.shape[:2] != res_data.shape[:2]:
                    rs, r = results_data.shape, res_data.shape
                    new_shape = (max(rs[0], r[0]), max(rs[1], r[1]))

                    results_data_ = np.zeros(new_shape+(rs[2],))*np.nan
                    results_data_[-rs[0]:, -rs[1]:, :] = results_data
                    results_data = results_data_

                    res_data_ = np.zeros(new_shape+(1,))*np.nan
                    res_data_[-r[0]:, -r[1]:, :] = res_data
                    res_data = res_data_

                results_data = np.concatenate([results_data, res_data], axis=2)

        if results_data is None:
            return None, None

        mean_data = np.nanmean(results_data, axis=2)
        if mean_data.shape[0] == 1:
            mean_data = mean_data[0, :]
        if len(mean_data) == 1:
            mean_data = mean_data[0]

        std_data = np.nanstd(results_data, axis=2)
        if std_data.shape[0] == 1:
            std_data = std_data[0, :]
        if len(std_data) == 1:
            std_data = std_data[0]

        return mean_data, std_data

    @staticmethod
    def plot_compare_metrics(results: List['ExperimenterResult'], metric_name: str, plot_evaluations=False, **kwargs):
        metrics = [res.metrics[metric_name] for res in results]
        n_eval = [res.n_eval for res in results] if plot_evaluations else None
        Metric.plot_multiple(metrics, n_eval=n_eval, **kwargs)


class Experimenter:
    """Main class that handles the experiment for a given problem and algorithm."""

    results_folder: Optional[str] = None

    def __init__(self, problem: Problem, algorithm: Algorithm, n_eval_max: int, algorithm_name: str = None,
                 metrics: List[Metric] = None, log_level='INFO', results_folder: str = None):
        self.problem = problem
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name or algorithm.__class__.__name__
        self.n_eval_max = n_eval_max
        self.metrics = metrics

        self.results_folder = results_folder or self.results_folder  # Turn class attr into instance attr
        self._log_level = log_level

    ### EFFECTIVENESS EXPERIMENTATION ###

    def run_effectiveness_parallel(self, n_repeat: int) -> List[ExperimenterResult]:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.run_effectiveness, i) for i in range(n_repeat)]
            concurrent.futures.wait(futures)

            return [fut.result() for fut in futures]

    def run_effectiveness(self, repeat_idx: int = 0, seed=None) -> ExperimenterResult:
        """
        Run the effectiveness experiment: find out how well the algorithm is able to approach the Pareto front. Simply
        runs the algorithm with a predefines maximum number of function evaluations.
        """
        self.capture_log(level=self._log_level)

        termination = EffectivenessTerminator(n_eval_max=self.n_eval_max, metrics=self.metrics)

        # Run the algorithm
        log.info('Running effectiveness experiment: %s / %s / %d' %
                 (self.problem.name(), self.algorithm_name, repeat_idx))
        result = minimize(
            self.problem, self.algorithm,
            termination=termination,
            copy_algorithm=True, copy_termination=True,
            seed=seed,
            save_history=True,
        )

        # Prepare experimenter results by including metrics
        result = ExperimenterResult.from_result(result)
        metrics: List[Metric] = result.algorithm.termination.metrics
        result.metrics = {met.name: met for met in metrics}

        # Store results and return
        result_path = self._get_effectiveness_result_path(repeat_idx=repeat_idx)
        with open(result_path, 'wb') as fp:
            pickle.dump(result, fp)

        log.info('Effectiveness experiment finished: %s / %s / %d' %
                 (self.problem.name(), self.algorithm_name, repeat_idx))
        return result

    def get_effectiveness_result(self, repeat_idx: int) -> Optional[ExperimenterResult]:
        result_path = self._get_effectiveness_result_path(repeat_idx=repeat_idx)
        if not os.path.exists(result_path):
            return
        with open(result_path, 'rb') as fp:
            return pickle.load(fp)

    def get_aggregate_effectiveness_results(self) -> ExperimenterResult:
        """Returns results aggregated for all individual runs, using mean and std."""
        results = []
        i = 0
        while True:
            result = self.get_effectiveness_result(repeat_idx=i)
            if result is None:
                break

            results.append(result)
            i += 1

        return ExperimenterResult.aggregate_results(results)

    def _get_effectiveness_result_path(self, repeat_idx: int) -> str:
        return self._get_problem_algo_results_path('result_%d.pkl' % repeat_idx)

    ### EFFICIENCY EXPERIMENTATION ###

    def run_efficiency_repeated(self, metric_termination: MetricTermination) -> List[ExperimenterResult]:
        """Run efficiency experiments for the amount of previously generated effectiveness results available."""
        results = []

        i = 0
        while True:
            result = self.run_efficiency(metric_termination, repeat_idx=i)
            if result is None:
                break

            results.append(result)
            i += 1

        return results

    def run_efficiency(self, metric_termination: MetricTermination, repeat_idx: int) -> Optional[ExperimenterResult]:
        """
        Run the efficiency experiment: determine after how many steps an algorithm would terminate if a certain metric
        would have been used to detect convergence. Uses effectiveness results to "replay" an optimization session and
        returns results as if the passed metric would actually have been used.
        """
        self.capture_log(self._log_level)

        effectiveness_result = self.get_effectiveness_result(repeat_idx=repeat_idx)
        if effectiveness_result is None:
            return

        termination = copy.deepcopy(metric_termination)

        # Simulate algorithm execution using provided termination metric
        log.info('Running efficiency experiment: %s / %s / %s / %d' %
                 (self.problem.name(), self.algorithm_name, metric_termination.metric_name, repeat_idx))
        history = []
        result = None
        algorithm: Algorithm
        for algorithm in effectiveness_result.history:
            history.append(algorithm)

            if not termination.do_continue(algorithm):  # Metric convergence
                algorithm.history = history
                n_steps = len(history)

                result = algorithm.result()
                algorithm.termination = termination
                result.algorithm = algorithm

                # Modify metrics to reflect number of steps
                result = ExperimenterResult.from_result(result)
                result.metric_converged = True
                result.termination = termination

                result.metrics = metrics = {}
                for name, metric in effectiveness_result.metrics.items():
                    mod_metric = copy.deepcopy(metric)
                    mod_metric.values = {key: values[:n_steps] for key, values in mod_metric.values.items()}
                    metrics[name] = mod_metric
                break

        if result is None:  # Metric not converged
            result = copy.deepcopy(effectiveness_result)
            result.metric_converged = False
            result.termination = termination

        # Store results and return
        result_path = self._get_efficiency_result_path(metric_termination, repeat_idx=repeat_idx)
        with open(result_path, 'wb') as fp:
            pickle.dump(result, fp)

        log.info('Efficiency experiment finished (converged: %r): %s / %s / %d' %
                 (result.metric_converged, self.problem.name(), self.algorithm_name, repeat_idx))
        return result

    def get_efficiency_result(self, metric_termination: MetricTermination, repeat_idx: int) \
            -> Optional[ExperimenterResult]:
        result_path = self._get_efficiency_result_path(metric_termination, repeat_idx=repeat_idx)
        if not os.path.exists(result_path):
            return
        with open(result_path, 'rb') as fp:
            return pickle.load(fp)

    def get_aggregate_efficiency_results(self, metric_termination: MetricTermination) -> ExperimenterResult:
        """Get efficiency results aggregated for all efficiency experiment runs."""
        results = []
        i = 0
        while True:
            result = self.get_efficiency_result(metric_termination, repeat_idx=i)
            if result is None:
                break

            results.append(result)
            i += 1

        return ExperimenterResult.aggregate_results(results)

    def _get_efficiency_result_path(self, metric_termination: MetricTermination, repeat_idx: int) -> str:
        return self._get_problem_algo_results_path(
            '%s/result_%d.pkl' % (secure_filename(metric_termination.metric_name), repeat_idx))

    ### HELPER FUNCTIONS ###

    @staticmethod
    @contextlib.contextmanager
    def temp_results():
        """
        Sets a temporary folder as results folder. Useful for running experiments and directly analyzing the
        results. Usage:

        with Experimenter.temp_results():
            ...
        """

        orig_res_folder = Experimenter.results_folder

        with tempfile.TemporaryDirectory() as tmp_dir:
            Experimenter.results_folder = tmp_dir
            yield

        Experimenter.results_folder = orig_res_folder

    def _get_problem_algo_results_path(self, sub_path: str = None) -> str:
        problem_algo_path = '%s/%s' % (secure_filename(self.problem.name()), secure_filename(self.algorithm_name))
        if sub_path is not None:
            problem_algo_path += '/'+sub_path
        return self._get_results_path(sub_path)

    def _get_results_path(self, sub_path: str = None) -> str:
        if self.results_folder is None:
            raise ValueError('Must set results_folder on the class!')

        path = self.results_folder
        if sub_path is not None:
            path = os.path.join(path, sub_path)

        os.makedirs(os.path.dirname(path) if sub_path is not None else path, exist_ok=True)
        return path

    @staticmethod
    def capture_log(level='INFO'):
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True,
            'formatters': {
                'console': {
                    'format': '%(levelname)- 8s %(asctime)s %(name)- 18s: %(message)s'
                },
            },
            'handlers': {
                'console': {
                    'level': level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'console',
                },
            },
            'loggers': {
                'arch_opt_exp': {
                    'handlers': ['console'],
                    'level': level,
                },
            },
        })
