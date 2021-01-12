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
from typing import *
import logging.config
import concurrent.futures
from arch_opt_exp.metrics import *
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

        self.metrics: Dict[str, Metric] = []
        self.metric_converged = None

    @classmethod
    def from_result(cls, result: Result) -> 'ExperimenterResult':
        obj = cls()
        for key in result.__dict__.keys():
            setattr(obj, key, getattr(result, key))
        return obj


class Experimenter:
    """Main class that handles the experiment for a given problem and algorithm."""

    results_folder: Optional[str] = None

    def __init__(self, problem: Problem, algorithm: Algorithm, n_eval_max: int, algorithm_name: str = None,
                 metrics: List[Metric] = None, log_level='INFO'):
        self.problem = problem
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name or algorithm.__class__.__name__
        self.n_eval_max = n_eval_max
        self.metrics = metrics

        self._res_folder = self.results_folder
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

    def _get_effectiveness_result_path(self, repeat_idx: int) -> str:
        return self._get_problem_algo_results_path('result_%d.pkl' % repeat_idx)

    ### EFFICIENCY EXPERIMENTATION ###

    def run_efficiency(self, metric_termination: MetricTermination, repeat_idx: int) -> ExperimenterResult:
        """
        Run the efficiency experiment: determine after how many steps an algorithm would terminate if a certain metric
        would have been used to detect convergence. Uses effectiveness results to "replay" an optimization session and
        returns results as if the passed metric would actually have been used.
        """
        self.capture_log(self._log_level)

        effectiveness_result = self.get_effectiveness_result(repeat_idx=repeat_idx)
        if effectiveness_result is None:
            raise RuntimeError('Effectiveness result not available: %d' % repeat_idx)

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
                result.metrics = metrics = {}
                for name, metric in effectiveness_result.metrics.items():
                    mod_metric = copy.deepcopy(metric)
                    mod_metric.values = {key: values[:n_steps] for key, values in mod_metric.values.items()}
                    metrics[name] = mod_metric
                break

        if result is None:  # Metric not converged
            result = copy.deepcopy(effectiveness_result)
            result.metric_converged = False

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

    def _get_efficiency_result_path(self, metric_termination: MetricTermination, repeat_idx: int) -> str:
        return self._get_problem_algo_results_path(
            '%s/result_%d.pkl' % (secure_filename(metric_termination.metric_name), repeat_idx))

    ### HELPER FUNCTIONS ###

    def _get_problem_algo_results_path(self, sub_path: str = None) -> str:
        problem_algo_path = '%s/%s' % (secure_filename(self.problem.name()), secure_filename(self.algorithm_name))
        if sub_path is not None:
            problem_algo_path += '/'+sub_path
        return self._get_results_path(sub_path)

    def _get_results_path(self, sub_path: str = None) -> str:
        if self._res_folder is None:
            raise ValueError('Must set results_folder on the class!')

        path = self._res_folder
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
