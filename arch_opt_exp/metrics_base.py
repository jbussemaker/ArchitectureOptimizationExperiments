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

import numpy as np
from typing import *
import matplotlib.pyplot as plt

from pymoo.model.algorithm import Algorithm
from pymoo.model.indicator import Indicator
from pymoo.model.termination import Termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = ['Metric', 'IndicatorMetric', 'MetricTermination']


class Metric:
    """More general metric class, that can output multiple values to track. Should be serializable."""

    def __init__(self):
        self.values = {name: [] for name in self.value_names}
        self.values_std = None  # Used in ExperimenterResults

    def calculate_step(self, algorithm: Algorithm):
        values = self._calculate_values(algorithm)
        names = self.value_names
        if len(values) != len(names):
            raise ValueError('Values should have the same length as the number of values')

        for i, name in enumerate(names):
            self.values[name].append(values[i])

    def results(self) -> Dict[str, np.ndarray]:
        return {key: np.array(value) for key, value in self.values.items()}

    def plot(self, std_sigma=1., show=True):
        for value_name in self.value_names:
            plt.figure()

            y = self.values[value_name]
            x = list(range(len(y)))
            y_err = np.array(self.values_std[value_name]) if self.values_std is not None else None

            kwargs = {'linewidth': 1}
            plt.plot(x, y, '-k', **kwargs)

            err_title = ''
            if y_err is not None:
                err_title = ' (std $\\sigma$ = %.2f)' % std_sigma
                plt.errorbar(x, y+y_err*std_sigma, fmt='--k', **kwargs)
                plt.errorbar(x, y-y_err*std_sigma, fmt='--k', **kwargs)

            plt.title('Metric: %s.%s%s' % (self.name, value_name, err_title))
            plt.xlim([0, x[-1]])
            plt.xlabel('Iteration')
            plt.ylabel(value_name)

        if show:
            plt.show()

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def value_names(self) -> List[str]:
        raise NotImplementedError

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError

    @classmethod
    def _get_pop_x(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Design vectors of the population: (n_pop, n_x)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('X')

    @classmethod
    def _get_pop_f(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Objective values of the population: (n_pop, n_f)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('F')

    @classmethod
    def _get_pop_g(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint values of the population: (n_pop, n_g)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('G')

    @classmethod
    def _get_pop_cv(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint violation values of the population: (n_pop, n_g)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('CV')

    @staticmethod
    def _get_pop(algorithm: Algorithm, feasible_only=False):
        pop = algorithm.pop
        if feasible_only:
            i_feasible = np.where(pop.get('feasible'))[0]
            return pop[i_feasible]
        return pop

    @classmethod
    def _get_opt_x(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Design vectors of the optimum population (non-dominated current Pareto front): (n_opt, n_x)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('X')

    @classmethod
    def _get_opt_f(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Objective values of the optimum population: (n_opt, n_f)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('F')

    @classmethod
    def _get_opt_g(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint values of the optimum population: (n_opt, n_g)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('G')

    @classmethod
    def _get_opt_cv(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint violation values of the optimum population: (n_opt, n_g)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('CV')

    @staticmethod
    def _get_opt(algorithm: Algorithm, feasible_only=False):
        opt = algorithm.opt
        if feasible_only:
            i_feasible = np.where(opt.get('feasible'))[0]
            return opt[i_feasible]
        return opt

    @staticmethod
    def get_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        i_non_dom = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return np.copy(f[i_non_dom, :])


class IndicatorMetric(Metric):
    """Metric based on a performance indicator. Performance indicators only depend on the objective values."""

    def __init__(self, indicator: Indicator):
        super(IndicatorMetric, self).__init__()
        self.indicator = indicator

    @property
    def name(self) -> str:
        return self.indicator.__class__.__name__

    @property
    def value_names(self) -> List[str]:
        return ['indicator']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        return [self.indicator.calc(self._get_opt_f(algorithm))]


class MetricTermination(Termination):
    """Termination based on a metric."""

    def __init__(self, metric: Metric, value_name: str = None, lower_limit: float = None, upper_limit: float = None):
        if lower_limit is None and upper_limit is None:
            raise ValueError('Provide at least either a lower or an upper limit!')
        self.metric = metric
        self.value_name = value_name or metric.value_names[0]
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        super(MetricTermination, self).__init__()

    @property
    def metric_name(self) -> str:
        return self.metric.name

    def _do_continue(self, algorithm: Algorithm, **kwargs):

        self.metric.calculate_step(algorithm)
        value = self.metric.values[self.value_name][-1]

        if self.lower_limit is not None and value <= self.lower_limit:
            return False
        if self.upper_limit is not None and value >= self.upper_limit:
            return False
        return True
