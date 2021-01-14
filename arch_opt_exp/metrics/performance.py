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
from arch_opt_exp.metrics_base import *
from pymoo.model.algorithm import Algorithm
from pymoo.performance_indicator.igd import IGD
from pymoo.performance_indicator.hv import Hypervolume
from pymoo.performance_indicator.igd_plus import IGDPlus
from pymoo.performance_indicator.distance_indicator import euclidean_distance

__all__ = ['SpreadMetric', 'DeltaHVMetric', 'IGDMetric', 'IGDPlusMetric', 'MaxConstraintViolationMetric']


class SpreadMetric(Metric):
    """
    Spread measures how well-spread a Pareto front is, representing the exploration performance of the algorithm. This
    metric only works for problems with 2 objectives. A value of 0 indicates a perfectly uniform spread.

    Implementation based on:
    Deb, K., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", 2002, 10.1109/4235.996017
    """

    @property
    def name(self) -> str:
        return 'spread'

    @property
    def value_names(self) -> List[str]:
        return ['delta']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if algorithm.problem.n_obj != 2:
            raise ValueError('Spread metric is only available for problems with 2 objectives!')

        # Get objective values of the current Pareto front (n_opt, n_obj), and sort along the first objective
        f = self._get_opt_f(algorithm)
        f = f[np.argsort(f[:, 0]), :]

        if f.shape[0] < 3:
            return [1.]

        dists = euclidean_distance(f[:-1, :], f[1:, :], norm=1)
        extreme_dists = dists[0]+dists[-1]  # d_f + d_i

        internal_dists = dists[1:-1]
        d_mean = np.mean(internal_dists)
        n_internal = len(internal_dists)

        # Equation (1), page 7 (188)
        delta = (extreme_dists + np.sum(np.abs(internal_dists - d_mean))) /\
                (extreme_dists + n_internal*d_mean)
        return [delta]


class DeltaHVMetric(Metric):
    """
    Metric measuring the difference to the pre-known hypervolume. It has a value between 1 and 0, where 0 means the
    hypervolume is exactly the same, meaning the true Pareto front has been found.

    Implementation based on:
    Palar, P.S., "On Multi-Objective Efficient Global Optimization Via Universal Kriging Surrogate Model", 2017,
        10.1109/CEC.2017.7969368
    """

    def __init__(self, pf: np.ndarray):
        super(DeltaHVMetric, self).__init__()

        self._hv = hv = Hypervolume(pf=pf, normalize=True)
        self.hv_true = hv.calc(pf)

    @property
    def name(self) -> str:
        return 'delta_hv'

    @property
    def value_names(self) -> List[str]:
        return ['delta_hv', 'hv', 'true_hv']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        # Calculate current hypervolume
        f = self._get_pop_f(algorithm)
        hv = self._hv.calc(f)

        # Calculate error metric
        delta_hv = (self.hv_true-hv)/self.hv_true

        return [delta_hv, hv, self.hv_true]


class IGDMetric(IndicatorMetric):
    """Inverse generational distance to the known pareto front."""

    def __init__(self, pf):
        super(IGDMetric, self).__init__(IGD(pf))


class IGDPlusMetric(IndicatorMetric):
    """Inverse generational distance (improved) to the known pareto front."""

    def __init__(self, pf):
        super(IGDPlusMetric, self).__init__(IGDPlus(pf))


class MaxConstraintViolationMetric(Metric):
    """Metric that simply returns the maximum constraint violation of the current optimal set. Note that depending on
    algorithm behavior, the current optimal set may contain only the least-violated point (e.g. for NSGA2)."""

    @property
    def name(self) -> str:
        return 'max_cv'

    @property
    def value_names(self) -> List[str]:
        return ['max_cv']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        cv = self._get_opt_cv(algorithm)
        if len(cv) == 0:
            return [0.]

        return [np.nanmax(cv)]
