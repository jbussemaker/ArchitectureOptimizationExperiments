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

import numpy as np
from typing import *
from arch_opt_exp.experiments.metrics_base import *
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.indicators.distance_indicator import euclidean_distance

__all__ = ['SpreadMetric', 'DeltaHVMetric', 'IGDMetric', 'IGDPlusMetric', 'MaxConstraintViolationMetric',
           'NrEvaluationsMetric', 'BestObjMetric']


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
            return [np.nan]

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

    def __init__(self, pf: np.ndarray, perc_pass: List[float] = None):
        self.is_one_dim = pf.shape[0] == 1
        self.max_f = np.max(pf, axis=0)
        self.pf_0 = pf[0, :]
        self._hv = hv = Hypervolume(pf=pf, normalize=True)
        self.hv_true = hv.do(pf)
        self.delta_hv0 = None

        self.perc_pass = perc_pass if perc_pass is not None else [.1, .2, .5]
        self.i_iter = 0
        self.is_passed = None

        super(DeltaHVMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'delta_hv'

    @property
    def value_names(self) -> List[str]:
        pp_names = [f'pass_{pp*100:.0f}' for pp in self.perc_pass]
        return ['delta_hv', 'hv', 'true_hv', 'ratio']+pp_names

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f_opt = self._get_opt_f(algorithm, feasible_only=True)
        f_all = self._get_pop_f(algorithm, valid_only=True)
        return self.calculate_delta_hv(f_opt, f_all)

    def calculate_delta_hv(self, f_opt: np.ndarray, f_all: np.ndarray) -> List[float]:
        def _get_iter_p_passed(ratio):
            if self.is_passed is None:
                self.is_passed = [None]*len(self.perc_pass)
            for i_pass, perc in enumerate(self.perc_pass):
                if self.is_passed[i_pass] is None and ratio <= perc:
                    self.is_passed[i_pass] = self.i_iter

            self.i_iter += 1
            return [passed if passed is not None else np.nan for passed in self.is_passed]

        # If we have only one point, calculate the relative distance to the optimal point instead (because true HV is 0)
        if self.is_one_dim:
            # Update max points
            if len(f_all) == 0:
                if self.max_f is not None:
                    return [np.nan]*len(self.value_names)
                max_f = self.max_f
            else:
                self.max_f = max_f = np.max(np.row_stack([f_all, [self.max_f]]), axis=0)

            # Update maximum distance to the optimal point (this represents the extend of the design space)
            true_dist = max_f-self.pf_0
            true_dist[true_dist == 0] = 1
            true_dist_m = np.sqrt(np.sum(true_dist**2))

            # Get the relative distance of the current best point to the optimal point
            f_rel_dist = (f_opt-self.pf_0)/true_dist
            f_rel_min_dist = np.min(np.sqrt(np.sum(f_rel_dist**2, axis=1)))

            if self.delta_hv0 is None:
                self.delta_hv0 = f_rel_min_dist
            f_ratio = f_rel_min_dist/self.delta_hv0

            res = [f_rel_min_dist, f_rel_min_dist*true_dist_m, true_dist_m, f_ratio]+_get_iter_p_passed(f_ratio)
            return res

        # Calculate current hypervolume
        hv = self._hv.do(f_opt)

        # Calculate error metric
        delta_hv = (self.hv_true-hv)/self.hv_true

        if self.delta_hv0 is None:
            self.delta_hv0 = delta_hv
        delta_hv_ratio = delta_hv/self.delta_hv0

        return [delta_hv, hv, self.hv_true, delta_hv_ratio]+_get_iter_p_passed(delta_hv_ratio)


class IGDMetric(IndicatorMetric):
    """Inverse generational distance to the known pareto front."""

    def __init__(self, pf):
        super(IGDMetric, self).__init__(IGD(pf, normalize=True))


class IGDPlusMetric(IndicatorMetric):
    """Inverse generational distance (improved) to the known pareto front."""

    def __init__(self, pf):
        super(IGDPlusMetric, self).__init__(IGDPlus(pf, normalize=True))


class MaxConstraintViolationMetric(Metric):
    """Metric that simply returns the maximum constraint violation of the current population."""

    def __init__(self):
        super(MaxConstraintViolationMetric, self).__init__()

        self._total_pop = None
        self._el_dup = DefaultDuplicateElimination()

    @property
    def name(self) -> str:
        return 'max_cv'

    @property
    def value_names(self) -> List[str]:
        return ['max_cv', 'min_cv', 'pop_max_cv', 'pop_min_cv', 'frac_nan']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if self._total_pop is None:
            self._total_pop = self._get_pop(algorithm)
        else:
            pop = Population.merge(self._total_pop, self._get_pop(algorithm))
            self._total_pop = self._el_dup.do(pop)

        cv = self._get_opt_cv(algorithm)
        if len(cv) == 0:
            return [0., 0., 0., 0., 0.]
        cv[np.isinf(cv)] = np.nan

        cv_pop = self._get_pop_cv(algorithm)
        cv_pop[np.isinf(cv_pop)] = np.nan

        cv_total_pop = self._total_pop.get('CV')
        cv_total_pop[np.isinf(cv_total_pop)] = np.nan
        frac_nan = np.sum(np.isnan(cv_total_pop))/len(cv_total_pop)

        return [np.nanmax(cv), np.nanmin(cv), np.nanmax(cv_pop), np.nanmin(cv_pop), frac_nan]


class NrEvaluationsMetric(Metric):
    """Metric that tracks the number of function evaluations after each algorithm step."""

    @property
    def name(self) -> str:
        return 'n_eval'

    @property
    def value_names(self) -> List[str]:
        return ['n_eval']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        return [algorithm.evaluator.n_eval]


class BestObjMetric(Metric):
    """Metric that tracks the current best (feasible) objective values."""

    def __init__(self, i_f=0):
        self.i_f = i_f
        super(BestObjMetric, self).__init__()

    @property
    def name(self):
        return 'f_best'

    @property
    def value_names(self) -> List[str]:
        return ['f_best']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if algorithm.opt is not None:
            return [algorithm.opt.get('F')[self.i_f, 0]]
        return [np.nan]
