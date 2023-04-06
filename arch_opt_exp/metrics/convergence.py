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
from pymoo.indicators.gd import GD
from pymoo.util.normalization import normalize
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.distance_indicator import DistanceIndicator

__all__ = ['HVMetric', 'DistanceIndicatorConvergenceMetric', 'GDConvergenceMetric', 'IGDConvergenceMetric',
           'CrowdingDistanceMetric', 'SteadyPerformanceIndicator', 'FitnessHomogeneityIndicator',
           'ConsolidationRatioMetric', 'MutualDominationRateMetric']


class HVMetric(Metric):
    """Hypervolume as a metric, without needing to know the location of the true Pareto front. Value rises until it
    stabilizes at an a-priori unknown value."""

    def __init__(self):
        super(HVMetric, self).__init__()

        self.nadir_point = None
        self.ideal_point = None

    @property
    def name(self) -> str:
        return 'hv'

    @property
    def value_names(self) -> List[str]:
        return ['hv']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f = self._get_pop_f(algorithm, valid_only=True)

        if self.nadir_point is None or self.ideal_point is None:
            self.nadir_point = np.max(f, axis=0)
            self.ideal_point = np.min(f, axis=0)

        hv_obj = Hypervolume(ref_point=np.ones(f.shape[1]))
        hv = hv_obj.do(normalize(f, xu=self.nadir_point, xl=self.ideal_point))

        return [hv]


class DistanceIndicatorConvergenceMetric(Metric):
    """Distance metric without the need for a Pareto front: calculates the distance metric between current and first
    population's Pareto front. Values rise (start at 0) until stabilization at an a-priori unknown point."""

    def __init__(self):
        super(DistanceIndicatorConvergenceMetric, self).__init__()

        self._pf = None

    @property
    def value_names(self) -> List[str]:
        return ['d']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if self._pf is None:
            self._pf = ref_pf = self.get_pareto_front(self._get_pop_f(algorithm))
        else:
            ref_pf = self._pf

        indicator = self._get_indicator(ref_pf)
        indicator.normalize = True

        current_pf = self.get_pareto_front(self._get_pop_f(algorithm))
        distance = indicator.do(current_pf)

        return [distance]

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _get_indicator(self, pf) -> DistanceIndicator:
        raise NotImplementedError


class GDConvergenceMetric(DistanceIndicatorConvergenceMetric):

    @property
    def name(self) -> str:
        return 'gd_conv'

    def _get_indicator(self, pf) -> DistanceIndicator:
        return GD(pf)


class IGDConvergenceMetric(DistanceIndicatorConvergenceMetric):

    @property
    def name(self) -> str:
        return 'igd_conv'

    def _get_indicator(self, pf) -> DistanceIndicator:
        return IGD(pf)


class CrowdingDistanceMetric(Metric):
    """
    Metric based on the crowding distance as used in the NSGA-II algorithm. Rudenko et al. observe that once the maximum
    crowding distance value in the current population stabilizes, the optimization run can be terminated.

    Implementation based on:
    Rudenko, O., "A Steady Performance Stopping Criterion for Pareto-based Evolutionary Algorithms", 2004
    """

    @property
    def name(self) -> str:
        return 'cd'

    @property
    def value_names(self) -> List[str]:
        return ['max', 'min', 'mean']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        # Check if the crowding distance has already been calculated
        pop = algorithm.pop
        if pop[0].get('crowding') is not None:
            crowding_distances = pop.get('crowding')
        else:
            crowding_distances = self._calculate_crowding_distances(pop)

        # Do not take infinites into account (cd = inf for points at the edge of each rank)
        crowding_distances[crowding_distances == np.inf] = np.nan
        crowding_distances = crowding_distances.astype(float)
        # crowding_distances[crowding_distances == None] = np.nan

        if np.all(np.isnan(crowding_distances)):
            return [np.nan, np.nan, np.nan]
        return [
            np.nanmax(crowding_distances),
            np.nanmin(crowding_distances),
            np.nanmean(crowding_distances),
        ]

    @staticmethod
    def _calculate_crowding_distances(pop) -> np.ndarray:
        """Based on pymoo.algorithms.moo.nsga2.RankAndCrowdingSurvival"""

        f = pop.get("F").astype(np.float, copy=False)
        fronts = NonDominatedSorting().do(f)

        for k, front in enumerate(fronts):
            crowding_of_front = calc_crowding_distance(f[front, :])

            for j, i in enumerate(front):
                pop[i].set('rank', k)
                pop[i].set('crowding', crowding_of_front[j])

        return pop.get('crowding')


class SteadyPerformanceIndicator(Metric):
    """
    Uses the stabilization of the standard deviation of the last n maximum crowding distances. This may reduce
    oscillation issues in detecting convergence.

    Implementation based on:
    Rudenko, O., "A Steady Performance Stopping Criterion for Pareto-based Evolutionary Algorithms", 2004
    """

    def __init__(self, n_last_steps: int = 10):
        super(SteadyPerformanceIndicator, self).__init__()

        self.n = n_last_steps
        self.cd_metric = CrowdingDistanceMetric()

    @property
    def name(self) -> str:
        return 'steady_performance'

    @property
    def value_names(self) -> List[str]:
        return ['std', 'max_cd']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        self.cd_metric.calculate_step(algorithm)
        max_cd_values = self.cd_metric.values['max']
        last_max_cd = max_cd_values[-1]

        if len(max_cd_values) < self.n:
            return [np.nan, last_max_cd]

        max_cd_std = np.std(max_cd_values[-self.n:])
        return [max_cd_std, last_max_cd]


class FitnessHomogeneityIndicator(Metric):
    """
    The Fitness Homogeneity Indicator (FHI) measures the standard deviation of the objectives. Once this stabilizes,
    convergence may be assumed.

    Implementation based on:
    Marti, L., "A Progress Indicator for Detecting Success and Failure in Evolutionary Multi-Objective Optimization",
        2010, 10.1109/CEC.2010.5586352
    """

    @property
    def name(self) -> str:
        return 'fhi'

    @property
    def value_names(self) -> List[str]:
        return ['fhi']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f = self._get_pop_f(algorithm, feasible_only=True)

        # Normalize objectives
        f_min, f_max = np.min(f, axis=0), np.max(f, axis=0)
        f_norm = (f-f_min)/(f_max-f_min)

        # Calculate standard deviation
        f_std = np.std(f_norm, axis=0)
        f_std_min = np.min(f_std)

        return [f_std_min]


class ConsolidationRatioMetric(Metric):
    """
    The consolidation ratio (CR) measures the ratio of the number of non-dominated solutions n steps ago that are also
    present in the current non-dominated set. CR has a value between 0 and 1, where 1 indicates full consolidation and
    therefore convergence.

    Implementation based on:
    Goel, T., "A Non-Dominance-Based Online Stopping Criterion for Multi-Objective Evolutionary Algorithms", 2010,
        10.1002/nme.2909
    """

    def __init__(self, n_delta: int = 1):
        super(ConsolidationRatioMetric, self).__init__()

        self.n_delta = n_delta
        self.nd_sets = []

    @property
    def name(self) -> str:
        return 'cr'

    @property
    def value_names(self) -> List[str]:
        return ['cr', 'n_nd', 'n_nd_prev', 'n_overlap']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        # Update current non-dominated set
        nd = self.get_pareto_front(self._get_pop_f(algorithm))
        n_nd = nd.shape[0]

        # Get non-dominated set to compare against
        if len(self.nd_sets) < self.n_delta:
            self.nd_sets.append(nd)
            return [0, n_nd, 0, 0]
        nd_prev = self.nd_sets[-self.n_delta]
        n_nd_prev = nd_prev.shape[0]

        # Calculate consolidation ratio
        _, nd_prev_in_nd_combined, _ = get_pareto_front_overlap(nd_prev, nd)
        n_overlap = nd_prev_in_nd_combined.shape[0]
        consolidation_ratio = n_overlap/n_nd

        self.nd_sets.append(nd)
        return [consolidation_ratio, n_nd, n_nd_prev, n_overlap]


def get_pareto_front_overlap(pf1: np.ndarray, pf2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines the overlap between two Pareto fronts. Returns:
    - The combined Pareto set
    - Points of the first Pareto set (pf1) in the combined Pareto set
    - Points of the second Pareto set (pf2) in the combined Pareto set"""

    # Combine the Pareto sets and get the new Pareto set
    pf_combined = Metric.get_pareto_front(np.concatenate([pf1, pf2], axis=0))

    pf1_in_pf_combined_mask = np.zeros(pf1.shape, dtype=bool)
    pf2_in_pf_combined_mask = np.zeros(pf2.shape, dtype=bool)
    for i_obj in range(pf_combined.shape[1]):
        _, i_intersect, _ = np.intersect1d(pf1[:, i_obj], pf_combined[:, i_obj], return_indices=True)
        pf1_in_pf_combined_mask[i_intersect, i_obj] = True

        _, i_intersect, _ = np.intersect1d(pf2[:, i_obj], pf_combined[:, i_obj], return_indices=True)
        pf2_in_pf_combined_mask[i_intersect, i_obj] = True

    pf1_in_pf_combined_mask = np.all(pf1_in_pf_combined_mask, axis=1)
    pf2_in_pf_combined_mask = np.all(pf2_in_pf_combined_mask, axis=1)

    pf1_in_pf_combined = np.copy(pf1[pf1_in_pf_combined_mask, :])
    pf2_in_pf_combined = np.copy(pf2[pf2_in_pf_combined_mask, :])

    return pf_combined, pf1_in_pf_combined, pf2_in_pf_combined


class MutualDominationRateMetric(Metric):
    """
    The Mutual Domination Rate (MDR) measures how much the new Pareto set is dominating the previous Pareto set. It
    takes a value between -1 and 1, where:
    - 1: much improvement (new Pareto set 100% dominates previous Pareto set)
    - 0: no substantial improvement (new Pareto set dominates previous Pareto set 50% and vice-versa)
    - -1: no improvement (previous Pareto set 100% dominates new Pareto set)

    Convergence is assumed when MDR reaches 0.

    Implementation based on:
    Martí, L., "A Stopping Criterion for Multi-Objective Evolutionary Algorithms", 2016, 10.1016/j.ins.2016.07.025
    """

    def __init__(self):
        super(MutualDominationRateMetric, self).__init__()
        self.pf_prev = None

    @property
    def name(self) -> str:
        return 'mdr'

    @property
    def value_names(self) -> List[str]:
        return ['mdr', 'n_par', 'n_par_dom', 'n_par_prev', 'n_par_prev_dom', 'n_par_comb']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        # Get current Pareto set
        pf = self.get_pareto_front(self._get_pop_f(algorithm))
        n_pf = pf.shape[0]

        # Get previous Pareto set
        if self.pf_prev is None:
            self.pf_prev = pf
            return [1, n_pf, n_pf, 0, 0, n_pf]

        pf_prev = self.pf_prev
        n_pf_prev = pf_prev.shape[0]

        # Get mutual domination
        pf_combined, pf_in_combined, pf_prev_in_combined = get_pareto_front_overlap(pf, pf_prev)
        n_pf_combined = pf_combined.shape[0]
        n_pf_dom = pf_in_combined.shape[0]
        n_pf_prev_dom = pf_prev_in_combined.shape[0]

        # Calculate mutual domination rate
        mdr = (n_pf_dom/n_pf) - (n_pf_prev_dom/n_pf_prev)

        self.pf_prev = pf
        return [mdr, n_pf, n_pf_dom, n_pf_prev, n_pf_prev_dom, n_pf_combined]
