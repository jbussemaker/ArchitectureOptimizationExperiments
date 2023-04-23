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
import numpy as np
from typing import *
from scipy.stats import norm
from scipy.special import ndtr

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.core.algorithm import filter_optimum
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance

from sb_arch_opt.algo.arch_sbo.infill import *

__all__ = ['ProbabilityOfImprovementInfill', 'LowerConfidenceBoundInfill', 'MinimumPoIInfill', 'EnsembleInfill',
           'IgnoreConstraints', 'FunctionEstimateConstrainedInfill', 'ExpectedImprovementInfill']


class ProbabilityOfImprovementInfill(ConstrainedInfill):
    """
    Probability of Improvement represents the probability that some point will be better than the current best estimate
    with some offset:

    PoI(x) = Phi((T - y(x))/sqrt(s(x)))
    where
    - Phi is the cumulative distribution function of the normal distribution
    - T is the improvement target (current best estimate minus some offset)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate

    PoI was developed for single-objective optimization, and because of the use of the minimum current objective value,
    it tends towards suggesting improvement points only at the edges of the Pareto front. It has been modified to
    evaluate the PoI with respect to the closest Pareto front point instead.

    Implementation based on:
    Hawe, G.I., "An Enhanced Probability of Improvement Utility Function for Locating Pareto Optimal Solutions", 2007
    """

    def __init__(self, f_min_offset: float = 0., **kwargs):
        self.f_min_offset = f_min_offset
        super().__init__(**kwargs)

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self._evaluate_f_poi(f_predict, f_var_predict, self.y_train[:, :f_predict.shape[1]], self.f_min_offset)

    @classmethod
    def _evaluate_f_poi(cls, f: np.ndarray, f_var: np.ndarray, f_current: np.ndarray, f_min_offset=0.) -> np.ndarray:
        # Normalize current and predicted objectives
        f_pareto = cls.get_pareto_front(f_current)
        nadir_point, ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        nadir_point[nadir_point == ideal_point] = 1.
        f_pareto_norm = (f_pareto-ideal_point)/(nadir_point-ideal_point)
        f_norm, f_var_norm = cls._normalize_f_var(f, f_var, nadir_point, ideal_point)

        # Get PoI for each point using closest point in the Pareto front
        f_poi = np.empty(f.shape)
        for i in range(f.shape[0]):
            i_par_closest = np.argmin(np.sum((f_pareto_norm-f_norm[i, :])**2, axis=1))
            f_par_targets = f_pareto_norm[i_par_closest, :]-f_min_offset
            poi = cls._poi(f_par_targets, f_norm[i, :], f_var_norm[i, :])
            f_poi[i, :] = 1.-poi

        return f_poi

    @staticmethod
    def _normalize_f_var(f: np.ndarray, f_var: np.ndarray, nadir_point, ideal_point):
        f_norm = (f-ideal_point)/(nadir_point-ideal_point)
        f_var_norm = f_var/((nadir_point-ideal_point+1e-30)**2)
        return f_norm, f_var_norm

    @staticmethod
    def _poi(f_targets: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        return norm.cdf((f_targets-f) / np.sqrt(f_var))


class LowerConfidenceBoundInfill(ConstrainedInfill):
    """
    The Lower Confidence Bound (LCB) represents the lowest expected value to be found at some point given its standard
    deviation.

    LCB(x) = y(x) - alpha * sqrt(s(x))
    where
    - y(x) the surrogate model estimate
    - alpha is a scaling parameter (typical value is 2) --> lower means more exploitation, higher more exploration
    - s(x) the surrogate model variance estimate

    Implementation based on:
    Cox, D., "A Statistical Method for Global Optimization", 1992, 10.1109/icsmc.1992.271617
    """

    def __init__(self, alpha: float = 2., **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        lcb = f_predict - self.alpha*np.sqrt(f_var_predict)
        return lcb


class MinimumPoIInfill(ConstrainedInfill):
    """
    The Minimum Probability of Improvement (MPoI) criterion is a multi-objective infill criterion and modifies the
    calculation of the domination probability by only considering one objective dimension at a time. This should reduce
    computational cost.

    Optionally multiplies the MPoI criteria by its first integral moment, to transform it to an EI-like metric. Uses a
    similar implementation as `EuclideanEIInfill`.

    Implementation based on:
    Rahat, A.A.M., "Alternative Infill Strategies for Expensive Multi-Objective Optimisation", 2017,
        10.1145/3071178.3071276
    Parr, J.M., "Improvement Criteria for Constraint Handling and Multiobjective Optimization", 2013
    """

    def __init__(self, euclidean=False, **kwargs):
        self.euclidean = euclidean
        self.f_pareto = None
        super().__init__(**kwargs)

    def get_n_infill_objectives(self) -> int:
        return 1

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        super().set_samples(x_train, y_train)
        self.f_pareto = self.get_pareto_front(y_train[:, :self.problem.n_obj])

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self.get_mpoi_f(f_predict, f_var_predict, self.f_pareto, self.euclidean)

    @classmethod
    def get_mpoi_f(cls, f_predict: np.ndarray, f_var_predict: np.ndarray, f_pareto: np.ndarray, euclidean: bool) \
            -> np.ndarray:

        mpoi = np.empty((f_predict.shape[0], 1))
        for i in range(f_predict.shape[0]):
            mpoi[i, 0] = cls._mpoi(f_pareto, f_predict[i, :], f_var_predict[i, :], euclidean=euclidean)

        mpoi[mpoi < 1e-6] = 0.
        return 1.-mpoi

    @classmethod
    def _mpoi(cls, f_pareto: np.ndarray, f_predict: np.ndarray, var_predict: np.ndarray, euclidean: bool) -> float:

        n, n_f = f_pareto.shape

        # Probability of being dominated for each point in the Pareto front along each objective dimension
        def cdf_not_better(f, f_pred, var_pred):  # Rahat 2017, Eq. 11, 12
            return ndtr((f_pred-f)/np.sqrt(var_pred))

        p_is_dom_dim = np.empty((n, n_f))
        for i_f in range(n_f):
            p_is_dom_dim[:, i_f] = cdf_not_better(f_pareto[:, i_f], f_predict[i_f], var_predict[i_f])

        # Probability of being dominated for each point along all dimensions: Rahat 2017, Eq. 10
        p_is_dom = np.prod(p_is_dom_dim, axis=1)

        # Probability of domination for each point: Rahat 2017, Eq. 13
        p_dom = 1-p_is_dom

        # Minimum probability of domination: Rahat 2017, Eq. 14
        min_poi = np.min(p_dom)

        # Multiply by distance to Pareto front if requested
        if euclidean:
            min_poi *= cls._get_euclidean_moment(min_poi, f_pareto, f_predict)

        return min_poi

    @classmethod
    def _get_euclidean_moment(cls, p_dominate: float, f_pareto: np.ndarray, f_predict: np.ndarray) -> float:

        # If the probability of domination is less than 50%, it means we are on the wrong side of the Pareto front
        if p_dominate < .5:
            return 0.

        return np.min(np.sqrt(np.sum((f_predict-f_pareto) ** 2, axis=1)))  # Parr Eq. 6.9


class EnsembleInfill(ConstrainedInfill):
    """
    Infill strategy that optimize multiple underlying infill criteria simultaneously, thereby getting the best
    compromise between what the different infills suggest.

    More information and application:
    Lyu, W. et al., 2018, July. Batch Bayesian optimization via multi-objective acquisition ensemble for automated
    analog circuit design. In International conference on machine learning (pp. 3306-3314). PMLR.

    Inspired by:
    Cowen-Rivers, A.I. et al., 2022. HEBO: pushing the limits of sample-efficient hyper-parameter optimisation. Journal
    of Artificial Intelligence Research, 74, pp.1269-1349.
    """

    def __init__(self, infills: List[ConstrainedInfill] = None, constraint_strategy: ConstraintStrategy = None):
        self.infills = infills
        super().__init__(constraint_strategy=constraint_strategy)

    def _initialize(self):
        # Get set of default infills if none given
        if self.infills is None:
            if self.problem.n_obj == 1:
                self.infills = [FunctionEstimateConstrainedInfill(), LowerConfidenceBoundInfill(),
                                ExpectedImprovementInfill(), ProbabilityOfImprovementInfill()]
            else:
                self.infills = [FunctionEstimateConstrainedInfill(), LowerConfidenceBoundInfill()]

        # Reset the constraint handling strategies of the underlying infills and initialize them
        for infill in self.infills:
            if isinstance(infill, ConstrainedInfill):
                infill.constraint_strategy = IgnoreConstraints()
            infill.initialize(self.problem, self.surrogate_model, self.normalization)

        super()._initialize()

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        super().set_samples(x_train, y_train)
        for infill in self.infills:
            infill.set_samples(x_train, y_train)

    def get_n_infill_objectives(self) -> int:
        return sum([infill.get_n_infill_objectives() for infill in self.infills])

    def evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        # Merge underlying infill criteria
        f_underlying = [infill.evaluate_f(f_predict, f_var_predict) for infill in self.infills]
        return np.column_stack(f_underlying)

    def select_infill_solutions(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        # Get the Pareto front
        opt_pop = filter_optimum(population, least_infeasible=True)

        # If we have less infills available than requested, return all
        if len(opt_pop) <= n_infill:
            return opt_pop

        # If there are less infills than objectives requested, randomly select from the Pareto front
        if n_infill <= self.n_f_ic:
            i_select = np.random.choice(len(opt_pop), n_infill)
            return opt_pop[i_select]

        # Select by repeatedly eliminating crowded points from the Pareto front
        for _ in range(len(opt_pop)-n_infill):
            crowding_of_front = calc_crowding_distance(opt_pop.get('F'))

            min_crowding = np.min(crowding_of_front)
            i_min_crowding = np.where(crowding_of_front == min_crowding)[0]
            i_remove = np.random.choice(i_min_crowding) if len(i_min_crowding) > 1 else i_min_crowding[0]

            i_keep = np.ones((len(opt_pop),), dtype=bool)
            i_keep[i_remove] = False
            opt_pop = opt_pop[i_keep]
        return opt_pop


class IgnoreConstraints(ConstraintStrategy):

    def get_n_infill_constraints(self) -> int:
        return 0

    def evaluate(self, x: np.ndarray, g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], 0))
