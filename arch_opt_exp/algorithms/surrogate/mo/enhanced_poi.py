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

import itertools
import numpy as np
from scipy.special import ndtr
import matplotlib.pyplot as plt
from arch_opt_exp.algorithms.surrogate.mo.mo_modulate import *
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *

__all__ = ['EnhancedPOIInfill', 'MOEnhancedPOIInfill', 'EuclideanEIInfill', 'MOEuclideanEIInfill']


class EnhancedPOIInfill(ProbabilityOfFeasibilityInfill):
    """
    Multi-objective modification of the Probability of Improvement: the Enhanced Probability of Improvement criterion.
    It measures the probability that a given design vector will improve at least k existing Pareto-optimal solutions.

    Note that the Enhanced PoI is a relatively computationally expensive infill criterion. The computation time scales
    quadratically with the size of the Pareto front (therefore also related to the population size).

    Implementation based on:
    Hawe, G.I., "An Enhanced Probability of Improvement Utility Function for Locating Pareto Optimal Solutions", 2007
    Keane, A.J., "Statistical Improvement Criteria for Use in Multiobjective Design Optimization", 2006, 10.2514/1.16875
    """

    def __init__(self, k: int = 1, **kwargs):
        super(EnhancedPOIInfill, self).__init__(**kwargs)

        self.k = k
        self.f_pareto = None
        self.f_pareto_sorted = None
        self.i_pts_list = None

    def set_training_values(self, x_train: np.ndarray, y_train: np.ndarray):
        super(EnhancedPOIInfill, self).set_training_values(x_train, y_train)

        self.f_pareto = self.get_pareto_front(y_train[:, :self.problem.n_obj])
        self.f_pareto_sorted = self._get_f_sorted(self.f_pareto)
        self.i_pts_list = self._get_i_pts_list(self.f_pareto, self.k)

    def get_n_infill_objectives(self) -> int:
        return 1

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:

        e_poi = np.empty((f_predict.shape[0], 1))
        for i in range(f_predict.shape[0]):
            e_poi[i, 0] = self._p_dominate(self.f_pareto_sorted, f_predict[i, :], f_var_predict[i, :], self.i_pts_list)

        # Normalize tto ensure spread if even if no points with probable improvement are found
        max_e_poi = np.max(e_poi)
        if max_e_poi != 0.:
            e_poi /= max_e_poi

        e_poi[e_poi < 1e-6] = 0.
        return 1.-e_poi

    @classmethod
    def _p_dominate(cls, f_pareto_sorted: np.ndarray, f_predict: np.ndarray, var_predict: np.ndarray, i_pts_list) \
            -> float:
        """
        Calculate the probability that a point with predicted multi-dimensional objective values f_predict and variance
        var_predict will dominate at least n_dominate current Pareto points. The equation from the paper is adapter for
        more than 2 dimensions by looping through the different points based on sorted objective values per objective,
        instead of assuming a sorting order a-priori.

        Based on the paper by Hawe et al (see class description).
        """

        n, n_f = f_pareto_sorted.shape

        # Function for getting the probability of domination for some given point position for a given objective
        # dimension:
        #  0 = point that is infinitely better --> probability of domination = 0
        #  1-n = 1-indexed Pareto point
        #  n+1 = point infinitely worse --> probability of domination = 1

        def cdf_better(f, f_pred, var_pred):  # Hawe 2007, Eq. 2, 3
            return ndtr((f-f_pred)/var_pred)

        phi_cache = np.empty((n_f, n+2))
        for i_f in range(n_f):
            phi_cache[i_f, 1:-1] = cdf_better(f_pareto_sorted[:, i_f], f_predict[i_f], var_predict[i_f])
        phi_cache[:, 0] = 0.
        phi_cache[:, -1] = 1.
        phi_cache_diff = np.diff(phi_cache, axis=1)

        # Loop over all combinations
        p = 0.
        for i_pts in i_pts_list:
            # Probability of domination for all dimensions
            p_dom = 1.
            for i_f, i_pt in enumerate(i_pts):
                p_dom *= phi_cache_diff[i_f, i_pt]

            p += p_dom
        return p

    @staticmethod
    def _get_f_sorted(f_pareto: np.ndarray) -> np.ndarray:
        return np.sort(f_pareto, axis=0)

    @staticmethod
    def _get_i_pts_list(f_pareto: np.ndarray, n_dominate):
        n, n_f = f_pareto.shape
        return [i_pts for i_pts in itertools.product(*[range(n + 1) for _ in range(n_f)]) if n-sum(i_pts) >= n_dominate]

    @classmethod
    def plot_p_dominate(cls, var=None, n_dominate=1, n_pareto=5, show=True):
        def _metric(_, f_pareto_sorted, f, f_var, i_pts_list):
            return cls._p_dominate(f_pareto_sorted, f, f_var, i_pts_list)

        cls._plot_f_metric(_metric, var=var, n_dominate=n_dominate, n_pareto=n_pareto, show=show)

    @classmethod
    def _plot_f_metric(cls, metric_func, var=None, n_dominate=1, n_pareto=5, show=True):
        # Construct example Pareto front
        f_pareto = np.zeros((n_pareto, 2))
        f_pareto[:, 0] = (1.-np.cos(.5*np.pi*np.linspace(0, 1, n_pareto+2)[1:-1]))**.8
        f_pareto[:, 1] = (1.-np.cos(.5*np.pi*(1-np.linspace(0, 1, n_pareto+2)[1:-1])))**.8

        f_pareto_sorted = cls._get_f_sorted(f_pareto)
        i_pts_list = cls._get_i_pts_list(f_pareto, n_dominate)

        if np.isscalar(var):
            var = [var, var]
        if var is None:
            var = [.1, .1]

        n = 25
        x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = metric_func(f_pareto, f_pareto_sorted, [x[i, j], y[i, j]], var, i_pts_list)

        plt.figure()
        plt.title('Probability of domination (var = %r)' % var)
        c = plt.contourf(x, y, z, 50, cmap='viridis')
        plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=5, c='k')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.colorbar(c)
        if show:
            plt.show()


class MOEnhancedPOIInfill(ModulatedMOInfill):
    """
    Modulate the single-objective Enhanced POI criterion to a multi-objective criterion to increase spread along the
    currently existing Pareto front.

    Note that the Enhanced PoI is a relatively computationally expensive infill criterion.
    """

    def __init__(self, **kwargs):
        underlying = EnhancedPOIInfill(**kwargs)
        super(MOEnhancedPOIInfill, self).__init__(underlying)


class EuclideanEIInfill(EnhancedPOIInfill):
    """
    Euclidean Expected Improvement (EEI) is the multi-objective extension of the expected improvement (EI) criterion. EI
    measures the expected improvement over the current best value given a function and variance estimate. The original
    derivation by Keane et al. shows a closed-form integral. Due to implementation and computation cost concerns, this
    implementation here is based on: Keane Eq. 16 (or Palar Eq. 13),

    where P[I] (the Probability of Improvement) is replaced by the Enhanced Probability of Improvement by Hawe et al.,
        see `EnhancedPOIInfill`),

    and the centroid of the integral is replaced by the closest euclidean distance as given by Parr Eq. 6.9 (p.82).

    Implementation based on:
    Keane, A.J., "Statistical Improvement Criteria for Use in Multiobjective Design Optimization", 2006, 10.2514/1.16875
    Palar, P.S., "On Multi-Objective Efficient Global Optimization via Universal Kriging Surrogate Model", 2017,
        10.1109/CEC.2017.7969368
    Parr, J.M., "Improvement Criteria for Constraint Handling and Multiobjective Optimization", 2013
    """

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:

        eei = np.empty((f_predict.shape[0], 1))
        for i in range(f_predict.shape[0]):
            eei[i, 0] = self._eei(self.f_pareto, self.f_pareto_sorted, f_predict[i, :], f_var_predict[i, :],
                                  self.i_pts_list)

        # Normalize tto ensure spread if even if no points with probable improvement are found
        max_eei = np.max(eei)
        if max_eei != 0.:
            eei /= max_eei

        eei[eei < 1e-6] = 0.
        return 1.-eei

    @classmethod
    def _eei(cls, f_pareto: np.ndarray, f_pareto_sorted: np.ndarray, f_predict: np.ndarray, var_predict: np.ndarray,
             i_pts_list):

        # Get probability of domination
        p_dominate = cls._p_dominate(f_pareto_sorted, f_predict, var_predict, i_pts_list)

        # If the probability of domination if less than 50%, it means we are on the wrong side of the Pareto front
        if p_dominate < .5:
            return 0.

        f_pareto_min_dist = np.min(np.sqrt(np.sum((f_predict-f_pareto)**2, axis=1)))  # Parr Eq. 6.9

        return p_dominate*f_pareto_min_dist


    @classmethod
    def plot_eei(cls, var=None, n_dominate=1, n_pareto=5, show=True):
        def _metric(f_pareto, f_pareto_sorted, f, f_var, i_pts_list):
            return cls._eei(f_pareto, f_pareto_sorted, f, f_var, i_pts_list)

        cls._plot_f_metric(_metric, var=var, n_dominate=n_dominate, n_pareto=n_pareto, show=show)


class MOEuclideanEIInfill(ModulatedMOInfill):
    """
    Modulate the single-objective EEI criterion to a multi-objective criterion to increase spread along the currently
    existing Pareto front.

    Note that the EEI is a relatively computationally expensive infill criterion.
    """

    def __init__(self, **kwargs):
        underlying = EuclideanEIInfill(**kwargs)
        super(MOEuclideanEIInfill, self).__init__(underlying)


if __name__ == '__main__':
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from smt.surrogate_models.kpls import KPLS
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
    from pymoo.factory import get_problem, get_reference_directions

    # EnhancedPOIInfill.plot_p_dominate(var=.05, n_dominate=1, n_pareto=5), exit()
    # EuclideanEIInfill.plot_eei(var=.05, n_dominate=1, n_pareto=5), exit()

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = KPLS(n_comp=5, theta0=[1e-2] * 5)
        sbo_epoi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=EnhancedPOIInfill(k=1),
            termination=10, verbose=True,
        )
        sbo_mo_epoi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=MOEnhancedPOIInfill(k=1),
            termination=10, verbose=True,
        )
        sbo_eei = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=EuclideanEIInfill(k=1),
            termination=10, verbose=True,
        )
        sbo_mo_eei = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=MOEuclideanEIInfill(k=1),
            termination=10, verbose=True,
        )
        sbo_y = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimateInfill(),
            termination=100, verbose=True,
        )

        n_eval, n_eval_sbo, n_repeat = 10000, 500, 8
        algorithms = [
            (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo_y.algorithm(infill_size=25, init_size=50), sbo_y.name, n_eval_sbo),

            (sbo_epoi.algorithm(init_size=50), sbo_epoi.name, 60),  # SO infill only generates 1 pt per iteration
            (sbo_mo_epoi.algorithm(infill_size=25, init_size=50), sbo_mo_epoi.name, n_eval_sbo),

            (sbo_eei.algorithm(init_size=50), sbo_eei.name, 60),  # SO infill only generates 1 pt per iteration
            (sbo_mo_eei.algorithm(infill_size=25, init_size=50), sbo_mo_eei.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = get_problem('dtlz2', n_var=11, n_obj=2)
        pf = problem.pareto_front(get_reference_directions('das-dennis', problem.n_obj, n_partitions=12))
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(pf),
            IGDMetric(pf),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, ['cr'], ['mdr']]

        # # Plot infill selection
        # algo_eval = algorithms[2]
        # res_infill: SurrogateBasedInfill = Experimenter(problem, algo_eval[0], n_eval_max=algo_eval[2])\
        #     .run_effectiveness().algorithm.infill
        # res_infill.plot_infill_selection(show=False)
        # for ii in range(problem.n_obj):
        #     res_infill.plot_model(i_y=ii, show=False)
        # for ii in range(problem.n_constr):
        #     res_infill.plot_model(i_y=problem.n_obj+ii, show=False)
        # plt.show(), exit()

        # Run algorithms
        results = [ExperimenterResult.aggregate_results(
            Experimenter(problem, algorithm, n_eval_max=n_eval_algo, metrics=metrics)
                .run_effectiveness_parallel(n_repeat=n_repeat)) for algorithm, _, n_eval_algo in algorithms]

        # Plot metrics
        for ii, metric in enumerate(metrics):
            ExperimenterResult.plot_compare_metrics(
                results, metric.name, titles=[name for _, name, _ in algorithms],
                plot_value_names=plot_names[ii], plot_evaluations=True, show=False)
        plt.show()
