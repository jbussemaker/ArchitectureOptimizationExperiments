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
from scipy.stats import norm
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *

__all__ = ['ProbabilityOfImprovementInfill', 'LowerConfidenceBoundInfill', 'EstimateVarianceInfill',
           'ExpectedImprovementInfill']


class ProbabilityOfImprovementInfill(ProbabilityOfFeasibilityInfill):
    """
    Probability of Improvement represents the probability that some point will be better than the current best estimate
    with some offset:

    PoI(x) = Phi((T - y(x))/s(x))
    where
    - Phi is the cumulative distribution function of the normal distribution
    - T is the improvement target (current best estimate minus some offset)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate

    PoI was developed for single-objective optimization, and because of the use of the minimum current objective value,
    it tends towards suggesting improvement points only at the edges of the Pareto front, which is not suited for
    multi-objective optimization.

    Implementation based on:
    Hawe, G.I., "An Enhanced Probability of Improvement Utility Function for Locating Pareto Optimal Solutions", 2007
    """

    def __init__(self, f_min_offset: float = 0., **kwargs):
        super(ProbabilityOfImprovementInfill, self).__init__(**kwargs)

        self.f_min_offset = f_min_offset

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        f_min = np.min(self.y_train[:, :f_predict.shape[1]], axis=0)
        f_targets = f_min-self.f_min_offset

        poi = self._poi(f_targets, f_predict, f_var_predict)
        f_poi = 1.-poi

        return f_poi

    @staticmethod
    def _poi(f_targets: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        return norm.cdf((f_targets-f) / f_var)


class LowerConfidenceBoundInfill(ProbabilityOfFeasibilityInfill):
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
        super(LowerConfidenceBoundInfill, self).__init__(**kwargs)

        self.alpha = alpha

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        lcb = f_predict-self.alpha*np.sqrt(f_var_predict)
        return lcb


class EstimateVarianceInfill(ProbabilityOfFeasibilityInfill):
    """Add the function estimate and the variances directly as objectives for the infill problem, so that the trade-off
    between exploration and exploitation is automatically satisfied."""

    def __init__(self, **kwargs):
        super(EstimateVarianceInfill, self).__init__(**kwargs)

        self.var_max = None

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj*2

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        n_f = f_predict.shape[1]
        f = np.empty((f_predict.shape[0], n_f*2))

        # Function estimates as first set of objectives
        f[:, :n_f] = f_predict[:, :]

        # Variances as second set of objectives
        if self.var_max is None:
            self.var_max = np.max(f_var_predict, axis=0)
            self.var_max[self.var_max == 0] = 1.

        f[:, n_f:] = 1.-f_var_predict/self.var_max

        return f


class ExpectedImprovementInfill(ProbabilityOfFeasibilityInfill):
    """
    The Expected Improvement (EI) naturally balances exploitation and exploration by representing the expected amount
    of improvement at some point taking into accounts its probability of improvement.

    EI(x) = (f_min-y(x)) * Phi((f_min - y(x))/s(x)) + s(x) * phi((f_min - y(x)) / s(x))
    where
    - f_min is the current best point (real)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate
    - Phi is the cumulative distribution function of the normal distribution
    - phi is the probability density function of the normal distribution

    EI was developed for single-objective optimization, and because of the use of the minimum current objective value,
    it tends towards suggesting improvement points only at the edges of the Pareto front, which is not suited for
    multi-objective optimization.

    Implementation based on:
    Jones, D.R., "Efficient Global Optimization of Expensive Black-Box Functions", 1998, 10.1023/A:1008306431147
    """

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        f_min = np.min(self.y_train[:, :f_predict.shape[1]], axis=0)

        ei = self._ei(f_min, f_predict, f_var_predict)

        f_ei = 1.-ei
        return f_ei

    @staticmethod
    def _ei(f_min: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        dy = f_min-f
        ei = dy*norm.cdf(dy/f_var) + f_var*norm.pdf(dy/f_var)
        return ei


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from smt.surrogate_models.kpls import KPLS
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
    from pymoo.factory import get_problem, get_reference_directions

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = KPLS(n_comp=5, theta0=[1e-2]*5)
        sbo_poi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ProbabilityOfImprovementInfill(f_min_offset=0.),
            termination=100, verbose=True,
        )
        sbo_lcb = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=LowerConfidenceBoundInfill(alpha=2.),
            termination=100, verbose=True,
        )
        sbo_est_var = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=EstimateVarianceInfill(),
            termination=100, verbose=True,
        )
        sbo_ei = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ExpectedImprovementInfill(),
            termination=100, verbose=True,
        )
        sbo_y = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimateInfill(),
            termination=100, verbose=True,
        )

        n_eval, n_eval_sbo, n_repeat = 10000, 500, 8
        algorithms = [
            (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo_y.algorithm(infill_size=50, init_size=100), sbo_y.name, n_eval_sbo),
            (sbo_lcb.algorithm(infill_size=50, init_size=100), sbo_lcb.name, n_eval_sbo),
            (sbo_est_var.algorithm(infill_size=50, init_size=100), sbo_est_var.name, n_eval_sbo),
            # (sbo_poi.algorithm(infill_size=50, init_size=100), sbo_poi.name, n_eval_sbo),
            # (sbo_ei.algorithm(infill_size=50, init_size=100), sbo_ei.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = get_problem('dtlz2', n_var=11, n_obj=3)
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
