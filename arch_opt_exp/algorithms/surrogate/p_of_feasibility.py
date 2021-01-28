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
from scipy.stats import norm
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

__all__ = ['ProbabilityOfFeasibilityInfill', 'FunctionEstimatePoFInfill', 'FunctionVariancePoFInfill']


class ProbabilityOfFeasibilityInfill(SurrogateInfill):
    """
    Infill that treats all constraints using the Probability of Feasibility (PoF) criterion.

    PoF(x) = Phi(-y(x)/sqrt(s(x)))
    where
    - Phi is the cumulative distribution function of the normal distribution
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate

    Implementation based on discussion in:
    Schonlau, M., "Global Versus Local Search in Constrained Optimization of Computer Models", 1998,
        10.1214/lnms/1215456182
    """

    def __init__(self, min_pof: float = .95):
        self.min_pof = min_pof
        super(ProbabilityOfFeasibilityInfill, self).__init__()

    @property
    def needs_variance(self):
        return True

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x)
        f_var, g_var = self.predict_variance(x)

        # Calculate Probability of Feasibility and transform to constraint (g < 0 --> PoF(g) > PoF_min)
        g_pof = g
        if self.n_constr > 0:
            g_pof = self.min_pof-self._pof(g, g_var)

        f_infill = self._evaluate_f(x, f, f_var)
        return f_infill, g_pof

    @staticmethod
    def _pof(g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        return norm.cdf(-g/np.sqrt(g_var))

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FunctionEstimatePoFInfill(ProbabilityOfFeasibilityInfill):
    """Probability of Feasible combined with direct function estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return f_predict


class FunctionVariancePoFInfill(ProbabilityOfFeasibilityInfill):
    """Probability of Feasible combined with function variance estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, x: np.ndarray, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return -np.sqrt(f_var_predict)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from smt.surrogate_models.kpls import KPLS
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from pymoo.factory import get_problem, get_reference_directions

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = KPLS(n_comp=5, theta0=[1e-2]*5)
        sbo_66 = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimatePoFInfill(min_pof=.66),
            termination=100, verbose=True,
        )
        sbo_95 = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimatePoFInfill(min_pof=.95),
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
            (sbo_95.algorithm(infill_size=50, init_size=100), sbo_95.name+' / 95', n_eval_sbo),
            (sbo_66.algorithm(infill_size=50, init_size=100), sbo_66.name+' / 66', n_eval_sbo),
            # (sbo_y.algorithm(infill_size=50, init_size=100), sbo_y.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = get_problem('C1DTLZ1', n_obj=2)
        ref_dirs = get_reference_directions('das-dennis', problem.n_obj, n_partitions=12)
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(problem.pareto_front(ref_dirs)),
            IGDMetric(problem.pareto_front(ref_dirs)),
            MaxConstraintViolationMetric(),
            InfillMetric(),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, None, ['min_range', 'g_min_range'], ['cr'], ['mdr']]

        # # Plot infill selection
        # res_infill: SurrogateBasedInfill = Experimenter(problem, algorithms[1][0], n_eval_max=algorithms[1][2])\
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
