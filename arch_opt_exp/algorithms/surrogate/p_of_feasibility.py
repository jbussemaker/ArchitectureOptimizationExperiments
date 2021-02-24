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
import matplotlib.pyplot as plt
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

        f_infill = self._evaluate_f(f, f_var)
        return f_infill, g_pof

    @staticmethod
    def _pof(g: np.ndarray, g_var: np.ndarray) -> np.ndarray:
        return norm.cdf(-g/np.sqrt(g_var))

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _evaluate_f_kwargs(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> dict:
        return kwargs

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        raise RuntimeError('Not implemented')

    @classmethod
    def plot(cls, var=None, n_pareto=5, concave=False, n=100, show=True, **kwargs):
        f_pareto = np.zeros((n_pareto, 2))
        f_pareto[:, 0] = (1.-np.cos(.5*np.pi*np.linspace(0, 1, n_pareto+2)[1:-1]))**.8
        f_pareto[:, 1] = (1.-np.cos(.5*np.pi*(1-np.linspace(0, 1, n_pareto+2)[1:-1])))**.8
        if concave:
            f_pareto = 1.-f_pareto

        if np.isscalar(var):
            var = [var, var]
        if var is None:
            var = [.1, .1]

        x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        f_eval = np.column_stack([x.ravel(), y.ravel()])
        f_var = np.ones(f_eval.shape)*var

        f_static_kwargs = cls._evaluate_f_kwargs(f_eval, f_var, f_pareto, **kwargs)
        f_underlying = cls._evaluate_f_static(f_eval, f_var, f_pareto, **f_static_kwargs)

        for i_f in range(f_underlying.shape[1]):
            plt.figure()

            c = plt.contourf(x, y, f_underlying[:, i_f].reshape(x.shape), 50, cmap='Blues_r')
            for edge in c.collections:  # https://github.com/matplotlib/matplotlib/issues/4419#issuecomment-101253070
                edge.set_edgecolor('face')
            plt.contour(x, y, f_underlying[:, i_f].reshape(x.shape), 5, colors='k', alpha=.5)

            plt.colorbar(c).set_label('Criterion $f_{%d}$' % i_f)
            plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=20, c='w', edgecolors='k')
            plt.ylim([0, 1]), plt.xlim([0, 1])
            plt.xlabel('$f_0$'), plt.ylabel('$f_1$')

        if show:
            plt.show()

    @classmethod
    def benchmark_evaluation_time(cls, n_pareto=5, n_f=1000, **kwargs):
        f_pareto = np.zeros((n_pareto, 2))
        f_pareto[:, 0] = (1.-np.cos(.5*np.pi*np.linspace(0, 1, n_pareto+2)[1:-1]))**.8
        f_pareto[:, 1] = (1.-np.cos(.5*np.pi*(1-np.linspace(0, 1, n_pareto+2)[1:-1])))**.8

        n = int(np.sqrt(n_f))
        x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        f_eval = np.column_stack([x.ravel(), y.ravel()])
        f_var = np.ones(f_eval.shape)*.0025

        f_static_kwargs = cls._evaluate_f_kwargs(f_eval, f_var, f_pareto, **kwargs)

        def _benchmark():
            cls._evaluate_f_static(f_eval, f_var, f_pareto, **f_static_kwargs)

        import timeit
        timer = timeit.Timer(_benchmark)
        _, time_r = timer.autorange()
        n_r = max(1, int(1./time_r))
        ex_time = timer.repeat(repeat=10, number=n_r)

        print('%s evaluation time for n_pareto=%2d, n_f=%4d (n_r=%3d): %.2g s' %
              (cls.__name__, n_pareto, f_eval.shape[0], n_r, min(ex_time)/n_r))


class FunctionEstimatePoFInfill(ProbabilityOfFeasibilityInfill):
    """Probability of Feasible combined with direct function estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return f_predict

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        return f


class FunctionVariancePoFInfill(ProbabilityOfFeasibilityInfill):
    """Probability of Feasible combined with function variance estimate for the objectives."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return -np.sqrt(f_var_predict)

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        return -np.sqrt(f_var)


if __name__ == '__main__':
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from smt.surrogate_models.kpls import KPLS
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from pymoo.factory import get_problem, get_reference_directions

    # FunctionEstimatePoFInfill.plot(var=.05**2, n_pareto=5, show=False)
    # FunctionVariancePoFInfill.plot(var=.05**2, n_pareto=5, show=True), exit()

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
