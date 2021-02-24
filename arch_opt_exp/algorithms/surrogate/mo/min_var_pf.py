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
import matplotlib.pyplot as plt
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *

from pymoo.model.problem import Problem
from pymoo.model.population import Population
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = ['MinVariancePFInfill']


class MinVariancePFInfill(FunctionEstimatePoFInfill):
    """
    Minimization of the Variance of Kriging-Predicted Front (MVPF).

    This works by first finding a new Pareto front directly using the predicted function value, and then selecting
    solutions with the highest variance for infill. This way, search is performed near the Pareto front, but with a
    high potential for exploration.

    Implementation based on:
    dos Passos, A.G., "Multi-Objective Optimization with Kriging Surrogates Using 'moko'", 2018, 10.1590/1679-78254324
    """

    def select_infill_solutions(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        # Get Pareto front and associated design points
        f = population.get('F')
        i_pf = self.get_i_pareto_front(f)
        pop_pf = population[i_pf]
        x_pf = pop_pf.get('X')
        g_pf = pop_pf.get('G')

        # Get variances
        f_var, _ = self.predict_variance(x_pf)

        # Select points with highest variances
        f_std_obj = 1.-np.sqrt(f_var)
        survival = RankAndCrowdingSurvival()
        pop_var = Population.new(X=x_pf, F=f_std_obj, G=g_pf)
        i_select = survival.do(infill_problem, pop_var, n_infill, return_indices=True)

        return pop_pf[i_select]

    @staticmethod
    def get_i_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        return NonDominatedSorting().do(f, only_non_dominated_front=True)


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

    MinVariancePFInfill.benchmark_evaluation_time(n_pareto=5, n_f=1000)
    MinVariancePFInfill.benchmark_evaluation_time(n_pareto=10, n_f=1000)
    MinVariancePFInfill.benchmark_evaluation_time(n_pareto=20, n_f=1000)
    exit()

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = KPLS(n_comp=5, theta0=[1e-2] * 5)
        sbo_mvpf = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=MinVariancePFInfill(),
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
            (sbo_y.algorithm(infill_size=25, init_size=50), sbo_y.name, n_eval_sbo),
            (sbo_mvpf.algorithm(infill_size=25, init_size=50), sbo_mvpf.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = get_problem('dtlz2', n_var=11, n_obj=2)
        pf = problem.pareto_front(get_reference_directions('das-dennis', problem.n_obj, n_partitions=12))
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(pf),
            IGDMetric(pf),
            InfillMetric(),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, ['min_range'], ['cr'], ['mdr']]

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
