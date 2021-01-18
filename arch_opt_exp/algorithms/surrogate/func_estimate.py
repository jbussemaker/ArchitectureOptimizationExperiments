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
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

__all__ = ['FunctionEstimateInfill']


class FunctionEstimateInfill(SurrogateInfill):
    """Infill that directly uses the underlying surrogate model prediction."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x)
        return f, g


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from smt.surrogate_models.rbf import RBF
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.problems.multi.zdt import ZDT1
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.random_search import *

    with Experimenter.temp_results():
        # Define algorithms to run
        sbo = SurrogateBasedInfill(
            surrogate_model=RBF(d0=1., poly_degree=-1, reg=1e-10),
            infill=FunctionEstimateInfill(),
            verbose=True,
        )

        n_eval, n_eval_sbo, n_repeat = 10000, 500, 8
        algorithms = [
            (RandomSearchAlgorithm(pop_size=100), 'Random search', n_eval),
            (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo.algorithm(infill_size=50, init_size=100), sbo.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = ZDT1()
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(problem.pareto_front()),
            IGDMetric(problem.pareto_front()),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, ['cr'], ['mdr']]

        # # Plot infill selection
        # res_infill: SurrogateBasedInfill = Experimenter(problem, algorithms[2][0], n_eval_max=algorithms[2][2])\
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
