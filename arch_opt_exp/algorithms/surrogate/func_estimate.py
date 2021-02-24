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
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
from pymoo.performance_indicator.distance_indicator import euclidean_distance

__all__ = ['FunctionEstimateInfill', 'FunctionEstimateDistanceInfill']


class FunctionEstimateInfill(SurrogateInfill):
    """Infill that directly uses the underlying surrogate model prediction."""

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f, g = self.predict(x)
        return f, g


class FunctionEstimateDistanceInfill(SurrogateInfill):
    """
    Search mechanism based on the Coordinate Perturbation (CP) methodology of DYCORS, from:
    Regis, R.G., "Combining Radial Basis Function Surrogates and Dynamic Coordinate Search in High-Dimensional
        Expensive Black-Box Optimization", 2013, 10.1080/0305215x.2012.687731

    In CP, multiple new candidate points are generated using a scheme were randomly (and with decreasing probability)
    several of the search coordinates of the current best point are perturbed. Each of the candidate point is then
    assigned a score based on the predicted function value by the RBF surrogate) and the distance from the best point.
    The point that minimizes some weighting between these two metrics is then chosen as the next selection point.

    To generalize this to a multi-objective surrogate-based infill mechanism, the following differences are implemented:
    - Objective values are normalized between 0 and 1; use euclidean distance from origin to calculate the RBF criterion
    - Add RBF criterion and minimum distance as distinct objectives to use the Pareto front between the two to more
      naturally select a weighting between exploration and exploitation
    - Do not apply normalization on the RBF and distance criteria, as not all points are known at the time of
      evaluation, and the Pareto front is used anyway for weighting
    """

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj+1

    def get_n_infill_constraints(self) -> int:
        return self.problem.n_constr

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f_predict, g = self.predict(x)

        f_dist = np.empty((f_predict.shape[0], f_predict.shape[1]+1))
        f_dist[:, :f_predict.shape[1]] = f_predict[:, :]

        f_dist[:, -1] = self._get_dist_obj(x)
        return f_dist, g

    def _get_dist_obj(self, x: np.ndarray) -> np.ndarray:
        x_train = self.x_train
        n_train = x_train.shape[0]
        dist = np.empty((x.shape[0], n_train))
        for i_x in range(x.shape[0]):
            dist[i_x, :] = euclidean_distance(np.tile(x[i_x, 0], (n_train, 1)), x_train, norm=1.)

        # Maximize the minimum distance
        return -np.min(dist, axis=1)


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
            termination=100, verbose=True,
        )
        sbo_cp = SurrogateBasedInfill(
            surrogate_model=RBF(d0=1., poly_degree=-1, reg=1e-10),
            infill=FunctionEstimateDistanceInfill(),
            termination=100, verbose=True,
        )

        validate_loo_cv = False
        n_eval, n_eval_sbo, n_repeat = 10000, 500, 8
        algorithms = [
            (RandomSearchAlgorithm(pop_size=100), 'Random search', n_eval),
            (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo.algorithm(infill_size=10, init_size=40), sbo.name, n_eval_sbo),
            (sbo_cp.algorithm(infill_size=50, init_size=100), sbo_cp.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = ZDT1()
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(problem.pareto_front()),
            IGDMetric(problem.pareto_front()),
            InfillMetric(),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, ['min_range'], ['cr'], ['mdr']]

        if validate_loo_cv:
            from arch_opt_exp.algorithms.surrogate.validation import SurrogateQualityMetric
            metrics += [SurrogateQualityMetric(include_loo_cv=True, n_loo_cv=10)]
            plot_names += [['loo_cv']]

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
