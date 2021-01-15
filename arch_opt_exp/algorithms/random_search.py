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

from pymoo.model.repair import Repair
from pymoo.util.misc import has_feasible
from pymoo.model.sampling import Sampling
from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.model.initialization import Initialization
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination

__all__ = ['RandomSearchAlgorithm']


class RandomSearchAlgorithm(Algorithm):
    """
    Algorithm that simply performs a random sampling at every step and keeps the best n design points. Useful as a
    benchmark algorithm for comparing new algorithms against.
    """

    def __init__(self, pop_size=100, sampling: Sampling = None, repair: Repair = None,
                 eliminate_duplicates=DefaultDuplicateElimination(), **kwargs):
        super(RandomSearchAlgorithm, self).__init__(**kwargs)

        self.pop_size = pop_size

        if sampling is None:
            sampling = FloatRandomSampling()

        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.initialization = Initialization(sampling, repair=repair, eliminate_duplicates=eliminate_duplicates)

        # After merging new and current population, use NSGA2's ranking and crowding mechanism to select the current
        # best points
        self.survival = RankAndCrowdingSurvival()

    def _initialize(self):
        pop = self._sample_eval_new_pop()

        # Run the survival to set attributes
        self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)

    def _next(self):
        new_pop = self._sample_eval_new_pop()

        # Merge the current and new populations and use rank and crowding distance to keep the best individuals
        merged_pop = Population.merge(self.pop, new_pop)
        self.pop = self.survival.do(self.problem, merged_pop, self.pop_size, algorithm=self)

    def _sample_eval_new_pop(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        return pop

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.problems.multi.zdt import ZDT1
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *

    with Experimenter.temp_results():
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

        # Run the algorithm
        algo = RandomSearchAlgorithm(pop_size=100)
        exp = Experimenter(problem, algo, n_eval_max=10000, metrics=metrics)
        res = exp.run_effectiveness()

        # Run the comparison algorithm: NSGA2
        algo_compare = NSGA2(pop_size=100)
        exp_compare = Experimenter(problem, algo_compare, n_eval_max=10000, metrics=metrics)
        res_compare = exp_compare.run_effectiveness()

        # Plot metrics
        for ii, metric in enumerate(metrics):
            ExperimenterResult.plot_compare_metrics(
                [res, res_compare], metric.name, titles=[algo.__class__.__name__, 'NSGA2'],
                plot_value_names=plot_names[ii], show=False)
        plt.show()
