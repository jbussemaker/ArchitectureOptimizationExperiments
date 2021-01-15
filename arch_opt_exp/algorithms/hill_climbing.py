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

from pymoo.model.repair import Repair
from pymoo.model.sampling import Sampling
from pymoo.util.dominator import Dominator
from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.performance_indicator.distance_indicator import euclidean_distance
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination

__all__ = ['HillClimbingAlgorithm', 'SimulatedAnnealingAlgorithm']


class HillClimbingAlgorithm(Algorithm):
    """
    A multi-objective hill climbing (local search) algorithm. For each individual in the population, a local
    climbing is performed. Local hill climbing works by varying one design variable at a time, and replacing
    individual if the modified individual dominates the original individual.

    Implementation based on local search in:
    Glover, F., "Handbook of Metaheuristics", 2003, 10.1007/b101874
    """

    def __init__(self, pop_size=100, sampling: Sampling = None, repair: Repair = None,
                 eliminate_duplicates=DefaultDuplicateElimination(), **kwargs):
        super(HillClimbingAlgorithm, self).__init__(**kwargs)

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

        self.repair = repair
        self.initialization = Initialization(sampling, repair=repair, eliminate_duplicates=eliminate_duplicates)

        self.n_max_eval = None
        if isinstance(self.termination, MaximumFunctionCallTermination):
            self.n_max_eval = self.termination.n_max_evals

    def _initialize(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        self.pop = pop

    def _next(self):
        for i in range(len(self.pop)):
            self.pop[i] = self._local_search(self.pop[i])

    def _local_search(self, individual: Individual) -> Individual:
        for i in range(self.problem.n_var):
            # Check if we passed the number of max evaluations
            if self.n_max_eval is not None and self.evaluator.n_eval >= self.n_max_eval:
                return individual

            better_individual = self._local_search_move(individual, i)

            # Better individual found
            if better_individual is not None:
                return better_individual

        # No better individual found
        return individual

    def _local_search_move(self, individual: Individual, i_move_var: int) -> Optional[Individual]:
        # Copy the individual and modify the design vector
        mod_x = np.copy(individual.get('X'))
        xl, xu = self.problem.xl[i_move_var], self.problem.xu[i_move_var]
        mod_x[i_move_var] = np.random.random()*(xu-xl)+xl
        mod_individual = Individual(X=mod_x)
        
        if self.repair:
            mod_individual = self.repair.do(self.problem, mod_individual)

        # Evaluate
        self.evaluator.eval(self.problem, mod_individual, algorithm=self)

        # Check if the modified solution is better
        # dom_relation: 1 = modified individual is better, 0 = mutually non-dominating, -1 = old individual is better
        dom_relation = Dominator.get_relation(
            mod_individual.get('F'), individual.get('F'), cva=mod_individual.get('CV'), cvb=individual.get('CV'))

        if self._accept_mod_individual(individual, mod_individual, dom_relation):
            return mod_individual

    def _accept_mod_individual(self, individual: Individual, mod_individual: Individual, dom_relation: int) -> bool:
        if dom_relation == 1:  # Modified individual is better
            return True

        if dom_relation == 0:  # If mutually non-dominating: 50% chance to replace
            return np.random.random() > .5

        return False


class SimulatedAnnealingAlgorithm(HillClimbingAlgorithm):
    """
    Multi-objective simulated annealing algorithm. Simulated annealing works the same as hill climbing, except that
    there exists a chance that a worse solution is accepted over a better solution. This mechanism can help escape a
    local optimum and therefore could increase the chance of finding a global optimum. The chance of accepting a worse
    solution is gradually lowered during the optimization run, simulating the material annealing process.

    Implementation and more information:
    Amine, K., "Multiobjective Simulated Annealing: Principles and Algorithm Variants", 2019, 10.1155/2019/8134674
    """

    def __init__(self, t0=1., beta=.1, t_min=0., **kwargs):
        super(SimulatedAnnealingAlgorithm, self).__init__(**kwargs)

        self.t = self.t0 = t0
        self.beta = beta
        self.t_min = t_min

        self.t_hist = []
        self.ideal_point = None
        self.nadir_point = None

    def _initialize(self):
        super(SimulatedAnnealingAlgorithm, self)._initialize()

        f = self.pop.get('F')
        self.ideal_point = np.min(f, axis=0)
        self.nadir_point = np.max(f, axis=0)

    def _next(self):
        self.t_hist.append(self.t)
        super(SimulatedAnnealingAlgorithm, self)._next()
        self._update_temperature()

    def _update_temperature(self):
        """Linear cooling scheme"""
        self.t = max(self.t_min, self.t-self.beta)

    def _accept_mod_individual(self, individual: Individual, mod_individual: Individual, dom_relation: int) -> bool:
        # Just accept if modified individual dominates
        if dom_relation == 1:
            return True

        return np.random.random() < self._get_p_accept(individual, mod_individual)

    def _get_p_accept(self, individual: Individual, mod_individual: Individual) -> float:
        if self.t <= 0.:
            return 0.

        # Normalize objectives and get distance
        ideal, nadir = self.ideal_point, self.nadir_point
        norm = nadir-ideal
        f0 = individual.get('F')
        f1 = mod_individual.get('F')

        delta_e = euclidean_distance(np.array([f0]), np.array([f1]), norm=norm)[0]

        # Accept probability is based on distance and temperature
        p_accept = np.exp(-delta_e/self.t)
        return p_accept


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.problems.multi.zdt import ZDT1
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.random_search import *

    with Experimenter.temp_results():
        # Define algorithms to run
        algorithms = [
            RandomSearchAlgorithm(pop_size=100),
            NSGA2(pop_size=100),
            HillClimbingAlgorithm(pop_size=100),
            SimulatedAnnealingAlgorithm(pop_size=100, beta=.01, t_min=.1),
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

        # Run algorithms
        results = [Experimenter(problem, algorithm, n_eval_max=10000, metrics=metrics).run_effectiveness()
                   for algorithm in algorithms]

        # Plot metrics
        for ii, metric in enumerate(metrics):
            ExperimenterResult.plot_compare_metrics(
                results, metric.name, titles=[algo.__class__.__name__ for algo in algorithms],
                plot_value_names=plot_names[ii], plot_evaluations=True, show=False)
        plt.show()
