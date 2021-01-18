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

from typing import *

from pymoo.model.repair import Repair
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.survival import Survival
from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.model.infill import InfillCriterion
from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import DuplicateElimination
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

__all__ = ['InfillBasedAlgorithm', 'ModelBasedInfillCriterion']


class InfillBasedAlgorithm(Algorithm):
    """Algorithm that uses some infill criterion to generate new points. The population is kept at the same size
    throughout the optimization, and updated by rank and crowding survival (like NSGA2)."""

    def __init__(self, infill_criterion: InfillCriterion, infill_size=None, init_sampling: Sampling = None,
                 init_size=100, survival: Survival = None, **kwargs):
        super(InfillBasedAlgorithm, self).__init__(**kwargs)

        self.init_size = init_size
        self.infill_size = infill_size or self.init_size

        self.infill = infill_criterion

        if init_sampling is None:
            init_sampling = LatinHypercubeSampling()

        self.initialization = Initialization(init_sampling, repair=infill_criterion.repair,
                                             eliminate_duplicates=infill_criterion.eliminate_duplicates)

        self.survival = survival or RankAndCrowdingSurvival()

    def _initialize(self):
        pop = self.initialization.do(self.problem, self.init_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)
        self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)

    def _next(self):
        # Create offspring using infill criterion
        off = self.infill.do(self.problem, self.pop, self.infill_size, algorithm=self)

        # Stop if no new offspring is generated
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # Evaluate and update population
        self.evaluator.eval(self.problem, off, algorithm=self)

        pop = Population.merge(self.pop, off)
        self.pop = self.survival.do(self.problem, pop, self.init_size, algorithm=self)


class ModelBasedInfillCriterion(InfillCriterion):
    """Infill criterion based on searching a model of the underlying design space."""

    def __init__(self, repair: Repair = None, eliminate_duplicates: DuplicateElimination = None, **kwargs):
        if eliminate_duplicates is None:
            eliminate_duplicates = DefaultDuplicateElimination()

        super(ModelBasedInfillCriterion, self).__init__(
            repair=repair, eliminate_duplicates=eliminate_duplicates, **kwargs)

        self._is_init = None
        self.problem: Problem = None
        self.total_pop: Population = None
        self._algorithm: Optional[Algorithm] = None

    def algorithm(self, infill_size=None, init_sampling: Sampling = None, init_size=100, survival: Survival = None,
                  **kwargs) -> InfillBasedAlgorithm:
        return InfillBasedAlgorithm(self, infill_size=infill_size, init_sampling=init_sampling, init_size=init_size,
                                    survival=survival, **kwargs)

    def do(self, problem, pop, n_offsprings, **kwargs):
        self._algorithm = kwargs.pop('algorithm', None)

        # Check if we need to initialize
        if self._is_init is None:
            self.problem = problem
            self._is_init = problem

            self._initialize()

        elif self._is_init is not problem:
            raise RuntimeError('An instance of a ModelBasedInfillCriterion can only be used with one Problem!')

        # Build the model
        if self.total_pop is None:
            self.total_pop = pop
            new_population = pop
        else:
            new_population = self.eliminate_duplicates.do(pop, self.total_pop)
            self.total_pop = Population.merge(self.total_pop, new_population)

        self._build_model(new_population)

        # Generate offspring (based on parent class)
        off = self._generate_infill_points(n_offsprings)

        off = self.repair.do(problem, off, **kwargs)
        off = self.eliminate_duplicates.do(off, pop)
        return off

    def _do(self, *args, **kwargs):
        raise RuntimeError

    def _initialize(self):
        """Initialize the underlying model. The problem is available from self.problem"""

    def _build_model(self, new_population: Population):
        """Update the underlying model. New population is given, total population is available from self.total_pop"""
        raise NotImplementedError

    def _generate_infill_points(self, n_infill: int) -> Population:
        """Generate new infill points"""
        raise NotImplementedError
