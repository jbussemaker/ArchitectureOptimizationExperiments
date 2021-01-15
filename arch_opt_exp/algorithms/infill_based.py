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

from pymoo.model.sampling import Sampling
from pymoo.model.survival import Survival
from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.model.infill import InfillCriterion
from pymoo.model.initialization import Initialization
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

__all__ = ['InfillBasedAlgorithm']


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
