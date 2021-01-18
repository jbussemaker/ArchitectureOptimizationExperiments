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

import copy
import logging
import numpy as np
from typing import *
from smt.surrogate_models.surrogate_model import SurrogateModel
from arch_opt_exp.algorithms.infill_based import ModelBasedInfillCriterion

from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.model.callback import Callback
from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.model.termination import Termination
from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.util.termination.max_gen import MaximumGenerationTermination

__all__ = ['SurrogateInfill', 'SurrogateModelFactory', 'SurrogateBasedInfill', 'SurrogateInfillOptimizationProblem']

log = logging.getLogger('arch_opt_exp.sur')


class SurrogateInfill:
    """Base class for a surrogate model infill criterion."""

    _exclude = ['surrogate_model']

    def __init__(self):
        self.problem: Problem = None
        self.surrogate_model: SurrogateModel = None
        self.n_constr = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def needs_variance(self):
        return False

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = self.surrogate_model.predict_values(x)
        return self._split_f_g(y)

    def predict_variance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_var = self.surrogate_model.predict_variances(x)
        return self._split_f_g(y_var)

    def _split_f_g(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_constr > 0:
            return y[:, :-self.n_constr], y[:, -self.n_constr:]
        return y, np.zeros((y.shape[0], 0))

    def initialize(self, problem: Problem, surrogate_model: SurrogateModel):
        self.problem = problem
        self.n_constr = problem.n_constr

        self.surrogate_model = surrogate_model

        self._initialize()

    @classmethod
    def select_infill_solutions(cls, population: Population, infill_problem: Problem, n_infill) -> Population:
        """Select infill solutions from resulting population using rank and crowding selection (from NSGA2) algorithm.
        This method can be overwritten to implement a custom selection strategy."""
        survival = RankAndCrowdingSurvival()
        return survival.do(infill_problem, population, n_infill)

    def _initialize(self):
        pass

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def get_n_infill_constraints(self) -> int:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""
        raise NotImplementedError


class SurrogateModelFactory:

    def __init__(self, klass: Type[SurrogateModel], **kwargs):
        self.klass = klass
        self.kwargs = kwargs

    def get(self) -> SurrogateModel:
        return self.klass(**self.kwargs)

    @classmethod
    def from_surrogate_model(cls, surrogate_model: SurrogateModel):
        klass = surrogate_model.__class__

        default_opts = {key: data['default'] for key, data in surrogate_model.options._declared_entries.items()}
        kwargs = {key: copy.deepcopy(value) for key, value in surrogate_model.options._dict.items()
                  if value != default_opts[key]}

        return cls(klass, **kwargs)


class SurrogateBasedInfill(ModelBasedInfillCriterion):
    """Infill criterion that searches a surrogate model to generate new infill points."""

    _exclude = ['_surrogate_model']

    def __init__(self, surrogate_model: Union[SurrogateModel, SurrogateModelFactory], infill: SurrogateInfill,
                 pop_size=None, termination: Union[Termination, int] = None, verbose=False, **kwargs):
        super(SurrogateBasedInfill, self).__init__(**kwargs)

        if infill.needs_variance and not surrogate_model.supports['variances']:
            raise ValueError(
                'Provided surrogate infill (%s) needs variances, but these are not supported by the underlying '
                'surrogate model (%s)!' % (infill.__class__.__name__, surrogate_model.__class__.__name__))

        if isinstance(surrogate_model, SurrogateModel):
            surrogate_model = SurrogateModelFactory.from_surrogate_model(surrogate_model)
        self.surrogate_model_factory = surrogate_model
        self._surrogate_model = None
        self.infill = infill

        self.x_train = None
        self.y_train = None

        self.pop_size = pop_size or 100
        self.termination = termination
        self.verbose = verbose

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def name(self):
        return '%s / %s' % (self.surrogate_model.__class__.__name__, self.infill.__class__.__name__)

    @property
    def surrogate_model(self):
        if self._surrogate_model is None:
            self._surrogate_model = self.surrogate_model_factory.get()
            self._surrogate_model.options['print_global'] = False
        return self._surrogate_model

    def _initialize(self):
        self.infill.initialize(self.problem, self.surrogate_model)

    def _build_model(self, new_population: Population):
        """Update the underlying model. New population is given, total population is available from self.total_pop"""

        x = self.total_pop.get('X')
        x_norm = self._normalize(x)

        y = f = self.total_pop.get('F')
        if self.problem.n_constr > 0:
            g = self.total_pop.get('G')
            y = np.append(f, g, axis=1)

        self.x_train = x_norm
        self.y_train = y

        self.surrogate_model.set_training_values(x_norm, y)
        self.surrogate_model.train()

    def _normalize(self, x) -> np.ndarray:
        xl, xu = self.problem.xl, self.problem.xu
        return (x-xl)/(xu-xl)

    def _denormalize(self, x_norm) -> np.ndarray:
        xl, xu = self.problem.xl, self.problem.xu
        return x_norm*(xu-xl)+xl

    def _generate_infill_points(self, n_infill: int) -> Population:
        # Create infill problem and algorithm
        problem = self._get_infill_problem()
        algorithm = self._get_infill_algorithm()
        termination = self._get_termination()

        # Run infill problem
        result = minimize(
            problem, algorithm,
            termination=termination,
            callback=SurrogateInfillCallback(verbose=self.verbose, n_points_outer=len(self.total_pop)),
        )

        # Select infill points and denormalize the design vectors
        pop = self.infill.select_infill_solutions(result.pop, problem, n_infill)

        x = self._denormalize(pop.get('X'))
        return Population.new(X=x)

    def _get_infill_problem(self):
        return SurrogateInfillOptimizationProblem(self.infill, self.problem.n_var)

    def _get_termination(self):
        termination = self.termination
        if termination is None or not isinstance(termination, Termination):
            termination = MaximumGenerationTermination(n_max_gen=termination or 20)
        return termination

    def _get_infill_algorithm(self):
        return NSGA2(pop_size=self.pop_size)


class SurrogateInfillCallback(Callback):
    """Callback for printing infill optimization progress."""

    def __init__(self, n_gen_report=5, verbose=False, n_points_outer=0):
        super(SurrogateInfillCallback, self).__init__()
        self.n_gen_report = n_gen_report
        self.verbose = verbose
        self.n_points_outer = n_points_outer

    def notify(self, algorithm: Algorithm, **kwargs):
        if self.verbose and algorithm.n_gen % self.n_gen_report == 0:
            log.info('Surrogate infill gen %d @ %d points evaluated (%d real unique)' %
                     (algorithm.n_gen, algorithm.evaluator.n_eval, self.n_points_outer))


class SurrogateInfillOptimizationProblem(Problem):
    """Problem class representing a surrogate infill problem given a SurrogateInfill instance."""

    def __init__(self, infill: SurrogateInfill, n_var):
        xl, xu = np.zeros(n_var), np.ones(n_var)

        n_obj = infill.get_n_infill_objectives()
        n_constr = infill.get_n_infill_constraints()

        super(SurrogateInfillOptimizationProblem, self).__init__(n_var, n_obj, n_constr, xl=xl, xu=xu)

        self.infill = infill

    def _evaluate(self, x, out, *args, **kwargs):
        f, g = self.infill.evaluate(x)

        if f.shape != (x.shape[0], self.n_obj):
            raise RuntimeError('Wrong objective results shape: %r != %r' % (f.shape, (x.shape[0], self.n_obj)))
        out['F'] = f

        if g is None:
            if self.n_constr > 0:
                raise RuntimeError('Expected constraint values')
        else:
            if g.shape != (x.shape[0], self.n_constr):
                raise RuntimeError('Wrong constraint results shape: %r != %r' % (g.shape, (x.shape[0], self.n_constr)))
            out['G'] = g
