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
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from smt.surrogate_models.surrogate_model import SurrogateModel
from arch_opt_exp.algorithms.infill_based import ModelBasedInfillCriterion

from pymoo.optimize import minimize
from pymoo.model.result import Result
from pymoo.model.problem import Problem
from pymoo.model.callback import Callback
from pymoo.model.population import Population
from pymoo.model.termination import Termination
from pymoo.model.algorithm import Algorithm, filter_optimum
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
        self.n_f_ic = None

        self.x_train = None
        self.y_train = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def needs_variance(self):
        return False

    def set_training_values(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

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

        self.n_f_ic = self.get_n_infill_objectives()

    def select_infill_solutions(self, population: Population, infill_problem: Problem, n_infill) -> Population:
        """Select infill solutions from resulting population using rank and crowding selection (from NSGA2) algorithm.
        This method can be overwritten to implement a custom selection strategy."""

        # If there is only one objective, select the best point to prevent selecting duplicate points
        if self.n_f_ic == 1:
            return filter_optimum(population, least_infeasible=True)

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

        self.opt_results: List[Result] = []

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

            if self.infill.needs_variance and not self.supports_variances:
                raise ValueError(
                    'Provided surrogate infill (%s) needs variances, but these are not supported by the underlying '
                    'surrogate model (%s)!' % (self.infill.__class__.__name__, self.surrogate_model.__class__.__name__))

        return self._surrogate_model

    @property
    def supports_variances(self):
        return self.surrogate_model.supports['variances']

    def _initialize(self):
        self.infill.initialize(self.problem, self.surrogate_model)

    def _build_model(self, new_population: Population):
        """Update the underlying model. New population is given, total population is available from self.total_pop"""

        x = self.total_pop.get('X')
        x_norm = self._normalize(x)

        y = f = self._normalize_y(self.total_pop.get('F'))
        if self.problem.n_constr > 0:
            g = self._normalize_y(self.total_pop.get('G'), keep_centered=True)
            y = np.append(f, g, axis=1)

        self.x_train = x_norm
        self.y_train = y

        self._train_model()

    def _train_model(self):
        self.surrogate_model.set_training_values(self.x_train, self.y_train)
        self.surrogate_model.train()

        self.infill.set_training_values(self.x_train, self.y_train)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        xl, xu = self.problem.xl, self.problem.xu
        return (x-xl)/(xu-xl)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        xl, xu = self.problem.xl, self.problem.xu
        return x_norm*(xu-xl)+xl

    @staticmethod
    def _normalize_y(y: np.ndarray, keep_centered=False) -> np.ndarray:
        y_min = np.min(y, axis=0)
        y_max = np.max(y, axis=0)

        norm = y_max-y_min
        norm[norm < 1e-6] = 1e-6

        if keep_centered:
            return y/norm
        return (y-y_min)/norm

    def _generate_infill_points(self, n_infill: int) -> Population:
        # Create infill problem and algorithm
        problem = self._get_infill_problem()
        algorithm = self._get_infill_algorithm()
        termination = self._get_termination()

        n_callback = 20
        if isinstance(termination, MaximumGenerationTermination):
            n_callback = int(termination.n_max_gen/5)

        # Run infill problem
        n_eval_outer = self._algorithm.evaluator.n_eval if self._algorithm is not None else -1
        result = minimize(
            problem, algorithm,
            termination=termination,
            callback=SurrogateInfillCallback(n_gen_report=n_callback, verbose=self.verbose,
                                             n_points_outer=len(self.total_pop), n_eval_outer=n_eval_outer),
        )
        self.opt_results.append(result)

        # Select infill points and denormalize the design vectors
        selected_pop = self.infill.select_infill_solutions(result.pop, problem, n_infill)
        result.opt = selected_pop

        x = self._denormalize(selected_pop.get('X'))
        return Population.new(X=x)

    def _get_infill_problem(self):
        return SurrogateInfillOptimizationProblem(self.infill, self.problem.n_var)

    def _get_termination(self):
        termination = self.termination
        if termination is None or not isinstance(termination, Termination):
            termination = MaximumGenerationTermination(n_max_gen=termination or 100)
        return termination

    def _get_infill_algorithm(self):
        return NSGA2(pop_size=self.pop_size)

    def plot_infill_selection(self, show=True):
        has_warned = False
        for i, result in enumerate(self.opt_results):
            obj = result.pop.get('F')
            obj_selected = result.opt.get('F')

            if obj.shape[1] < 2:
                log.warning('Less than 2 objectives: cannot plot solution selection')
                return
            if obj.shape[1] > 2 and not has_warned:
                has_warned = True
                log.warning('More than 2 objectives: solution selection plots might be affected')

            plt.figure()
            plt.title('Infill criteria new solutions: %d (n_obj = %d)' % (i, obj.shape[1]))
            plt.scatter(obj[:, 0], obj[:, 1], s=5, c='k', label='Identified')
            plt.scatter(obj_selected[:, 0], obj_selected[:, 1], s=5, c='g', label='Selected')
            plt.xlabel('Infill objective 1'), plt.ylabel('Infill objective 2'), plt.legend()

        if show:
            plt.show()

    def plot_model(self, i_x: List[int] = None, i_y: int = None, line_at_level: float = None, show=True):

        has_var = self.supports_variances
        is_one_dim = self.problem.n_var == 1 or (i_x is not None and len(i_x) == 1)
        if i_x is None:
            i_x = [0] if is_one_dim else [0, 1]
        if i_y is None:
            i_y = 0

        x_train, y_train = self.x_train, self.y_train
        self._train_model()

        x = np.linspace(0, 1, 100)
        if is_one_dim:
            xx = np.ones((len(x), self.problem.n_var))*.5
            xx[:, i_x[0]] = x
            y = self.surrogate_model.predict_values(xx)[:, i_y]

            plt.figure()
            plt.plot(x, y, 'k', linewidth=1.)
            if has_var:
                y_var = self.surrogate_model.predict_variances(xx)[:, i_y]
                plt.plot(x, y+y_var, '--k', linewidth=1.)
                plt.plot(x, y-y_var, '--k', linewidth=1.)
            plt.scatter(x_train[:, i_x[0]], y_train[:, i_y], c='k', marker='x')
            plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$y_{%d}$' % i_y)

        else:
            x2 = np.linspace(0, 1, 100)
            xx1, xx2 = np.meshgrid(x, x2)

            xx = np.ones((xx1.size, self.problem.n_var))*.5
            xx[:, i_x[0]] = xx1.ravel()
            xx[:, i_x[1]] = xx2.ravel()
            y = self.surrogate_model.predict_values(xx)[:, i_y]
            yy = y.reshape(xx1.shape)

            # Contour
            plt.figure()
            c = plt.contourf(xx1, xx2, yy, 50)

            if line_at_level is not None:
                plt.contour(xx1, xx2, yy, [line_at_level], colors='k', linewidth=1.)

            plt.scatter(x_train[:, i_x[0]], x_train[:, i_x[1]], c='k', marker='x')

            plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
            cb = plt.colorbar(c)
            cb.set_label('$y_{%d}$' % i_y)

            # Contour (variance)
            if has_var:
                yy_var = self.surrogate_model.predict_variances(xx)[:, i_y].reshape(xx1.shape)

                plt.figure()
                c = plt.contourf(xx1, xx2, yy_var, 50)
                plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
                cb = plt.colorbar(c)
                cb.set_label('$var_{%d}$' % i_y)

            # Error
            plt.figure()
            y_err = y_train[:, i_y]-self.surrogate_model.predict_values(x_train)[:, i_y]
            c = plt.scatter(x_train[:, i_x[0]], x_train[:, i_x[1]], s=1, c=y_err, cmap='RdYlBu',
                            norm=SymLogNorm(linthresh=1e-3, base=10.))

            plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
            cb = plt.colorbar(c)
            cb.set_label('$err_{%d}$' % i_y)

        if show:
            plt.show()


class SurrogateInfillCallback(Callback):
    """Callback for printing infill optimization progress."""

    def __init__(self, n_gen_report=20, verbose=False, n_points_outer=0, n_eval_outer=0):
        super(SurrogateInfillCallback, self).__init__()
        self.n_gen_report = n_gen_report
        self.verbose = verbose
        self.n_points_outer = n_points_outer
        self.n_eval_outer = n_eval_outer

    def notify(self, algorithm: Algorithm, **kwargs):
        if self.verbose and algorithm.n_gen % self.n_gen_report == 0:
            log.info('Surrogate infill gen %d @ %d points evaluated (%d real unique, %d eval)' %
                     (algorithm.n_gen, algorithm.evaluator.n_eval, self.n_points_outer, self.n_eval_outer))


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
