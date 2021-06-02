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

import timeit
import logging
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from arch_opt_exp.surrogates.model import *
from arch_opt_exp.metrics_base import Metric
from arch_opt_exp.algorithms.infill_based import *
from arch_opt_exp.problems.discretization import *

from pymoo.optimize import minimize
from pymoo.model.result import Result
from pymoo.model.problem import Problem
from pymoo.model.callback import Callback
from pymoo.model.population import Population
from pymoo.model.termination import Termination
from pymoo.model.algorithm import Algorithm, filter_optimum
from pymoo.algorithms.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.max_gen import MaximumGenerationTermination

__all__ = ['SurrogateInfill', 'SurrogateBasedInfill', 'SurrogateInfillOptimizationProblem', 'InfillMetric',
           'TrainingMetric']

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
        self.is_active = None
        self.is_int_mask = None
        self.is_cat_mask = None

        self.f_infill_log = []
        self.g_infill_log = []
        self.n_eval_infill = 0
        self.time_eval_infill = 0.

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def needs_variance(self):
        return False

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray, is_int_mask: np.ndarray = None,
                    is_cat_mask: np.ndarray = None, is_active: np.ndarray = None):
        self.x_train = x_train
        self.y_train = y_train
        self.is_active = is_active

        self.is_int_mask = is_int_mask
        self.is_cat_mask = is_cat_mask

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_active, x = self._is_active(x)
        y = self.surrogate_model.predict(x, is_active=is_active)
        return self._split_f_g(y)

    def predict_variance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_active, x = self._is_active(x)
        y_var = self.surrogate_model.predict_variance(x, is_active=is_active)
        return self._split_f_g(y_var)

    def _is_active(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_active, x = MixedIntProblemHelper.is_active(self.problem, self._denormalize(x))
        return is_active, self._normalize(x)

    def _normalize(self, x) -> np.ndarray:
        return MixedIntProblemHelper.normalize(self.problem, x)

    def _denormalize(self, x_norm) -> np.ndarray:
        return MixedIntProblemHelper.denormalize(self.problem, x_norm)

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

    @staticmethod
    def get_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        i_non_dom = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return np.copy(f[i_non_dom, :])

    def reset_infill_log(self):
        self.f_infill_log = []
        self.g_infill_log = []
        self.n_eval_infill = 0
        self.time_eval_infill = 0.

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""

        s = timeit.default_timer()
        f_infill, g_infill = self._evaluate(x)
        self.time_eval_infill += timeit.default_timer()-s

        self.f_infill_log.append(f_infill)
        self.g_infill_log.append(g_infill)
        self.n_eval_infill += x.shape[0]
        return f_infill, g_infill

    def _initialize(self):
        pass

    def get_n_infill_objectives(self) -> int:
        raise NotImplementedError

    def get_n_infill_constraints(self) -> int:
        raise NotImplementedError

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate the surrogate infill objectives (and optionally constraints). Use the predict and predict_variance
        methods to query the surrogate model on its objectives and constraints."""
        raise NotImplementedError


class SurrogateBasedInfill(ModelBasedInfillCriterion):
    """Infill criterion that searches a surrogate model to generate new infill points."""

    _exclude = ['_surrogate_model', 'opt_results']

    def __init__(self, surrogate_model: Union[SurrogateModel, SurrogateModelFactory], infill: SurrogateInfill,
                 pop_size=None, termination: Union[Termination, int] = None, verbose=False, infill_force_cont=False,
                 **kwargs):
        super(SurrogateBasedInfill, self).__init__(**kwargs)

        if isinstance(surrogate_model, SurrogateModel):
            surrogate_model = SurrogateModelFactory(surrogate_model)
        self.surrogate_model_factory = surrogate_model
        self._surrogate_model = None
        self.infill = infill
        self._infill_force_cont = infill_force_cont

        self.x_train = None
        self.y_train = None
        self.y_train_min = None
        self.y_train_max = None
        self.y_train_centered = None
        self.n_train = 0
        self.is_active = None
        self.time_train = None

        self.pop_size = pop_size or 100
        self.termination = termination
        self.verbose = verbose

        self.opt_results: List[Result] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def name(self):
        return '%s / %s' % (self.surrogate_model.__class__.__name__, self.infill.__class__.__name__)

    @property
    def surrogate_model(self) -> SurrogateModel:
        if self._surrogate_model is None:
            self._surrogate_model = self.surrogate_model_factory.get()

            if self.infill.needs_variance and not self.supports_variances:
                raise ValueError(
                    'Provided surrogate infill (%s) needs variances, but these are not supported by the underlying '
                    'surrogate model (%s)!' % (self.infill.__class__.__name__, self.surrogate_model.__class__.__name__))

        return self._surrogate_model

    @property
    def supports_variances(self):
        return self.surrogate_model.supports_variance()

    @property
    def is_int_mask(self) -> Optional[np.ndarray]:
        return MixedIntProblemHelper.get_is_int_mask(self.problem)

    @property
    def is_cat_mask(self) -> Optional[np.ndarray]:
        return MixedIntProblemHelper.get_is_cat_mask(self.problem)

    def _initialize(self):
        self.infill.initialize(self.problem, self.surrogate_model)

    def _build_model(self, new_population: Population):
        """Update the underlying model. New population is given, total population is available from self.total_pop"""

        x = self.total_pop.get('X')

        if self.total_pop.has('is_active'):
            is_active = self.total_pop.get('is_active')
        else:
            is_active, x = MixedIntProblemHelper.is_active(self.problem, x)

        x_norm = self._normalize(x)

        f, self.y_train_min, self.y_train_max = self._normalize_y(self.total_pop.get('F'))
        f_is_invalid = np.bitwise_or(np.isnan(f), np.isinf(f))
        f[f_is_invalid] = 1.

        self.y_train_centered = [False]*f.shape[1]
        y = f
        if self.problem.n_constr > 0:
            g_real = self.total_pop.get('G')
            g_is_invalid = np.bitwise_or(np.isnan(g_real), np.isinf(g_real))
            g_real[g_is_invalid] = 1.

            g, g_min, g_max = self._normalize_y(self.total_pop.get('G'), keep_centered=True)
            y = np.append(f, g, axis=1)

            self.y_train_min = np.append(self.y_train_min, g_min)
            self.y_train_max = np.append(self.y_train_max, g_max)
            self.y_train_centered += [True]*g.shape[1]

        self.x_train = x_norm
        self.y_train = y
        self.is_active = is_active

        self._train_model()

    def _train_model(self):
        s = timeit.default_timer()
        is_int_mask, is_cat_mask = self.is_int_mask, self.is_cat_mask
        self.surrogate_model.set_samples(self.x_train, self.y_train, is_int_mask=is_int_mask, is_cat_mask=is_cat_mask,
                                         is_active=self.is_active)
        self.infill.set_samples(self.x_train, self.y_train, is_int_mask=is_int_mask, is_cat_mask=is_cat_mask,
                                is_active=self.is_active)

        self.surrogate_model.train()
        self.n_train += 1
        self.time_train = timeit.default_timer()-s

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return MixedIntProblemHelper.normalize(self.problem, x)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return MixedIntProblemHelper.denormalize(self.problem, x_norm)

    @staticmethod
    def _normalize_y(y: np.ndarray, keep_centered=False, y_min=None, y_max=None):
        if y_min is None:
            y_min = np.nanmin(y, axis=0)
        if y_max is None:
            y_max = np.nanmax(y, axis=0)

        norm = y_max-y_min
        norm[norm < 1e-6] = 1e-6

        if keep_centered:
            return y/norm, y_min, y_max
        return (y-y_min)/norm, y_min, y_max

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
        if self.opt_results is None:
            self.opt_results = []
        self.opt_results.append(result)

        # Select infill points and denormalize the design vectors
        selected_pop = self.infill.select_infill_solutions(result.pop, problem, n_infill)
        result.opt = selected_pop

        x = self._denormalize(selected_pop.get('X'))
        return Population.new(X=x)

    def _get_infill_problem(self):
        return SurrogateInfillOptimizationProblem(self.infill, self.problem, force_continuous=self._infill_force_cont)

    def _get_termination(self):
        termination = self.termination
        if termination is None or not isinstance(termination, Termination):
            termination = MaximumGenerationTermination(n_max_gen=termination or 100)
        return termination

    def _get_infill_algorithm(self):
        return NSGA2(pop_size=self.pop_size)

    def plot_infill_selection(self, show=True):
        has_warned = False
        for i, result in enumerate(self.opt_results or []):
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

    @staticmethod
    def plot_model_problem(surrogate_model: SurrogateModel, problem: Problem, n_pts=20):
        from arch_opt_exp.algorithms.surrogate.func_estimate import FunctionEstimateInfill
        surogate_infill = FunctionEstimateInfill()
        sbo_infill = SurrogateBasedInfill(surrogate_model=surrogate_model, infill=surogate_infill)
        sbo_infill._generate_infill_points = lambda *args, **kwargs: []

        algorithm = sbo_infill.algorithm(init_size=n_pts)
        algorithm.setup(problem, termination=MaximumGenerationTermination(n_max_gen=1))
        pop = algorithm.initialization.do(algorithm.problem, n_pts, algorithm=algorithm)
        pop = algorithm.evaluator.eval(algorithm.problem, pop, algorithm=algorithm)
        sbo_infill.do(algorithm.problem, pop, 0, algorithm=algorithm)

        sbo_infill.plot_model(i_x=[0], plot_problem=True, train=False, show=False)
        sbo_infill.plot_model(i_x=[1], plot_problem=True, train=False, show=False)
        sbo_infill.plot_model(plot_problem=True, train=False)

    def plot_model(self, i_x: List[int] = None, i_y: int = None, line_at_level: float = None, plot_problem=False,
                   train=True, show=True):

        has_var = self.supports_variances
        is_one_dim = self.problem.n_var == 1 or (i_x is not None and len(i_x) == 1)
        if i_x is None:
            i_x = [0] if is_one_dim else [0, 1]
        if i_y is None:
            i_y = 0

        x_train, y_train, is_active = self.x_train, self.y_train, self.is_active
        if train:
            self._train_model()

        def _problem_eval_norm(xx_eval):
            yy_prob = self.problem.evaluate(self._denormalize(xx_eval))[:, [i_y]]
            yy_prob, _, _ = self._normalize_y(yy_prob, keep_centered=self.y_train_centered[i_y],
                                              y_min=self.y_train_min[[i_y]], y_max=self.y_train_max[[i_y]])
            return yy_prob

        x = np.linspace(0, 1, 100)
        if is_one_dim:
            xx = np.ones((len(x), self.problem.n_var))*.5
            xx[:, i_x[0]] = x
            xx_is_active, xx_imp = MixedIntProblemHelper.is_active(self.problem, self._denormalize(xx))
            xx_imp = self._normalize(xx_imp)
            y = self.surrogate_model.predict(xx_imp, is_active=xx_is_active)[:, i_y]

            plt.figure()
            if plot_problem:
                plt.plot(x, _problem_eval_norm(xx), 'b', linewidth=1., label='Problem')
            plt.plot(x, y, 'k', linewidth=1., label='Predicted')
            if has_var:
                y_std = np.sqrt(self.surrogate_model.predict_variance(xx_imp, is_active=xx_is_active)[:, i_y])
                plt.plot(x, y+y_std, '--k', linewidth=1.)
                plt.plot(x, y-y_std, '--k', linewidth=1.)
            plt.scatter(x_train[:, i_x[0]], y_train[:, i_y], c='k', marker='x', label='Samples')
            plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$y_{%d}$' % i_y)
            plt.legend()

        else:
            x2 = np.linspace(0, 1, 100)
            xx1, xx2 = np.meshgrid(x, x2)

            xx = np.ones((xx1.size, self.problem.n_var))*.5
            xx[:, i_x[0]] = xx1.ravel()
            xx[:, i_x[1]] = xx2.ravel()
            xx_is_active, xx_imp = MixedIntProblemHelper.is_active(self.problem, self._denormalize(xx))
            xx_imp = self._normalize(xx_imp)
            y = self.surrogate_model.predict(xx_imp, is_active=xx_is_active)[:, i_y]
            yy = y.reshape(xx1.shape)

            # Contour
            plt.figure()
            c = plt.contourf(xx1, xx2, yy, 50)

            if line_at_level is not None:
                plt.contour(xx1, xx2, yy, [line_at_level], colors='k', linewidth=1.)

            plt.scatter(x_train[:, i_x[0]], x_train[:, i_x[1]], c='k', marker='x')

            plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
            plt.colorbar(c).set_label('$y_{%d}$' % i_y)

            # Contour (variance)
            if has_var:
                yy_std = np.sqrt(
                    self.surrogate_model.predict_variance(xx_imp, is_active=xx_is_active)[:, i_y].reshape(xx1.shape))

                plt.figure()
                c = plt.contourf(xx1, xx2, yy_std, 50)
                plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
                plt.colorbar(c).set_label('$std_{%d}$' % i_y)

            # Prediction error
            plt.figure()
            y_err = y_train[:, i_y]-self.surrogate_model.predict(x_train, is_active=is_active)[:, i_y]
            c = plt.scatter(x_train[:, i_x[0]], x_train[:, i_x[1]], s=1, c=y_err, cmap='RdYlBu',
                            norm=SymLogNorm(linthresh=1e-3, base=10.))

            plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
            try:
                plt.colorbar(c).set_label('$err_{%d}$' % i_y)
            except ValueError:
                pass

            # Problem error
            if plot_problem:
                yy_err = yy-_problem_eval_norm(xx).reshape(xx1.shape)
                v_max = np.max(np.abs([np.max(yy_err), np.min(yy_err)]))

                plt.figure()
                c = plt.contourf(xx1, xx2, yy_err, 50, cmap='RdBu', vmin=-v_max, vmax=v_max)
                plt.xlabel('$x_{%d}$' % i_x[0]), plt.ylabel('$x_{%d}$' % i_x[1])
                plt.colorbar(c).set_label('$err_{%d}$' % i_y)

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


class SurrogateInfillOptimizationProblem(MixedIntBaseProblem):
    """Problem class representing a surrogate infill problem given a SurrogateInfill instance."""

    def __init__(self, infill: SurrogateInfill, problem: Problem, impute=True, force_continuous=False):
        n_var = problem.n_var
        xl, xu = np.zeros(n_var), np.ones(n_var)

        is_int_mask = MixedIntProblemHelper.get_is_int_mask(problem)
        is_cat_mask = MixedIntProblemHelper.get_is_cat_mask(problem)
        if force_continuous:
            is_int_mask[:] = False
            is_cat_mask[:] = False

        is_discrete_mask = MixedIntProblemHelper.get_is_discrete_mask(problem)
        if np.any(is_discrete_mask):
            xu[is_discrete_mask] = problem.xu[is_discrete_mask]

        n_obj = infill.get_n_infill_objectives()
        n_constr = infill.get_n_infill_constraints()

        super(SurrogateInfillOptimizationProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, impute=impute, n_var=n_var, n_obj=n_obj,
            n_constr=n_constr, xl=xl, xu=xu)

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


class InfillMetric(Metric):
    """
    Metric tracking the range of infill objective values encountered during the infill search. Automatically detects if
    it is being executing on a problem with a surrogate infill algorithm.
    """

    @property
    def name(self) -> str:
        return 'infill'

    @property
    def value_names(self) -> List[str]:
        return ['min', 'max', 'min_range', 'g_min', 'g_max', 'g_min_range', 'time_infill']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        surrogate_infill = self._get_surrogate_infill(algorithm)
        if surrogate_infill is None:
            return [np.nan]*7

        f_infill, g_infill = surrogate_infill.f_infill_log, surrogate_infill.g_infill_log

        f_all = self._concatenated_values(f_infill)
        f_min = f_max = f_min_range = np.nan
        if f_all is not None:
            f_min, f_max, f_min_range = self._get_metrics(f_all)

        g_all = self._concatenated_values(g_infill)
        g_min = g_max = g_min_range = np.nan
        if g_all is not None:
            g_min, g_max, g_min_range = self._get_metrics(g_all)

        if surrogate_infill.n_eval_infill == 0:
            time_per_infill_sample = np.nan
        else:
            time_per_infill_sample = surrogate_infill.time_eval_infill/surrogate_infill.n_eval_infill

        # Reset the infill values log to track values in the next infill iteration
        surrogate_infill.reset_infill_log()

        return f_min, f_max, f_min_range, g_min, g_max, g_min_range, time_per_infill_sample

    @staticmethod
    def _concatenated_values(infill_values: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        filtered_values = [values for values in infill_values if values is not None]
        if len(filtered_values) == 0:
            return

        concatenated = np.concatenate(filtered_values, axis=0)
        if concatenated.shape[1] == 0:
            return
        return concatenated

    @staticmethod
    def _get_metrics(concat_values: np.ndarray) -> Tuple[float, ...]:
        min_values = np.nanmin(concat_values, axis=0)
        max_values = np.nanmax(concat_values, axis=0)
        ranges = max_values-min_values

        min_min, max_max = np.min(min_values), np.max(max_values)
        min_range = np.min(ranges)
        return min_min, max_max, min_range

    @staticmethod
    def _get_surrogate_infill(algorithm: Algorithm) -> Optional[SurrogateInfill]:
        if isinstance(algorithm, InfillBasedAlgorithm) and isinstance(algorithm.infill, SurrogateBasedInfill):
            return algorithm.infill.infill


class TrainingMetric(Metric):
    """Metric that tracks the number of times the used surrogate model was trained."""

    @property
    def name(self) -> str:
        return 'training'

    @property
    def value_names(self) -> List[str]:
        return ['n_train', 'n_samples', 'time_train', 'n_pop']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        n_pop = len(algorithm.pop)

        surrogate_infill = self._get_surrogate_infill(algorithm)
        if surrogate_infill is None:
            return [np.nan]*3+[n_pop]

        n_train = surrogate_infill.n_train
        n_samples = surrogate_infill.x_train.shape[0] if surrogate_infill.x_train is not None else np.nan
        time_train = surrogate_infill.time_train or np.nan

        return [n_train, n_samples, time_train, n_pop]

    @staticmethod
    def _get_surrogate_infill(algorithm: Algorithm) -> Optional[SurrogateBasedInfill]:
        if isinstance(algorithm, InfillBasedAlgorithm) and isinstance(algorithm.infill, SurrogateBasedInfill):
            return algorithm.infill


if __name__ == '__main__':
    from pymoo.problems.single.himmelblau import Himmelblau
    from arch_opt_exp.algorithms.surrogate.func_estimate import FunctionEstimateInfill

    prob = Himmelblau()
    n_pts_sample = 10

    # from arch_opt_exp.surrogates.smt.smt_krg import SMTKrigingSurrogateModel
    # sm = SMTKrigingSurrogateModel()
    from arch_opt_exp.surrogates.sklearn_models.gp import SKLearnGPSurrogateModel
    sm = SKLearnGPSurrogateModel()

    SurrogateBasedInfill.plot_model_problem(sm, prob, n_pts=n_pts_sample)
