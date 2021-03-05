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
from pymoo.model.repair import Repair
from pymoo.model.population import Population
from pymoo.model.problem import MetaProblem, Problem

__all__ = ['MixedIntBaseProblem', 'MixedIntProblemHelper', 'MixedIntProblem', 'MixedIntRepair']


class MixedIntBaseProblem(Problem):
    """
    Base class for defining a mixed-discrete problem.

    The following nomenclature is adhered to:
    - Continuous variables: real-valued (float) design variables
    - Integer variables: discrete design variables, where a notion of order and distance is defined
    - Categorical variables: discrete design variables, where order and distance is undefined
    - Discrete variables: integer or categorical variables
    - Mixed-integer or mixed-discrete problem: problem with both continuous and discrete variables

    Design variables can take the following values:
    - Continuous: any floating-point value between its lower and upper bounds (inclusive)
    - Integer: any integer (whole number) between its lower and upper bounds (inclusive)
    - Categorical: encoded as integers between 0 and n_categories-1

    Examples:
    - Continuous: engine bypass ratio between 3 and 8
    - Integer: number of engines on an aircraft between 2 and 4
    - Categorical: T-tail (0), V-tail (1), or conventional tail (2)
    """

    is_int_mask: np.ndarray
    is_cat_mask: np.ndarray

    @property
    def is_discrete_mask(self):
        return np.bitwise_or(self.is_int_mask, self.is_cat_mask)

    @property
    def is_cont_mask(self):
        return ~self.is_discrete_mask

    def get_repair(self) -> Repair:
        return MixedIntRepair(self.is_discrete_mask)

    def correct_x(self, x: np.ndarray) -> np.ndarray:
        return MixedIntRepair.correct_x(self.is_discrete_mask, x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return self.normalize_mi(x, self.xl, self.xu, self.is_cont_mask, self.is_discrete_mask)

    @staticmethod
    def normalize_mi(x: np.ndarray, xl, xu, is_cont_mask, is_discrete_mask) -> np.ndarray:
        x_norm = x.copy()
        x_norm[:, is_cont_mask] = (x[:, is_cont_mask]-xl[is_cont_mask])/(xu[is_cont_mask]-xl[is_cont_mask])
        x_norm[:, is_discrete_mask] = x[:, is_discrete_mask]-xl[is_discrete_mask]
        return x_norm

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return self.denormalize_mi(x_norm, self.xl, self.xu, self.is_cont_mask, self.is_discrete_mask)

    @staticmethod
    def denormalize_mi(x_norm: np.ndarray, xl, xu, is_cont_mask, is_discrete_mask) -> np.ndarray:
        x = x_norm.copy()
        x[:, is_cont_mask] = x[:, is_cont_mask]*(xu[is_cont_mask]-xl[is_cont_mask])+xl[is_cont_mask]
        x[:, is_discrete_mask] = x[:, is_discrete_mask]+xl[is_discrete_mask]
        return x

    def so_run(self, n_repeat=8, pop_size=100, n_eval_max=5000, show=True):
        from pymoo.algorithms.so_genetic_algorithm import GA
        from pymoo.factory import get_sampling, get_crossover, get_mutation
        from arch_opt_exp.experimenter import Experimenter, ExperimenterResult
        from arch_opt_exp.metrics.performance import BestObjMetric, MaxConstraintViolationMetric
        from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, \
            MixedVariableCrossover

        with Experimenter.temp_results():
            algorithm = GA(
                pop_size=pop_size,
                sampling=MixedVariableSampling(self.is_discrete_mask, {
                    False: get_sampling('real_random'),
                    True: get_sampling('int_random'),
                }),
                crossover=MixedVariableCrossover(self.is_discrete_mask, {
                    False: get_crossover('real_sbx', prob=.9, eta=3.),
                    True: get_crossover('int_ux', prob=.9),
                }),
                mutation=MixedVariableMutation(self.is_discrete_mask, {
                    False: get_mutation('real_pm', eta=3.),
                    True: get_mutation('bin_bitflip'),
                }),
                repair=self.get_repair(),
            )

            metrics = [BestObjMetric(), MaxConstraintViolationMetric()]
            exp = Experimenter(self, algorithm, n_eval_max=n_eval_max, metrics=metrics)
            if n_repeat == 1:
                results = exp.run_effectiveness()
            else:
                results = ExperimenterResult.aggregate_results(exp.run_effectiveness_parallel(n_repeat=n_repeat))

            for i in range(len(metrics)):
                ExperimenterResult.plot_compare_metrics([results], metrics[i].name, plot_evaluations=True,
                                                        show=show and (i == len(metrics)-1))

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError


class MixedIntProblemHelper:

    @staticmethod
    def get_is_int_mask(problem: Problem) -> np.ndarray:
        if isinstance(problem, MixedIntBaseProblem) or hasattr(problem, 'is_int_mask'):
            return problem.is_int_mask

        return np.zeros((problem.n_var,), dtype=bool)

    @staticmethod
    def get_is_cat_mask(problem: Problem) -> np.ndarray:
        if isinstance(problem, MixedIntBaseProblem) or hasattr(problem, 'is_cat_mask'):
            return problem.is_cat_mask

        return np.zeros((problem.n_var,), dtype=bool)

    @classmethod
    def get_is_discrete_mask(cls, problem: Problem) -> np.ndarray:
        return np.bitwise_or(cls.get_is_int_mask(problem), cls.get_is_cat_mask(problem))

    @classmethod
    def get_is_cont_mask(cls, problem: Problem) -> np.ndarray:
        return ~cls.get_is_discrete_mask(problem)

    @classmethod
    def normalize(cls, problem: Problem, x: np.ndarray) -> np.ndarray:
        if isinstance(problem, MixedIntBaseProblem) or hasattr(problem, 'normalize'):
            return problem.normalize(x)

        is_cont_mask = cls.get_is_cont_mask(problem)
        if np.all(is_cont_mask):
            xl, xu = problem.xl, problem.xu
            return (x-xl)/(xu-xl)

        return MixedIntBaseProblem.normalize_mi(
            x, problem.xl, problem.xu, is_cont_mask, cls.get_is_discrete_mask(problem))

    @classmethod
    def denormalize(cls, problem: Problem, x_norm: np.ndarray) -> np.ndarray:
        if isinstance(problem, MixedIntBaseProblem) or hasattr(problem, 'denormalize'):
            return problem.denormalize(x_norm)

        is_cont_mask = cls.get_is_cont_mask(problem)
        if np.all(is_cont_mask):
            xl, xu = problem.xl, problem.xu
            return x_norm*(xu-xl)+xl

        return MixedIntBaseProblem.denormalize_mi(
            x_norm, problem.xl, problem.xu, is_cont_mask, cls.get_is_discrete_mask(problem))

    @staticmethod
    def get_repair(problem: Problem) -> Optional[Repair]:
        if isinstance(problem, MixedIntBaseProblem) or hasattr(problem, 'get_repair'):
            return problem.get_repair()


class MixedIntProblem(MetaProblem, MixedIntBaseProblem):
    """Creates a mixed-integer problem from an existing problem, by mapping the first n (if not given: all) variables to
    integers, with a given number of choices."""
    problem: Problem

    def __init__(self, problem: Problem, n_choices=11, n_vars_mixed_int: int = None):
        super(MixedIntProblem, self).__init__(problem)

        self.n_choices = n_choices

        if n_vars_mixed_int is None:
            n_vars_mixed_int = self.problem.n_var
        self.n_vars_mixed_int = n_vars_mixed_int

        if not self.problem.has_bounds():
            raise ValueError('Underlying problem should have bounds defined')
        self.xl_orig = self.problem.xl
        self.xu_orig = self.problem.xu

        self.xl = xl = np.copy(self.problem.xl)
        xl[:self.n_vars_mixed_int] = 0
        self.xu = xu = np.copy(self.problem.xu)
        xu[:self.n_vars_mixed_int] = self.n_choices-1

        n_vars_real = self.problem.n_var-self.n_vars_mixed_int
        self.mask = ['int' for _ in range(self.n_vars_mixed_int)]+['real' for _ in range(n_vars_real)]
        self.is_int_mask = [self.mask[i] == 'int' for i in range(len(self.mask))]
        self.is_cat_mask = np.zeros((len(self.is_int_mask),), dtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        x_underlying = self._map_x(self._correct_x(x))
        self.problem._evaluate(x_underlying, out, *args, **kwargs)

    def _map_x(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)

        n = self.n_vars_mixed_int
        xl, xu = self.xl, self.xu
        xl_orig, xu_orig = self.xl_orig, self.xu_orig

        x[:, :n] = ((x[:, :n]-xl[:n])/(xu[:n]-xl[:n]))*(xu_orig[:n]-xl_orig[:n])+xl_orig[:n]
        return x

    def _correct_x(self, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)

        n = self.n_vars_mixed_int
        x[:, :n] = np.round(x[:, :n].astype(np.float64)).astype(np.int)
        return x


class MixedIntRepair(Repair):
    """Repair operator to make sure that integer variables are actually integers after sampling or mating."""

    def __init__(self, is_discrete_mask):
        super(MixedIntRepair, self).__init__()

        self.is_discrete_mask = is_discrete_mask

    def _do(self, problem: Problem, pop: Union[Population, np.ndarray], **kwargs):
        is_array = not isinstance(pop, Population)
        x = pop if is_array else pop.get("X")

        x = self.correct_x(self.is_discrete_mask, x)

        if is_array:
            return x
        pop.set("X", x)
        return pop

    @staticmethod
    def correct_x(is_discrete_mask, x: np.ndarray) -> np.ndarray:
        x = np.copy(x)
        x[:, is_discrete_mask] = np.round(x[:, is_discrete_mask].astype(np.float64)).astype(np.int)
        return x
