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

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
import itertools
import numpy as np
from pymoo.core.repair import Repair
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling, sampling_lhs_unit

__all__ = ['RepairedExhaustiveSampling', 'RepairedLatinHypercubeSampling', 'RepairedRandomSampling', 'get_init_sampler']


def get_init_sampler(repair: Repair = None, lhs=True, **kwargs):
    sampling = RepairedLatinHypercubeSampling(repair=repair, **kwargs) \
        if lhs else RepairedExhaustiveSampling(repair=repair, **kwargs)
    # Samples are already repaired because we're using the repaired samplers
    return Initialization(sampling, eliminate_duplicates=DefaultDuplicateElimination())


class RepairedExhaustiveSampling(Sampling):

    def __init__(self, repair: Repair = None, n_cont=5, remove_duplicates=True):
        super().__init__()
        self._repair = repair
        self._n_cont = n_cont
        self._remove_duplicates = remove_duplicates

    def _do(self, problem: Problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        is_cont = np.ones((len(xl),), dtype=bool)
        if problem.vars is not None:
            for i, var in enumerate(problem.vars):
                if not isinstance(var, Real):
                    is_cont[i] = False

        opt_values = [np.linspace(xl[i], xu[i], self._n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1)
                      for i in range(len(xl))]
        x = np.array([np.array(dv) for dv in itertools.product(*opt_values)])

        pop = Population.new(X=x)
        pop = self._repair.do(problem, pop)

        if self._remove_duplicates:
            gb_needed = ((len(pop)**2)*8)/(1024**3)
            if gb_needed < 2:
                pop = DefaultDuplicateElimination().do(pop)

        return pop.get('X')


class RepairedLatinHypercubeSampling(LatinHypercubeSampling):
    """Latin hypercube sampling only returning repaired samples."""

    def __init__(self, repair: Repair = None, **kwargs):
        super().__init__(**kwargs)
        self._repair = repair

    def _do(self, problem: Problem, n_samples, **kwargs):
        if self._repair is None:
            return super()._do(problem, n_samples, **kwargs)

        xl, xu = problem.bounds()

        X = sampling_lhs_unit(n_samples, problem.n_var, smooth=self.smooth)
        X = self.repair_x(problem, X, xl, xu)

        if self.criterion is not None:
            score = self.criterion(X)
            for j in range(1, self.iterations):

                _X = sampling_lhs_unit(n_samples, problem.n_var, smooth=self.smooth)
                _X = self.repair_x(problem, _X, xl, xu)
                _score = self.criterion(_X)

                if _score > score:
                    X, score = _X, _score

        return xl + X * (xu - xl)

    def repair_x(self, problem, x, xl, xu):
        x_abs = x*(xu-xl)+xl
        x_abs = self._repair.do(problem, Population.new(X=x_abs)).get("X")
        return (x_abs-xl)/(xu-xl)


class RepairedRandomSampling(FloatRandomSampling):

    _n_comb_gen_all_max = 100e3

    def __init__(self, repair: Repair = None):
        self._repair = repair
        self.track_x_last_init = False
        self.x_last_init = None
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        is_cont = np.ones((len(xl),), dtype=bool)
        if problem.vars is not None:
            for i, var in enumerate(problem.vars):
                if not isinstance(var, Real):
                    is_cont[i] = False

        n_cont = 5
        opt_values = [np.linspace(xl[i], xu[i], n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1)
                      for i in range(len(xl))]
        n_opt_values = int(np.cumprod([len(values) for values in opt_values], dtype=float)[-1])
        x = None
        if n_opt_values < self._n_comb_gen_all_max:
            try:
                x = np.array([np.array(dv) for dv in itertools.product(*opt_values)])
                if n_samples < x.shape[0]:
                    i_x = np.random.choice(x.shape[0], size=n_samples, replace=False)
                    x = x[i_x, :]
            except MemoryError:
                pass
        if x is None:
            x = np.empty((n_samples, len(opt_values)))
            for i_x in range(n_samples):
                x[i_x, :] = [np.random.choice(opt_values_i) for opt_values_i in opt_values]

        if self.track_x_last_init:
            self.x_last_init = x
        x = self._repair.do(problem, Population.new(X=x)).get("X")
        return x
