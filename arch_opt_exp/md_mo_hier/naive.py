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
import timeit
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.design_space import *

__all__ = ['NaiveProblem']


class NaiveDesignSpace(ImplicitArchDesignSpace):

    def __init__(self, *args, **kwargs):
        self.last_corr_times = []
        super().__init__(*args, **kwargs)

    def _correct_x_corrector(self, x: np.ndarray, is_active: np.ndarray):
        s = timeit.default_timer()
        super()._correct_x_corrector(x, is_active)
        self.last_corr_times.append(timeit.default_timer()-s)

    @property
    def all_discrete_x_by_trial_and_imputation(self):
        raise MemoryError


class NaiveProblem(ArchOptProblemBase):

    def __init__(self, problem: ArchOptProblemBase, return_mod_x=False, correct=False, return_activeness=False):
        if problem.design_space.is_explicit():
            raise RuntimeError('Explicit DS not supported!')
        self._problem = problem

        design_space = NaiveDesignSpace(
            problem.des_vars,
            self._correct_x,
            self._is_conditionally_active,
            self._get_n_valid_discrete,
            self._get_n_active_cont_mean,
            self._gen_all_discrete_x,
        )

        super().__init__(design_space, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr)

        self._return_mod_x = return_mod_x
        self._do_correct = correct = correct and return_mod_x
        self._return_activeness = return_activeness and correct

    def _is_conditionally_active(self):
        if self._return_activeness:
            return self._problem.is_conditionally_active
        return np.zeros((self.n_var,), dtype=bool)

    def _get_n_valid_discrete(self) -> int:
        if self._return_activeness:
            return self._problem.get_n_valid_discrete()
        return self.get_n_declared_discrete()

    def _gen_all_discrete_x(self):
        if self._return_activeness:
            return self._problem.all_discrete_x

    def might_have_hidden_constraints(self):
        return self._problem.might_have_hidden_constraints()

    def get_n_batch_evaluate(self):
        return self._problem.get_n_batch_evaluate()

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        out = self._problem.evaluate(x.copy(), return_as_dictionary=True)

        if self._return_mod_x:
            x[:, :] = out['X']
        if self._return_activeness:
            self._correct_x_impute(x, is_active_out)

        f_out[:, :] = out['F']
        if self.n_ieq_constr > 0:
            g_out[:, :] = out['G']

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        if self._do_correct:
            x[:, :], is_active_ = self._problem.correct_x(x)
            if self._return_activeness:
                is_active[:, :] = is_active_
        else:
            self._problem.design_space.round_x_discrete(x)

    def _calc_pareto_front(self, *args, **kwargs):
        return self._problem.pareto_front()

    def _calc_pareto_set(self, *args, **kwargs):
        return self._problem.pareto_set()

    def __repr__(self):
        return f'{self.__class__.__name__}({self._problem!r}, return_mod_x={self._return_mod_x}, ' \
               f'correct={self._do_correct}, return_activeness={self._return_activeness})'
