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
import numpy as np
from typing import *
from sb_arch_opt.problem import *
from sb_arch_opt.algo.simple_sbo.algo import *
from sb_arch_opt.algo.simple_sbo.infill import *

__all__ = ['HiddenConstraintStrategy', 'HiddenConstraintsSBO', 'HCInfill']


class HiddenConstraintStrategy:
    """
    Base class for implementing a strategy for dealing with hidden constraints.
    """

    @staticmethod
    def is_failed(y: np.ndarray):
        return np.any(~np.isfinite(y), axis=1)

    def needs_variance(self) -> bool:
        """Whether the strategy needs a surrogate model that also provides variance estimates"""
        return False

    def initialize(self, problem: ArchOptProblemBase):
        pass

    def mod_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Modify inputs and outputs for the surrogate model used for the main infill function"""
        return x, y

    def prepare_infill_search(self, x: np.ndarray, y: np.ndarray):
        """Prepare infill search given the (non-modified) normalized inputs and outputs"""

    def adds_infill_constraint(self) -> bool:
        """Whether the strategy adds an inequality constraint to the infill search problem"""
        return False

    def evaluate_infill_constraint(self, x: np.ndarray) -> np.ndarray:
        """If the problem added an infill constraint, evaluate it here, returning an nx-length vector"""

    def mod_infill_objectives(self, x: np.ndarray, f_infill: np.ndarray) -> np.ndarray:
        """Modify the infill objectives (in-place)"""
        return f_infill

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class HiddenConstraintsSBO(SBOInfill):
    """SBO algorithm with hidden constraint strategy"""

    def __init__(self, *args, hc_strategy: HiddenConstraintStrategy = None, **kwargs):
        self.hc_strategy = hc_strategy
        super().__init__(*args, **kwargs)

    @property
    def surrogate_model(self):
        model = super().surrogate_model
        if self.hc_strategy is None:
            raise ValueError('HC strategy not set!')
        if self.hc_strategy.needs_variance() and not self.supports_variances:
            raise ValueError(f'HC {self.hc_strategy!r} needs variance which is not provided by the surrogate model')
        return model

    def _get_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.hc_strategy.mod_xy_train(x, y)

    def _build_model(self):
        if self.hc_strategy is None:
            raise ValueError('HC strategy not set!')
        self.hc_strategy.initialize(self.problem)
        super()._build_model()

        x = self.total_pop.get('X')
        y = self.total_pop.get('F')
        if self.problem.n_ieq_constr > 0:
            y = np.column_stack([y, self.total_pop.get('G')])

        self.hc_strategy.prepare_infill_search(x, y)

    def _get_infill_problem(self, infill: SurrogateInfill = None, force_new_points=None):
        hc_infill = HCInfill(self.infill if infill is None else infill, self.hc_strategy)
        return super()._get_infill_problem(hc_infill, force_new_points=force_new_points)

    def plot_state(self, x_infill=None, save_path=None, plot_std=False, show=True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import CenteredNorm
        problem = self.problem
        total_pop = self.total_pop
        x_train = total_pop.get('X')
        is_failed_train = ArchOptProblemBase.get_failed_points(total_pop)
        n_fail = np.sum(is_failed_train)

        x1, x2 = np.linspace(problem.xl[0], problem.xu[0], 100), np.linspace(problem.xl[1], problem.xu[1], 100)
        xx1, xx2 = np.meshgrid(x1, x2)
        x_eval = np.ones((len(xx1.ravel()), x_train.shape[1]))
        x_eval *= .5*(problem.xu-problem.xl)+problem.xl
        x_eval[:, 0] = xx1.ravel()
        x_eval[:, 1] = xx2.ravel()
        out_plot = problem.evaluate(x_eval, return_as_dictionary=True)
        is_failed_ref = ArchOptProblemBase.get_failed_points(out_plot)
        pov_ref = (1-is_failed_ref.astype(float)).reshape(xx1.shape)

        def _plot_sfc(z, z_name, path_post, is_g=False):
            plt.figure()
            plt.title(f'SBO model for {problem.__class__.__name__}\n'
                      f'{len(total_pop)} points, {n_fail} failed ({100*n_fail/len(total_pop):.0f}%)')
            zz = z.reshape(xx1.shape)
            c = plt.contourf(xx1, xx2, zz, 50, cmap='RdBu_r' if is_g else 'viridis',
                             norm=CenteredNorm() if is_g else None)
            plt.colorbar(c).set_label(z_name)
            if is_g:
                plt.contour(xx1, xx2, zz, [0], linewidths=2, colors='k')
            plt.contour(xx1, xx2, pov_ref, [.5], linewidths=.5, colors='r')
            plt.scatter(x_train[is_failed_train, 0], x_train[is_failed_train, 1], s=25, c='r', marker='x')
            plt.scatter(x_train[~is_failed_train, 0], x_train[~is_failed_train, 1], s=25, color=(0, 1, 0), marker='x')
            if x_infill is not None:
                plt.scatter([x_infill[0]], [x_infill[1]], s=50, c='b', marker='x')
            plt.xlabel('$x_0$'), plt.ylabel('$x_1$')
            if save_path is not None:
                plt.savefig(f'{save_path}_{path_post}.png')

        x_eval_norm = self.normalization.forward(x_eval)
        y_predicted = self.surrogate_model.predict_values(x_eval_norm)
        y_predicted_std = np.sqrt(self.surrogate_model.predict_variances(x_eval_norm))
        y_names = [f'f{i}' for i in range(problem.n_obj)]+[f'g{i}' for i in range(problem.n_ieq_constr)]
        for iy in range(y_predicted.shape[1]):
            for do_plot_std in [False, True]:
                if do_plot_std and not plot_std:
                    continue
                _plot_sfc((y_predicted_std if do_plot_std else y_predicted)[:, iy],
                          f'{y_names[iy]}{" std dev" if do_plot_std else ""}', f'y{iy}{"_std" if do_plot_std else ""}',
                          is_g=iy >= problem.n_obj)

        infill = self.infill
        f_infill, g_infill = infill.evaluate(x_eval_norm)
        if self.hc_strategy.adds_infill_constraint():
            g_hc = self.hc_strategy.evaluate_infill_constraint(x_eval_norm)
            g_infill = np.column_stack([g_infill, g_hc])
        for i in range(f_infill.shape[1]):
            _plot_sfc(f_infill[:, i], f'Infill $f_{i}$', f'infill_f{i}')
        for i in range(g_infill.shape[1]):
            _plot_sfc(g_infill[:, i], f'Infill $g_{i}$', f'infill_g{i}', is_g=True)

        if show:
            plt.show()
        plt.close('all')


class HCInfill(SurrogateInfill):
    """Infill that wraps another infill and modifies it for dealing with hidden constraints"""

    def __init__(self, infill: SurrogateInfill, hc_strategy: HiddenConstraintStrategy):
        self._infill = infill
        self._hc_strategy = hc_strategy
        super().__init__()

    @property
    def needs_variance(self):
        return self._infill.needs_variance

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        self._infill.set_samples(x_train, y_train)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._infill.predict(x)

    def _initialize(self):
        self._infill.initialize(self.problem, self.surrogate_model, self.normalization)

    def select_infill_solutions(self, population, infill_problem, n_infill):
        return self._infill.select_infill_solutions(population, infill_problem, n_infill)

    def reset_infill_log(self):
        super().reset_infill_log()
        self._infill.reset_infill_log()

    def predict_variance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._infill.predict_variance(x)

    def get_n_infill_objectives(self) -> int:
        return self._infill.get_n_infill_objectives()

    def get_n_infill_constraints(self) -> int:
        n_constr = self._infill.get_n_infill_constraints()
        if self._hc_strategy.adds_infill_constraint():
            n_constr += 1
        return n_constr

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f_infill, g_infill = self._infill.evaluate(x)
        f_infill = self._hc_strategy.mod_infill_objectives(x, f_infill)

        if self._hc_strategy.adds_infill_constraint():
            g_hc = self._hc_strategy.evaluate_infill_constraint(x)
            g_infill = np.column_stack([g_infill, g_hc]) if g_infill is not None else np.array([g_hc]).T

        return f_infill, g_infill
