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
import matplotlib.pyplot as plt
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

__all__ = ['ModulatedMOInfill']


class ModulatedMOInfill(SurrogateInfill):
    """
    Multi-objective infill criteria themselves might be represented by a single-objective design space: a criteria
    representing the improvement of the Pareto front might have the same value at different points at the Pareto front.
    To help with exploration, the single-objective MO infill criterion might be turned into a multi-objective criterion
    by modulating the criterion for all original objective values. This helps in finding multiple infill points along
    the current Pareto front at each infill iteration.
    """

    def __init__(self, underlying: SurrogateInfill):
        super(ModulatedMOInfill, self).__init__()

        self.underlying = underlying

    @property
    def needs_variance(self):
        return self.underlying.needs_variance

    def initialize(self, *args, **kwargs):
        super(ModulatedMOInfill, self).initialize(*args, **kwargs)
        self.underlying.initialize(*args, **kwargs)

    def set_training_values(self, x_train: np.ndarray, y_train: np.ndarray):
        super(ModulatedMOInfill, self).set_training_values(x_train, y_train)
        self.underlying.set_training_values(x_train, y_train)

    def get_n_infill_objectives(self) -> int:
        return self.underlying.get_n_infill_objectives()*self.problem.n_obj

    def get_n_infill_constraints(self) -> int:
        return self.underlying.get_n_infill_constraints()

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        f_underlying, g = self.underlying.evaluate(x)
        f_predicted, _ = self.predict(x)

        f_modulated = self._get_f_modulated(f_predicted, f_underlying)
        return f_modulated, g

    @staticmethod
    def _get_f_modulated(f_predicted: np.ndarray, f_underlying: np.ndarray) -> np.ndarray:
        n_f = f_predicted.shape[1]

        f_modulated = np.empty((f_predicted.shape[0], n_f*f_underlying.shape[1]))
        for i_f_underlying in range(f_underlying.shape[1]):
            f_modulated[:, i_f_underlying*n_f:i_f_underlying*n_f+n_f] = f_underlying[:, [i_f_underlying]]*f_predicted

        return f_modulated

    @classmethod
    def plot(cls, var=None, n_pareto=5, show=True, **kwargs):
        f_pareto = np.zeros((n_pareto, 2))
        f_pareto[:, 0] = (1.-np.cos(.5*np.pi*np.linspace(0, 1, n_pareto+2)[1:-1]))**.8
        f_pareto[:, 1] = (1.-np.cos(.5*np.pi*(1-np.linspace(0, 1, n_pareto+2)[1:-1])))**.8

        if np.isscalar(var):
            var = [var, var]
        if var is None:
            var = [.1, .1]

        n = 25
        x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        f_eval = np.column_stack([x.ravel(), y.ravel()])
        f_var = np.ones(f_eval.shape)*var

        f_underlying = cls._get_plot_f_underlying(f_eval, f_var, f_pareto, **kwargs)

        for i_f in range(f_underlying.shape[1]):
            plt.figure()
            c = plt.contourf(x, y, f_underlying[:, i_f].reshape(x.shape), 50, cmap='Blues_r')
            for edge in c.collections:  # https://github.com/matplotlib/matplotlib/issues/4419#issuecomment-101253070
                edge.set_edgecolor('face')
            plt.contour(x, y, f_underlying[:, i_f].reshape(x.shape), 5, colors='k', alpha=.5)

            plt.colorbar(c).set_label('Criterion $f$')
            plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=20, c='w', edgecolors='k')
            plt.ylim([0, 1]), plt.xlim([0, 1])
            plt.xlabel('$f_0$'), plt.ylabel('$f_1$')

        f_modulated = cls._get_f_modulated(f_eval, f_underlying)
        n_f = f_underlying.shape[1]
        for i_f_mod in range(f_modulated.shape[1]):
            # i_f_underlying = i_f_mod % n_f
            i_f_orig = int(np.floor(i_f_mod/n_f))

            plt.figure()
            c = plt.contourf(x, y, f_modulated[:, i_f_mod].reshape(x.shape), 50, cmap='Blues_r')
            for edge in c.collections:  # https://github.com/matplotlib/matplotlib/issues/4419#issuecomment-101253070
                edge.set_edgecolor('face')
            plt.contour(x, y, f_modulated[:, i_f_mod].reshape(x.shape), 5, colors='k', alpha=.5)

            plt.colorbar(c).set_label('Criterion $f$ modulated over $f_{%d}$' % (i_f_orig,))
            plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=20, c='w', edgecolors='k')
            plt.ylim([0, 1]), plt.xlim([0, 1])
            plt.xlabel('$f_0$'), plt.ylabel('$f_1$')

        if show:
            plt.show()

    @classmethod
    def _get_plot_f_underlying(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        raise RuntimeError('Not implemented')
