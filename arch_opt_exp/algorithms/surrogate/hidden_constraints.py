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
from pymoo.model.problem import Problem
from arch_opt_exp.surrogates.model import *

__all__ = ['HCStrategy', 'MaxValueHCStrategy', 'AvoidHCStrategy', 'PredictHCStrategy']


class HCStrategy:
    """
    Base class for dealing with hidden constraints in surrogate-based optimization.
    """

    def get_training_set(self, x, is_active, f, g) -> Tuple[np.ndarray, ...]:
        """Modify the (normalized) objectives and constraint values for training. Returns x, is_active, f, g"""
        raise NotImplementedError

    def get_n_g_infill(self) -> int:
        """Number of additional constraints to add to the infill optimization problem."""
        raise NotImplementedError

    def get_f_g_infill(self, x, f, g, problem: Problem, surrogate_model: SurrogateModel) -> Tuple[np.ndarray, ...]:
        """Modify infill problem and objectives."""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class MaxValueHCStrategy(HCStrategy):
    """Replace invalid objective and constraint values with maximum of the valid values."""

    def get_training_set(self, x, is_active, f, g) -> Tuple[np.ndarray, ...]:
        f_is_nan = np.any(np.isnan(f), axis=1)
        f[f_is_nan] = np.nanmax(f, axis=0)

        g_is_nan = np.any(np.isnan(g), axis=1)
        g[g_is_nan] = 1.

        return x, is_active, f, g

    def get_n_g_infill(self) -> int:
        return 0

    def get_f_g_infill(self, x, f, g, problem: Problem, surrogate_model: SurrogateModel) -> Tuple[np.ndarray, ...]:
        return f, g

    def __repr__(self):
        return '%s()' % self.__class__.__name__


class SeparateInfeasibleHCStrategy(HCStrategy):
    """Base class for HC strategy that removes infeasible points from the training set."""

    def __init__(self):
        super(SeparateInfeasibleHCStrategy, self).__init__()
        self._x_hc = None

    def get_training_set(self, x, is_active, f, g) -> Tuple[np.ndarray, ...]:
        is_nan = np.any(np.isnan(f), axis=1) | np.any(np.isnan(g), axis=1)
        self._x_hc = x[is_nan, :]

        keep = ~is_nan
        return x[keep, :], is_active[keep, :], f[keep, :], g[keep, :]

    def get_n_g_infill(self) -> int:
        raise NotImplementedError

    def get_f_g_infill(self, x, f, g, problem: Problem, surrogate_model: SurrogateModel) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class AvoidHCStrategy(SeparateInfeasibleHCStrategy):
    """
    Avoid hidden constraint areas by keeping a minimum distance to infeasible points. Inspired by:
    https://www.datadvance.net/blog/tech-tips/2021/handling-non-numerical-values-in-the-design-space-exploration-and-predictive-modeling-studies.html
    """

    def __init__(self, d_avoid=.05):
        super(AvoidHCStrategy, self).__init__()
        self.d_avoid = d_avoid

    def get_n_g_infill(self) -> int:
        return 1

    def get_f_g_infill(self, x, f, g, problem: Problem, surrogate_model: SurrogateModel) -> Tuple[np.ndarray, ...]:
        g_avoid_nan = self._evaluate_g_avoid_hc(x, problem)
        g = np.column_stack([g, g_avoid_nan])
        return f, g

    def _evaluate_g_avoid_hc(self, x, problem: Problem) -> np.ndarray:
        if self._x_hc.shape[0] == 0:
            return np.zeros((x.shape[0],))

        x_norm = problem.xu-problem.xl
        d_hc = np.empty((x.shape[0],))
        for i in range(x.shape[0]):
            d_hc[i] = np.min(np.sqrt(np.sum(((self._x_hc - x[i, :]) / x_norm) ** 2, axis=1)))

        return self.d_avoid-d_hc

    def __repr__(self):
        return '%s(d_avoid=%r)' % (self.__class__.__name__, self.d_avoid)


class PredictHCStrategy(SeparateInfeasibleHCStrategy):
    """
    Trains a second predictor to try to avoid areas with hidden constraints. Inspired by:
    Lee et al., "Optimization Subject to Hidden Constraints via Statistical Emulation", 2010
    """

    def __init__(self, sm: SurrogateModel, hc_predict_max=.5):
        super(PredictHCStrategy, self).__init__()
        self._sm = sm.copy()
        self._hc_max = hc_predict_max

        self._sm_hc: Optional[SurrogateModel] = None

    def get_training_set(self, x, is_active, f, g) -> Tuple[np.ndarray, ...]:
        x, is_active, f, g = super(PredictHCStrategy, self).get_training_set(x, is_active, f, g)

        if self._x_hc.shape[0] == 0:
            self._sm_hc = None
        else:
            x_is_hc = np.row_stack([x, self._x_hc])
            y_is_hc = np.append(np.ones((x.shape[0],))*-.5, np.ones((self._x_hc.shape[0],))*.5)[:, None]

            self._sm_hc = sm_hc = self._sm.copy()
            sm_hc.set_samples(x_is_hc, y_is_hc)
            sm_hc.train()

        return x, is_active, f, g

    def get_n_g_infill(self) -> int:
        return 1

    def get_f_g_infill(self, x, f, g, problem: Problem, surrogate_model: SurrogateModel) -> Tuple[np.ndarray, ...]:
        if self._sm_hc is None:
            g_hc = np.zeros((x.shape[0], 1))
        else:
            g_hc = self._sm_hc.predict(x)+.5-self._hc_max

        g = np.column_stack([g, g_hc])
        return f, g

    def __repr__(self):
        return '%s(%r, hc_predict_max=%r)' % (self.__class__.__name__, self._sm, self._hc_max)
