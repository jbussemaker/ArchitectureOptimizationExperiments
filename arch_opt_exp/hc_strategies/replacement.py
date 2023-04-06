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
from scipy.spatial import distance
from smt.surrogate_models.krg import KRG
from arch_opt_exp.hc_strategies.sbo_with_hc import *

__all__ = ['ReplacementHCStrategyBase', 'GlobalWorstReplacement', 'LocalReplacement', 'PredictedWorstReplacement']


class ReplacementHCStrategyBase(HiddenConstraintStrategy):
    """Base class for a strategy that replaces failed outputs by some value"""

    def mod_xy_train(self, x_norm: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Separate into failed and valid (non-failed) set
        is_failed = self.is_failed(y)
        x_valid = x_norm[~is_failed, :]
        y_valid = y[~is_failed, :]
        x_failed = x_norm[is_failed, :]
        y_failed = y[is_failed, :]

        # If there are no failed points, no need to replace
        if x_failed.shape[0] == 0:
            return x_norm, y

        # If there are no valid points, replace with 1
        if y_valid.shape[0] == 0:
            y_failed_replace = np.ones(y_failed.shape)
        else:
            y_failed_replace = self._replace_y(x_failed, y_failed, x_valid, y_valid)

        # Replace values
        y = y.copy()
        y[is_failed, :] = y_failed_replace
        return x_norm, y

    def _replace_y(self, x_failed: np.ndarray, y_failed: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) \
            -> np.ndarray:
        """Return values for replacing y_failed (x values are normalized)"""
        raise NotImplementedError

    def get_replacement_strategy_name(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return f'Replacement: {self.get_replacement_strategy_name()}'

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class GlobalWorstReplacement(ReplacementHCStrategyBase):
    """Replace failed values with the worst values known for these outputs"""

    def _replace_y(self, x_failed: np.ndarray, y_failed: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) \
            -> np.ndarray:
        # Get global worst values
        y_worst = np.max(y_valid, axis=0)

        # Replace
        y_replace = np.zeros(y_failed.shape)+y_worst
        return y_replace

    def get_replacement_strategy_name(self) -> str:
        return 'Global Worst'


class LocalReplacement(ReplacementHCStrategyBase):
    """Replace failed values with the worst or mean values of the closest n valid point"""

    def __init__(self, n: int = 1, mean=False):
        self.n = n
        self.mean = mean
        super().__init__()

    def _replace_y(self, x_failed: np.ndarray, y_failed: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) \
            -> np.ndarray:
        # Get distances from failed points to valid points
        n, mean = self.n, self.mean
        x_dist = distance.cdist(x_failed, x_valid)

        def _agg(y_):
            if mean:
                return np.mean(y_, axis=0)
            return np.max(y_, axis=0)

        # Replace values with max or mean of closest n points
        y_replace = np.zeros(y_failed.shape)
        for ix in range(x_failed.shape[0]):
            ix_min = np.argsort(x_dist[ix, :])[:n]
            y_replace[ix, :] = _agg(y_valid[ix_min, :])

        return y_replace

    def get_replacement_strategy_name(self) -> str:
        operator = 'Mean' if self.mean else 'Worst'
        if self.n != 1:
            return f'Local {self.n} {operator}'
        return f'Local {operator}'

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self.n}, mean={self.mean!r})'


class PredictedWorstReplacement(ReplacementHCStrategyBase):
    """Replace failed values by the worst predicted value of a model trained on the valid points"""

    def __init__(self, mul: float = 1.):
        self.mul = mul
        super().__init__()

    def _replace_y(self, x_failed: np.ndarray, y_failed: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) \
            -> np.ndarray:
        # Normalize valid y
        y_min, y_max = np.min(y_valid, axis=0), np.max(y_valid, axis=0)
        norm = y_max-y_min
        norm[norm < 1e-6] = 1e-6
        y_norm = (y_valid - y_min) / norm

        # Train Kriging surrogate model
        model = KRG(print_global=False)
        model.set_training_values(x_valid, y_norm)
        model.train()

        # Predict values of failed points
        y_predict = model.predict_values(x_failed)
        y_predict_var = model.predict_variances(x_failed)

        # Replace failed points with mean + sigma*var of the prediction
        y_replace = y_predict + self.mul * np.sqrt(y_predict_var)
        y_replace = y_replace*norm + y_min

        return y_replace

    def get_replacement_strategy_name(self) -> str:
        sigma_str = '' if self.mul == 1. else f' (mul = {self.mul:.1f})'
        return f'Predicted Worst{sigma_str}'

    def __repr__(self):
        return f'{self.__class__.__name__}(mul={self.mul:.2f})'
