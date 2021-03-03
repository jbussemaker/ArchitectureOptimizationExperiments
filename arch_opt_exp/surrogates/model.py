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

import copy
import numpy as np

__all__ = ['SurrogateModel', 'SurrogateModelFactory']


class SurrogateModel:
    """
    Base class for the surrogate model as used in this package. Should be pickleable.
    """

    def copy(self) -> 'SurrogateModel':
        """Return an uninitialized copy of the surrogate model."""
        return copy.deepcopy(self)

    @staticmethod
    def _get_mask(x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        return np.zeros((x.shape[1],), dtype=bool) if mask is None else mask

    def set_samples(self, x: np.ndarray, y: np.ndarray, is_int_mask: np.ndarray = None, is_cat_mask: np.ndarray = None):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def supports_variance(self) -> bool:
        raise NotImplementedError

    def predict_variance(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SurrogateModelFactory:

    def __init__(self, surrogate_model: SurrogateModel):
        self.base: SurrogateModel = surrogate_model.copy()

    def get(self):
        return self.base.copy()
