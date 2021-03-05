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

import pickle
import logging
import warnings
import numpy as np
from typing import *
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Kernel
from arch_opt_exp.surrogates.model import SurrogateModel
from sklearn.gaussian_process import GaussianProcessRegressor
from arch_opt_exp.surrogates.sklearn_models.distance_base import *
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

__all__ = ['SKLearnGPSurrogateModel']

log = logging.getLogger('arch_opt_exp.sklearn_gp')

# Do not display ConvergeWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class SKLearnGPSurrogateModel(SurrogateModel):
    """
    Adapter for scikit-learn Gaussian process surrogate models. Support for mixed-integer variables.

    Either treats integer variables as discrete variables or not (continuous relaxation).
    """

    _exclude = ['__models', '_samples', '_ny']

    def __init__(self, kernel: Kernel = None, alpha: float = None, int_as_discrete=False):
        if kernel is None:
            kernel = ConstantKernel(1.)*Matern(1., nu=1.5)
        self.kernel = kernel

        self.alpha = alpha or 1e-10
        self.int_as_discrete = int_as_discrete

        self.alpha_multiply = np.sqrt(10.)
        self.alpha_multiply_try = 20

        self.__models = None
        self._samples = None
        self._is_active = None
        self._is_int_mask = None
        self._is_cat_mask = None
        self._ny = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def _models(self):
        if self.__models is None and self._ny is not None:
            self.__models = [GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha) for _ in range(self._ny)]
        return self.__models

    def set_samples(self, x: np.ndarray, y: np.ndarray, is_int_mask: np.ndarray = None, is_cat_mask: np.ndarray = None,
                    is_active: np.ndarray = None):
        self._samples = (x, y)
        self._is_active = is_active
        self._is_int_mask = self._get_mask(x, is_int_mask)
        self._is_cat_mask = self._get_mask(x, is_cat_mask)
        self._ny = y.shape[1]

        if isinstance(self.kernel, (MixedIntKernel, CustomDistanceKernel)):
            is_discrete_mask = np.bitwise_or(self._is_int_mask, self._is_cat_mask) \
                if self.int_as_discrete else self._is_cat_mask
            self.kernel.set_discrete_mask(is_discrete_mask)
            self.kernel.set_samples(x, y, self._is_int_mask, self._is_cat_mask, is_active=is_active)

    def train(self):
        if self._samples is None:
            raise ValueError('Samples not set')
        x, y = self._samples

        # Reset alpha
        for i in range(self._ny):
            self._models[i].alpha = self.alpha

        # Dynamically increase alpha until fitting is successful
        n_try = self.alpha_multiply_try
        for i in range(self._ny):
            for j in range(n_try):
                try:
                    self._models[i].fit(x, y[:, i])
                    break

                except np.linalg.LinAlgError:
                    # Increasing alpha (noise factor) to be able to handle ill-conditioned kernel matrices better
                    if j < n_try-1:
                        self._models[i].alpha *= self.alpha_multiply
                        log.info('Try %d: alpha[%d] increased to %.2g' % (j, i, self._models[i].alpha))

                    else:
                        # Dump the training values and raise exception
                        log.warning('Cannot find alpha for training the surrogate model (%d tries)' % n_try)
                        with open('dump.pkl', 'wb') as fp:
                            pickle.dump((x, y[:, i]), fp)

                        raise

    def predict(self, x: np.ndarray, is_active: np.ndarray = None) -> np.ndarray:
        self._predict_set_is_active(x, is_active=is_active)

        y = np.empty((x.shape[0], self._ny))
        for i in range(self._ny):
            y[:, i] = self._models[i].predict(x)
        return y

    def supports_variance(self) -> bool:
        return True

    def predict_variance(self, x: np.ndarray, is_active: np.ndarray = None) -> np.ndarray:
        self._predict_set_is_active(x, is_active=is_active)

        y_std = np.empty((x.shape[0], self._ny))
        for i in range(self._ny):
            _, y_std_i = self._models[i].predict(x, return_std=True)
            y_std[:, i] = y_std_i

        # Standard deviation is square root of variance, however it seems that y_std already represents the variance
        y_std[y_std < 0.] = 0.
        return y_std

    def _predict_set_is_active(self, x: np.ndarray, is_active: np.ndarray = None):
        if is_active is None:
            is_active = np.ones(x.shape, dtype=bool)
        if isinstance(self.kernel, (MixedIntKernel, CustomDistanceKernel)):
            self.kernel.predict_set_is_active(is_active)
