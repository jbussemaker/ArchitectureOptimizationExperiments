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
from arch_opt_exp.surrogates import SurrogateModel
from smt.applications.mixed_integer import MixedIntegerSurrogateModel, FLOAT, INT, ENUM
from smt.surrogate_models.surrogate_model import SurrogateModel as SMTSurrogateModelBase


class SMTSurrogateModel(SurrogateModel):
    """
    Adapter for SMT surrogate models.

    If the surrogate model is to be automatically wrapped in the mixed-integer application, it is assumed that the
    input variables are normalized according to the logic of MixedIntBaseProblem: continuous between 0 and 1, discrete
    between 0 and n-1. The mixed-integer approach combines continuous relaxation for integer variables and dummy
    coding for categorical variables.

    SMT surrogate models do not take design variable hierarchy into account.
    """

    _exclude = ['_smt', '_xt_last', '_is_int_mask', '_is_cat_mask']

    def __init__(self, auto_wrap_mixed_int=False):
        self.auto_wrap_mi = auto_wrap_mixed_int

        self._smt: Optional[SMTSurrogateModelBase] = None
        self._xt_last = None
        self._is_int_mask = None
        self._is_cat_mask = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    def set_samples(self, x: np.ndarray, y: np.ndarray, is_int_mask: np.ndarray = None, is_cat_mask: np.ndarray = None,
                    is_active: np.ndarray = None):
        self._xt_last = x
        self._is_int_mask = is_int_mask = self._get_mask(x, is_int_mask)
        self._is_cat_mask = is_cat_mask = self._get_mask(x, is_cat_mask)

        self._smt = surrogate_model = self._create_surrogate_model()

        # Automatically wrap the surrogate model in the MixedIntegerSurrogateModel if there are discrete variables
        if self.auto_wrap_mi and np.any(np.bitwise_or(is_int_mask, is_cat_mask)):
            x_types, x_limits = [], []
            for i_x in range(x.shape[1]):
                if is_int_mask[i_x]:
                    x_types += [INT]
                    x_limits += [[0, int(np.max(x[:, i_x]))]]

                elif is_cat_mask[i_x]:
                    n = int(np.max(x[:, i_x])+1)
                    if n == 1:
                        n = 2
                    x_types += [(ENUM, n)]
                    x_limits += [list(range(n))]

                else:
                    x_types += [FLOAT]
                    x_limits += [[0., 1.]]

            self._smt = MixedIntegerSurrogateModel(x_types, x_limits, surrogate_model, input_in_folded_space=True)

        self._smt.set_training_values(x, y)

    def train(self):
        self._smt.train()

    def predict(self, x: np.ndarray, is_active: np.ndarray = None) -> np.ndarray:
        try:
            return self._smt.predict_values(x)

        except FloatingPointError:
            return np.zeros((x.shape[0], self._smt.ny))*np.nan

    def supports_variance(self) -> bool:
        smt = self._smt
        if smt is None:
            smt = self._create_surrogate_model()
        return smt.supports['variances']

    def predict_variance(self, x: np.ndarray, is_active: np.ndarray = None) -> np.ndarray:
        # There is a bug in SMT preventing a vectorized evaluation:
        # https://github.com/SMTorg/smt/issues/160

        try:
            variance = np.zeros((x.shape[0], self._smt.ny))
            for i in range(x.shape[0]):
                variance[i, :] = self._smt.predict_variances(x[[i], :])
            return variance

        except FloatingPointError:
            return np.zeros((x.shape[0], self._smt.ny))*np.nan

    def _create_surrogate_model(self) -> SMTSurrogateModelBase:
        raise NotImplementedError
