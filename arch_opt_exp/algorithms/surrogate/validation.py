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
from arch_opt_exp.metrics_base import Metric
from arch_opt_exp.surrogates.validation import *
from arch_opt_exp.algorithms.infill_based import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

from pymoo.model.algorithm import Algorithm

__all__ = ['SurrogateQualityMetric']


class SurrogateQualityMetric(Metric):
    """
    Outputs several surrogate model quality metrics:
    - RMSE: root mean square error
    - AME: absolute max error
    - MAE: mean absolute error
    - R2: R-squared value
    - LOOCV (optional): leave-one-out cross-validation rmse
    """

    def __init__(self, include_loo_cv=False, n_loo_cv: int = None):
        self.include_loo_cv = include_loo_cv
        self.n_loo_cv = n_loo_cv or 50
        super(SurrogateQualityMetric, self).__init__()

    @property
    def name(self) -> str:
        return 'sm_quality'

    @property
    def value_names(self) -> List[str]:
        names = ['rmse', 'ame', 'mae', 'r2']
        if self.include_loo_cv:
            names += ['loo_cv']
        return names

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        surrogate_infill = self._get_surrogate_infill(algorithm)
        if surrogate_infill is None:
            return [np.nan]*len(self.value_names)

        surrogate_model = surrogate_infill.surrogate_model
        if surrogate_model is None:
            return [np.nan]*len(self.value_names)

        xt, yt = surrogate_infill.x_train, surrogate_infill.y_train
        is_int_mask, is_cat_mask = surrogate_infill.is_int_mask, surrogate_infill.is_cat_mask

        y_err = yt-surrogate_model.predict(xt)

        values = [
            np.max(self._get_rmse(y_err)),
            np.max(self._get_ame(y_err)),
            np.max(self._get_mae(y_err)),
            np.min(self._get_r2(y_err, yt)),
        ]
        if self.include_loo_cv:
            values += [np.max(LOOCrossValidation.cross_validate(
                surrogate_model, xt, yt, n_train=self.n_loo_cv, is_int_mask=is_int_mask, is_cat_mask=is_cat_mask))]

        return values

    @staticmethod
    def _get_rmse(y_error: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(y_error**2, axis=0))

    @staticmethod
    def _get_ame(y_error: np.ndarray) -> np.ndarray:
        return np.max(np.abs(y_error), axis=0)

    @staticmethod
    def _get_mae(y_error: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(y_error), axis=0)

    @staticmethod
    def _get_r2(y_error: np.ndarray, yt: np.ndarray) -> np.ndarray:
        return 1-np.sum(y_error**2, axis=0)/np.sum((yt-np.mean(yt, axis=0))**2, axis=0)

    @staticmethod
    def _get_surrogate_infill(algorithm: Algorithm) -> Optional[SurrogateInfill]:
        if isinstance(algorithm, InfillBasedAlgorithm) and isinstance(algorithm.infill, SurrogateBasedInfill):
            return algorithm.infill.infill
