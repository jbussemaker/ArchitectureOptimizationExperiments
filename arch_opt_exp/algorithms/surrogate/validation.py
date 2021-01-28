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
from arch_opt_exp.metrics_base import Metric
from arch_opt_exp.algorithms.infill_based import *
from smt.surrogate_models.surrogate_model import SurrogateModel
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

from pymoo.model.algorithm import Algorithm

__all__ = ['SurrogateQualityMetric', 'LOOCrossValidation']


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

        y_err = yt-surrogate_model.predict_values(xt)

        values = [
            np.max(self._get_rmse(y_err)),
            np.max(self._get_ame(y_err)),
            np.max(self._get_mae(y_err)),
            np.min(self._get_r2(y_err, yt)),
        ]
        if self.include_loo_cv:
            values += [np.max(LOOCrossValidation.cross_validate(surrogate_model, xt, yt, n_train=self.n_loo_cv))]

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


class LOOCrossValidation:
    """Leave-one-out cross validation of a surrogate model: trains a surrogate model k-times with each time one random
    sample left out, and then computing the root mean square error (RMSE) of each of the training round errors."""

    @classmethod
    def cross_validate(cls, surrogate_model: SurrogateModel, xt: np.ndarray, yt: np.ndarray, n_train: int = None) \
            -> np.ndarray:
        if n_train is None:
            n_train = xt.shape[0]
        if n_train > xt.shape[0]:
            n_train = xt.shape[0]

        i_leave_out = np.random.choice(xt.shape[0], n_train, replace=False)
        errors = np.empty((n_train, yt.shape[1]))
        for i, i_lo in enumerate(i_leave_out):
            errors[i, :] = cls._get_error(surrogate_model, xt, yt, i_lo)

        rmse = np.sqrt(np.mean(errors**2, axis=0))
        return rmse

    @classmethod
    def _get_error(cls, surrogate_model: SurrogateModel, xt: np.ndarray, yt: np.ndarray, i_leave_out) -> np.ndarray:
        x_lo = xt[i_leave_out, :]
        y_lo = yt[i_leave_out, :]
        xt = np.delete(xt, i_leave_out, axis=0)
        yt = np.delete(yt, i_leave_out, axis=0)

        surrogate_model_copy = cls._copy_surrogate_model(surrogate_model)
        surrogate_model_copy.options['print_global'] = False
        surrogate_model_copy.set_training_values(xt, yt)
        surrogate_model_copy.train()

        y_lo_predict = surrogate_model_copy.predict_values(np.atleast_2d(x_lo))
        return y_lo_predict-y_lo

    @classmethod
    def _copy_surrogate_model(cls, surrogate_model: SurrogateModel) -> SurrogateModel:
        return SurrogateModelFactory.copy_surrogate_model(surrogate_model)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from smt.surrogate_models.krg import KRG
    from smt.sampling_methods.lhs import LHS
    from pymoo.problems.single.himmelblau import Himmelblau

    sm = KRG(theta0=[1e-1]*2)
    prob = Himmelblau()
    n_loo_cv_pts = 10
    n_pts_test = [10, 15, 20, 50, 75, 100]
    n_loo_cv_repeat = 5

    lhs = LHS(xlimits=np.array([[0, 1]]*2))
    scores = []
    scores_std = []
    for n_pts in n_pts_test:
        xt_test = lhs(n_pts)
        yt_test = prob.evaluate(xt_test)

        run_scores = [LOOCrossValidation.cross_validate(sm, xt_test, yt_test, n_train=n_loo_cv_pts)
                      for _ in range(n_loo_cv_repeat)]
        scores.append(np.mean(run_scores))
        scores_std.append(np.std(run_scores))

    scores, scores_std = np.array(scores), np.array(scores_std)

    plt.figure(), plt.title('LOO-CV')
    plt.semilogy(n_pts_test, scores, '-xk', linewidth=1)
    plt.plot(n_pts_test, scores+scores_std, '--k', linewidth=1)
    plt.plot(n_pts_test, scores-scores_std, '--k', linewidth=1)
    plt.xlabel('Number of training points'), plt.ylabel('LOO-CV score')
    plt.show()
