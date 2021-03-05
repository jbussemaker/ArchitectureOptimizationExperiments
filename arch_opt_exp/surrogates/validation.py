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
import matplotlib.pyplot as plt
from pymoo.model.repair import Repair
from pymoo.model.problem import Problem
from pymoo.model.initialization import Initialization
from arch_opt_exp.surrogates.model import SurrogateModel
from pymoo.model.duplicate import DefaultDuplicateElimination
from arch_opt_exp.problems.discretization import MixedIntProblemHelper
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

__all__ = ['LOOCrossValidation']


class LOOCrossValidation:
    """Leave-one-out cross validation of a surrogate model: trains a surrogate model k-times with each time one random
    sample left out, and then computing the root mean square error (RMSE) of each of the training round errors."""

    @classmethod
    def check_sample_sizes(cls, surrogate_model: SurrogateModel, problem: Problem, n_train=10, n_pts_test=None,
                           n_repeat=5, repair: Repair = None, print_progress=False, show=True):
        if n_pts_test is None:
            n_pts_test = [10, 15, 20, 50, 75, 100]

        is_int_mask = MixedIntProblemHelper.get_is_int_mask(problem)
        is_cat_mask = MixedIntProblemHelper.get_is_cat_mask(problem)

        if repair is None:
            repair = MixedIntProblemHelper.get_repair(problem)

        init_sampling = Initialization(LatinHypercubeSampling(), repair=repair,
                                       eliminate_duplicates=DefaultDuplicateElimination())
        scores = []
        scores_std = []
        for i, n_pts in enumerate(n_pts_test):
            if print_progress:
                print('LOOCV set for %d pts (%d/%d)' % (n_pts, i+1, len(n_pts_test)))

            xt_test = init_sampling.do(problem, n_pts).get('X')
            yt_test = problem.evaluate(xt_test)

            is_active, xt_test = MixedIntProblemHelper.is_active(problem, xt_test)
            xt_test = MixedIntProblemHelper.normalize(problem, xt_test)

            run_scores = []
            for j in range(n_repeat):
                if print_progress:
                    print('LOOCV x %d (%d/%d)' % (n_train, j+1, n_repeat))
                run_scores.append(cls.cross_validate(
                    surrogate_model, xt_test, yt_test, n_train=n_train, is_int_mask=is_int_mask,
                    is_cat_mask=is_cat_mask, is_active=is_active))
            scores.append(np.mean(run_scores))
            scores_std.append(np.std(run_scores))

        scores, scores_std = np.array(scores), np.array(scores_std)

        plt.figure(), plt.title('LOO-CV')
        plt.semilogy(n_pts_test, scores, '-xk', linewidth=1)
        plt.plot(n_pts_test, scores+scores_std, '--k', linewidth=1)
        plt.plot(n_pts_test, scores-scores_std, '--k', linewidth=1)
        plt.xlabel('Number of training points'), plt.ylabel('LOO-CV score')

        if show:
            plt.show()

    @classmethod
    def cross_validate(cls, surrogate_model: SurrogateModel, xt: np.ndarray, yt: np.ndarray, n_train: int = None,
                       is_int_mask: np.ndarray = None, is_cat_mask: np.ndarray = None, is_active: np.ndarray = None) \
            -> np.ndarray:
        if n_train is None:
            n_train = xt.shape[0]
        if n_train > xt.shape[0]:
            n_train = xt.shape[0]

        i_leave_out = np.random.choice(xt.shape[0], n_train, replace=False)
        errors = np.empty((n_train, yt.shape[1]))
        for i, i_lo in enumerate(i_leave_out):
            errors[i, :] = cls._get_error(
                surrogate_model, xt, yt, i_lo, is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, is_active=is_active)

        rmse = np.sqrt(np.mean(errors**2, axis=0))
        return rmse

    @classmethod
    def _get_error(cls, surrogate_model: SurrogateModel, xt: np.ndarray, yt: np.ndarray, i_leave_out,
                   is_int_mask: np.ndarray = None, is_cat_mask: np.ndarray = None, is_active: np.ndarray = None) \
            -> np.ndarray:
        x_lo = xt[i_leave_out, :]
        y_lo = yt[i_leave_out, :]
        is_active_lo = is_active[i_leave_out, :] if is_active is not None else None
        xt = np.delete(xt, i_leave_out, axis=0)
        yt = np.delete(yt, i_leave_out, axis=0)

        surrogate_model_copy = cls._copy_surrogate_model(surrogate_model)
        surrogate_model_copy.set_samples(xt, yt, is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, is_active=is_active)
        surrogate_model_copy.train()

        y_lo_predict = surrogate_model_copy.predict(np.atleast_2d(x_lo), is_active=is_active_lo)
        return y_lo_predict-y_lo

    @classmethod
    def _copy_surrogate_model(cls, surrogate_model: SurrogateModel) -> SurrogateModel:
        return surrogate_model.copy()


if __name__ == '__main__':
    from pymoo.problems.single.himmelblau import Himmelblau
    from arch_opt_exp.surrogates.smt_models.smt_krg import SMTKrigingSurrogateModel
    LOOCrossValidation.check_sample_sizes(SMTKrigingSurrogateModel(theta0=1e-1), Himmelblau(), print_progress=True)
