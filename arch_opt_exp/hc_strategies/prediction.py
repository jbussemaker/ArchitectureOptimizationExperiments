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
from sb_arch_opt.sampling import *
from arch_opt_exp.hc_strategies.sbo_with_hc import *
from sb_arch_opt.problems.hidden_constraints import *

from smt.surrogate_models.surrogate_model import SurrogateModel

__all__ = ['PredictionHCStrategy', 'PredictorInterface',
           'RandomForestClassifier', 'KNNClassifier', 'GPClassifier', 'SVMClassifier', 'LinearRBFRegressor',
           'GPRegressor', 'VariationalGP']


class PredictorInterface:
    """Interface class for some validity predictor"""
    _training_doe = {}
    _reset_pickle_keys = []

    def __init__(self):
        self.training_set = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._reset_pickle_keys:
            if key in state:
                state[key] = None
        return state

    def get_stats(self, problem: ArchOptProblemBase = None, min_pov=.5, n=50, train=True, plot=True, save_ref=True,
                  save_path=None, show=True):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        if problem is None:
            problem = Alimo()

        if train:
            cache_key = (repr(problem), n)
            if cache_key in self._training_doe:
                x_train = self._training_doe[cache_key]
            else:
                self._training_doe[cache_key] = x_train = HierarchicalRandomSampling().do(problem, n).get('X')
            out_train = problem.evaluate(x_train, return_as_dictionary=True)
            x_norm = (x_train-problem.xl)/(problem.xu-problem.xl)

            is_failed_train = problem.get_failed_points(out_train)
            self.train(x_norm, 1-is_failed_train.astype(float))
        else:
            if self.training_set is None:
                return
            x_norm, is_valid_train = self.training_set
            x_train = x_norm*(problem.xu-problem.xl)+problem.xl
            is_failed_train = (1-is_valid_train).astype(bool)

        x1, x2 = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        xx1, xx2 = np.meshgrid(x1, x2)
        x_eval = .5*np.ones((len(xx1.ravel()), x_train.shape[1]))
        x_eval[:, 0] = xx1.ravel()
        x_eval[:, 1] = xx2.ravel()
        x_eval_norm = x_eval
        x_eval = x_eval*(problem.xu-problem.xl) + problem.xl
        out_plot = problem.evaluate(x_eval, return_as_dictionary=True)
        is_failed_ref = problem.get_failed_points(out_plot)
        pov_ref = (1-is_failed_ref.astype(float)).reshape(xx1.shape)

        pov_predicted = self.evaluate_probability_of_validity(x_eval_norm)
        pov_predicted = pov_predicted_roc = np.clip(pov_predicted, 0, 1)
        pov_predicted = pov_predicted.reshape(xx1.shape)

        # For the ROC curve, ensure we sufficiently cover the design space
        if x_train.shape[1] > 2:
            x_eval_roc_abs = HierarchicalRandomSampling().do(problem, 10000).get('X')
            is_failed_ref = problem.get_failed_points(problem.evaluate(x_eval_roc_abs, return_as_dictionary=True))
            x_eval_roc = (x_eval_roc_abs-problem.xl)/(problem.xu-problem.xl)
            pov_predicted_roc = np.clip(self.evaluate_probability_of_validity(x_eval_roc), 0, 1)

        # Get ROC curve: false positive rate vs true positive rate for various minimum pov's
        # https://en.wikipedia.org/wiki/Receiver_operating_characteristic
        thr_values = np.linspace(0, 1, 201)
        fpr = np.zeros((len(thr_values),))
        tpr = np.zeros((len(thr_values),))
        acc = np.zeros((len(thr_values),))
        is_valid_ref = ~is_failed_ref
        n_pos, n_neg = np.sum(is_valid_ref), np.sum(~is_valid_ref)
        for i, thr_value in enumerate(thr_values):
            predicted = pov_predicted_roc >= thr_value
            tpr[i] = np.sum(predicted & is_valid_ref) / n_pos
            fpr[i] = np.sum(predicted & ~is_valid_ref) / n_neg
            acc[i] = (np.sum(predicted & is_valid_ref) + np.sum(~predicted & ~is_valid_ref)) / len(is_valid_ref)

        def _plot_contour(pov, title, ref_border=None):
            plt.figure(), plt.title(title)
            c = plt.contourf(xx1, xx2, pov, 50, cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(c).set_label('Probability of Validity')
            border = plt.contour(xx1, xx2, pov, [min_pov], linewidths=1, colors='k')
            if ref_border is not None:
                for line in ref_border.allsegs[0]:
                    plt.plot(line[:, 0], line[:, 1], '--k', linewidth=.5)

            plt.scatter(x_norm[is_failed_train, 0], x_norm[is_failed_train, 1], s=25, c='r', marker='x')
            plt.scatter(x_norm[~is_failed_train, 0], x_norm[~is_failed_train, 1], s=25, color=(0, 1, 0), marker='x')

            plt.xlabel('$x_0$'), plt.ylabel('$x_1$')
            return border

        if plot:
            min_pov_border = _plot_contour(pov_ref, f'Reference: {problem.__class__.__name__}')
            if save_path is not None and save_ref:
                plt.savefig(f'{save_path}_ref.png')
            _plot_contour(pov_predicted, f'{self!s} @ {problem.__class__.__name__}', min_pov_border)
            if save_path is not None:
                plt.savefig(f'{save_path}_predicted.png')

            i_max_acc = np.argmax(acc)
            i_setting = np.argmin(np.abs(thr_values-min_pov))

            plt.figure()
            plt.title(f'ROC: {self!s} @ {problem.__class__.__name__}\n'
                      f'Max accuracy: {acc[i_max_acc]*100:.1f}% @ min_pov = {thr_values[i_max_acc]:.3f}\n'
                      f'Selected accuracy: {acc[i_setting]*100:.1f}% @ min_pov = {thr_values[i_setting]:.3f}')
            plt.plot([0, 1], [0, 1], '--k', linewidth=.5)

            # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
            points = np.column_stack([fpr, tpr]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='RdYlBu', norm=plt.Normalize(0, 1))
            lc.set_array(thr_values)
            lc.set_linewidth(2)
            line = plt.gca().add_collection(lc)
            plt.colorbar(line, ax=plt.gca()).set_label('Probability of Validity threshold')

            plt.xlim([0, 1]), plt.ylim([0, 1]), plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(f'{save_path}_roc.png')

        if show:
            plt.show()
        return fpr, tpr, acc, thr_values

    def initialize(self, problem: ArchOptProblemBase):
        pass

    def train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        """Train the model (x's are normalized), y_is_valid is a vector"""
        raise NotImplementedError

    def evaluate_probability_of_validity(self, x_norm: np.ndarray) -> np.ndarray:
        """Get the probability of validity (0 to 1) at nx points (x is normalized); should return a vector!"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class PredictionHCStrategy(HiddenConstraintStrategy):
    """Base class for a strategy that predictions where failed regions occur"""

    def __init__(self, predictor: PredictorInterface, constraint=True, min_pov=.5):
        self.predictor = predictor
        self.constraint = constraint
        self.min_pov = min_pov
        super().__init__()

    def initialize(self, problem: ArchOptProblemBase):
        self.predictor.initialize(problem)

    def mod_xy_train(self, x_norm: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Remove failed points form the training set
        is_not_failed = ~self.is_failed(y)
        return x_norm[is_not_failed, :], y[is_not_failed, :]

    def prepare_infill_search(self, x_norm: np.ndarray, y: np.ndarray):
        is_failed = self.is_failed(y)
        y_is_valid = (~is_failed).astype(float)
        self.predictor.train(x_norm, y_is_valid)
        self.predictor.training_set = (x_norm, y_is_valid)

    def adds_infill_constraint(self) -> bool:
        return self.constraint

    def evaluate_infill_constraint(self, x_norm: np.ndarray) -> np.ndarray:
        pov = self.predictor.evaluate_probability_of_validity(x_norm)
        pov = np.clip(pov, 0, 1)
        return self.min_pov-pov

    def mod_infill_objectives(self, x_norm: np.ndarray, f_infill: np.ndarray) -> np.ndarray:
        pov = self.predictor.evaluate_probability_of_validity(x_norm)
        pov = np.clip(pov, 0, 1)

        # The infill objectives are a minimization of some value between 0 and 1:
        # - The function-based infills (prediction mean), the underlying surrogates are trained on normalized y values
        # - The expected improvement is normalized between 0 and 1, where 1 corresponds to no expected improvement
        return 1-((1-f_infill).T*pov).T

    def __str__(self):
        type_str = 'G' if self.constraint else 'F'
        return f'Prediction {type_str}: {self.predictor!s}'

    def __repr__(self):
        min_pov_str = f', min_pov={self.min_pov}' if self.constraint else ''
        return f'{self.__class__.__name__}({self.predictor!r}, constraint={self.constraint}{min_pov_str})'


class SKLearnClassifier(PredictorInterface):
    _reset_pickle_keys = ['_predictor']

    def __init__(self):
        self._predictor = None
        self._trained_single_class = None
        super().__init__()

    def evaluate_probability_of_validity(self, x_norm: np.ndarray) -> np.ndarray:
        if self._trained_single_class is not None:
            return np.ones((x_norm.shape[0],))*self._trained_single_class

        pov = self._predictor.predict_proba(x_norm)[:, 1]  # Probability of belonging to class 1 (valid points)
        return pov[:, 0] if len(pov.shape) == 2 else pov

    def train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        # Check if we are training a classifier with only 1 class
        self._trained_single_class = y_is_valid[0] if len(set(y_is_valid)) == 1 else None
        self._train(x_norm, y_is_valid)

    def _train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class RandomForestClassifier(SKLearnClassifier):

    def __init__(self, n: int = 100):
        self.n = n
        super().__init__()

    def _train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from sklearn.ensemble import RandomForestClassifier
        self._predictor = clf = RandomForestClassifier(n_estimators=self.n)
        clf.fit(x_norm, y_is_valid)

    def __str__(self):
        return f'Random Forest Classifier ({self.n})'

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self.n})'


class KNNClassifier(SKLearnClassifier):

    def __init__(self, k: int = 5, k_dim: float = None):
        self.k = self.k0 = k
        self.k_dim = k_dim
        super().__init__()

    def initialize(self, problem: ArchOptProblemBase):
        if self.k_dim is not None:
            self.k = max(self.k0, int(self.k_dim*problem.n_var))

    def _train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from sklearn.neighbors import KNeighborsClassifier
        self._predictor = clf = KNeighborsClassifier(n_neighbors=self.k)
        clf.fit(x_norm, y_is_valid)

    def __str__(self):
        return f'KNN Classifier ({self.k})'

    def __repr__(self):
        return f'{self.__class__.__name__}(k={self.k})'


class GPClassifier(SKLearnClassifier):

    def __init__(self, nu=2.5):
        self.nu = nu
        super().__init__()

    def _train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import Matern
        self._predictor = clf = GaussianProcessClassifier(kernel=1.*Matern(length_scale=.1, nu=self.nu))
        clf.fit(x_norm, y_is_valid)

    def __str__(self):
        return f'GP Classifier (nu = {self.nu})'

    def __repr__(self):
        return f'{self.__class__.__name__}(nu={self.nu})'


class SVMClassifier(SKLearnClassifier):

    def _train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from sklearn.svm import SVR
        self._predictor = clf = SVR(kernel='rbf')
        clf.fit(x_norm, y_is_valid)

    def evaluate_probability_of_validity(self, x_norm: np.ndarray) -> np.ndarray:
        return self._predictor.predict(x_norm)

    def __str__(self):
        return f'RBF SVM'


class VariationalGP(PredictorInterface):
    _reset_pickle_keys = ['_model']

    def __init__(self):
        self._model = None
        super().__init__()

    def train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        import tensorflow as tf
        from trieste.models.gpflow import build_vgp_classifier, VariationalGaussianProcess
        from trieste.models.optimizer import BatchOptimizer
        from trieste.observer import Dataset
        from trieste.space import Box

        # https://secondmind-labs.github.io/trieste/1.0.0/notebooks/failure_ego.html#Build-GPflow-models
        dataset = Dataset(tf.constant(x_norm, dtype=tf.float64), tf.cast(y_is_valid[:, None], dtype=tf.float64))
        search_space = Box([0]*x_norm.shape[1], [1]*x_norm.shape[1])
        classifier = build_vgp_classifier(dataset, search_space, noise_free=True)

        self._model = model = VariationalGaussianProcess(
            classifier, BatchOptimizer(tf.optimizers.Adam(1e-3)), use_natgrads=True)
        model.optimize(dataset)

    def evaluate_probability_of_validity(self, x_norm: np.ndarray) -> np.ndarray:
        import tensorflow as tf
        pov, _ = self._model.predict(tf.constant(x_norm, dtype=tf.float64))
        return pov.numpy()[:, 0]

    def __str__(self):
        return 'Variational GP'


class SMTPredictor(PredictorInterface):
    _reset_pickle_keys = ['_model']

    def __init__(self):
        self._model: Optional[SurrogateModel] = None
        super().__init__()

    def evaluate_probability_of_validity(self, x_norm: np.ndarray) -> np.ndarray:
        return self._model.predict_values(x_norm)[:, 0]

    def train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LinearRBFRegressor(SMTPredictor):

    def train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from smt.surrogate_models.rbf import RBF
        self._model = model = RBF(print_global=False, poly_degree=1)
        model.set_training_values(x_norm, y_is_valid)
        model.train()

    def __str__(self):
        return 'Linear RBF'


class GPRegressor(SMTPredictor):

    def train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from smt.surrogate_models.krg import KRG
        self._model = model = KRG(print_global=False)
        model.set_training_values(x_norm, y_is_valid)
        model.train()

    def __str__(self):
        return 'GP'


if __name__ == '__main__':
    # GPRegressor().get_stats()
    # GPRegressor().get_stats(HCBranin())
    # LinearRBFRegressor().get_stats()

    # RandomForestClassifier(n=500).get_stats()
    # KNNClassifier(k=5).get_stats()
    # GPClassifier().get_stats()
    # SVMClassifier().get_stats()

    # VariationalGP().get_stats()

    # GPClassifier().get_stats(CantileveredBeamHC())
    RandomForestClassifier(n=500).get_stats(HierarchicalRosenbrockHC())
