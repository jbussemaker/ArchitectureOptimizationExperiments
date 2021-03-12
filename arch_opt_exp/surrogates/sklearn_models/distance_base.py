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

import math
import numpy as np
from typing import *
from sklearn.gaussian_process.kernels import \
    Matern, _check_length_scale, pdist, squareform, cdist, gamma, kv, _approx_fprime, KernelOperator, Kernel,\
    ConstantKernel, Hyperparameter

__all__ = ['CustomDistanceKernel', 'MixedIntKernel', 'Distance', 'WeightedDistance', 'IsDiscreteMask', 'Hyperparameter']


IsDiscreteMask = Union[np.ndarray, List[bool]]


class Distance:
    """
    Base class for implementing some component-wise distance calculation: d(x, x')
    Support for integer, categorical, and hierarchical design variables.
    """

    def __init__(self):
        self.xt = None
        self.yt = None
        self.is_active = None
        self.is_int_mask = None
        self.is_cat_mask = None
        self.is_discrete_mask = None
        self.is_cont_mask = None

        self.predict_is_active = None

    def set_samples(self, x, y, is_int_mask: IsDiscreteMask, is_cat_mask: IsDiscreteMask, is_active: np.ndarray = None):
        """Set before training the GP model."""
        self.xt = x
        self.yt = y
        self.is_active = is_active if is_active is not None else np.ones(x.shape, dtype=bool)

        self.is_int_mask = MixedIntKernel.get_discrete_mask(is_int_mask)
        self.is_cat_mask = MixedIntKernel.get_discrete_mask(is_cat_mask)
        self.is_discrete_mask = np.bitwise_or(self.is_int_mask, self.is_cat_mask)
        self.is_cont_mask = ~self.is_discrete_mask

        self._process_samples(x, y)

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        pass

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        pass

    def get_hyperparameter_values(self) -> list:
        return []

    def set_hyperparameter_values(self, values: list):
        pass

    def predict_set_is_active(self, is_active: np.ndarray):
        self.predict_is_active = is_active

    def __call__(self, u: Union[np.ndarray, list], v: Union[np.ndarray, list], u_is_active: np.ndarray = None,
                 v_is_active: np.ndarray = None, **kwargs) -> float:
        u, v = np.atleast_1d(u), np.atleast_1d(v)
        if u_is_active is None:
            u_is_active = np.ones((len(u),), dtype=bool)
        if v_is_active is None:
            v_is_active = np.ones((len(v),), dtype=bool)
        return float(self._call(u, v, u_is_active, v_is_active))

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray) -> float:
        raise NotImplementedError

    def kernel(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class WeightedDistance(Distance):
    """Distance that uses weighting terms (e.g. theta) in all its dimensions."""

    def __init__(self, theta0=1., fix_theta=False, theta_bounds=(1e-5, 1e5)):
        self.theta = [theta0]
        self.fix_theta = fix_theta
        self.theta_bounds = theta_bounds
        super(WeightedDistance, self).__init__()

    def set_samples(self, *args, **kwargs):
        super(WeightedDistance, self).set_samples(*args, **kwargs)

        if len(self.theta) == 1:
            self.theta = np.ones((self.xt.shape[1],)) * self.theta[0]
            if self.xt.shape[1] == 0:
                self.fix_theta = True

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        if self.fix_theta:
            return []
        return [
            Hyperparameter('theta', 'numeric', self.theta_bounds, n_elements=self.xt.shape[1]),
        ]

    def get_hyperparameter_values(self) -> list:
        if self.fix_theta:
            return []
        return [self.theta]

    def set_hyperparameter_values(self, values: list):
        if not self.fix_theta:
            self.theta, = values

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray) -> float:
        raise NotImplementedError

    def kernel(self, **kwargs):
        raise NotImplementedError


class MixedIntKernel(KernelOperator):
    """
    A mixed integer kernel that is based on the principle by Roustant et al. that the kernel value is the Hadamard
    product of the continuous kernel and the discrete kernel.

    See for example Eq. 8 in:
    Pelamatti 2019: "Surrogate Model Based Optimization of Constrained Mixed Variable Problems"

    Implementation inspired by the Product KernelOperator.
    """

    def __init__(self, cont_kernel: Kernel, discrete_kernel: Kernel, is_discrete_mask=None):
        super(MixedIntKernel, self).__init__(cont_kernel, discrete_kernel)

        self._is_discrete_mask: Optional[IsDiscreteMask] = is_discrete_mask
        self._is_cont_mask: Optional[IsDiscreteMask] = ~is_discrete_mask if is_discrete_mask is not None else None

    def set_discrete_mask(self, is_discrete_mask: IsDiscreteMask):
        self._is_discrete_mask = MixedIntKernel.get_discrete_mask(is_discrete_mask)
        self._is_cont_mask = ~self._is_discrete_mask

        if isinstance(self.k2, CustomDistanceKernel):
            self.k2.set_discrete_mask(is_discrete_mask[is_discrete_mask])

    def set_samples(self, x, y, is_int_mask: IsDiscreteMask, is_cat_mask: IsDiscreteMask, is_active: np.ndarray = None):
        if isinstance(self.k2, CustomDistanceKernel):
            m = self._is_discrete_mask
            if is_active is not None:
                is_active = is_active[:, m]
            self.k2.set_samples(x[:, m], y, is_int_mask[m], is_cat_mask[m], is_active=is_active)

    def predict_set_is_active(self, is_active: np.ndarray):
        if isinstance(self.k2, CustomDistanceKernel):
            self.k2.predict_set_is_active(is_active[:, self._is_discrete_mask])

    def get_params(self, deep=True):
        params = {
            'cont_kernel': self.k1,
            'discrete_kernel': self.k2,
            'is_discrete_mask': self._is_discrete_mask,
        }
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(('cont_kernel__' + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(('discrete_kernel__' + k, val) for k, val in deep_items)

        return params

    def __call__(self, X, Y=None, eval_gradient=False):
        x_cont, x_dis = self._split(X)
        y_cont = y_dis = None
        if Y is not None:
            y_cont, y_dis = self._split(Y)

        if x_cont.shape[1] == 0:  # No continuous variables
            if eval_gradient:
                K2, K2_gradient = self.k2(x_dis, y_dis, eval_gradient=eval_gradient)
                n_k1 = self.k1.bounds.shape[0]
                return K2, np.dstack((np.zeros(K2_gradient.shape[:2]+(n_k1,)), K2_gradient))
            return self.k2(x_dis, y_dis)

        elif x_dis.shape[1] == 0:  # No discrete variables
            if eval_gradient:
                K1, K1_gradient = self.k1(x_cont, y_cont, eval_gradient=eval_gradient)
                n_k2 = self.k2.bounds.shape[0]
                return K1, np.dstack((K1_gradient, np.zeros(K1_gradient.shape[:2]+(n_k2,))))
            return self.k1(x_cont, y_cont)

        if eval_gradient:
            K1, K1_gradient = self.k1(x_cont, y_cont, eval_gradient=True)
            K2, K2_gradient = self.k2(x_dis, y_dis, eval_gradient=True)
            return K1 * K2, np.dstack((K1_gradient * K2[:, :, np.newaxis],
                                       K2_gradient * K1[:, :, np.newaxis]))
        else:
            return self.k1(x_dis, y_dis) * self.k2(x_dis, y_dis)

    def diag(self, X):
        x_cont, x_dis = self._split(X)

        if x_cont.shape[1] == 0:
            return self.k2.diag(x_dis)
        elif x_dis.shape[1] == 0:
            return self.k1.diag(x_cont)

        return self.k1.diag(x_cont) * self.k2.diag(x_dis)

    def _split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._is_cont_mask is None or self._is_discrete_mask is None:
            raise ValueError('Discrete/continuous masks not set!')

        x_cont = x[:, self._is_cont_mask].astype(np.float)
        x_dis = x[:, self._is_discrete_mask].astype(np.int)
        return x_cont, x_dis

    @staticmethod
    def get_discrete_mask(is_discrete_mask: IsDiscreteMask) -> np.ndarray:
        if isinstance(is_discrete_mask, np.ndarray):
            return is_discrete_mask
        return np.array(is_discrete_mask, dtype=bool)

    @staticmethod
    def get_cont_kernel() -> Kernel:
        return ConstantKernel(1.)*Matern(1., nu=1.5)


class CustomDistanceKernel(Matern):
    """
    Same as the Matern kernel, but with a custom (discrete) distance metric. Special values of nu:
    nu=0      --> k(x, x') = d(x, x')
    nu=.5     --> k(x, x') = exp(-theta*d(x, x'))
    nu=np.inf --> k(x, x') = exp(-theta*.5*d(x, x')**2)

    The distance metric can be an instance of Distance, or some distance name as available in the scipy.spatial.distance
    package (see https://docs.scipy.org/doc/scipy/reference/spatial.distance.html).

    Note that the length_scale hyperparameter implementation does not work well for discrete variables! It is better to
    use WeightedDistance to tune the scales for all dimensions separately.
    """

    def __init__(self, metric: Union[str, Distance] = None, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=.5,
                 _is_discrete_mask=None, _train_is_active=None, _predict_is_active=None, **metric_hp):
        super(CustomDistanceKernel, self).__init__(
            length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.metric = metric if metric is not None else 'euclidean'

        self._is_discrete_mask: Optional[IsDiscreteMask] = _is_discrete_mask
        self._is_cont_mask: Optional[IsDiscreteMask] = ~_is_discrete_mask if _is_discrete_mask is not None else None

        self._train_is_active = _train_is_active
        self._predict_is_active = _predict_is_active

        self.__metric_hp = None
        if len(metric_hp) > 0:
            for key, value in metric_hp.items():
                setattr(self, key, value)
            self._set_metric_hp_from_attr()

    def set_discrete_mask(self, is_discrete_mask: IsDiscreteMask):
        self._is_discrete_mask = MixedIntKernel.get_discrete_mask(is_discrete_mask)
        self._is_cont_mask = ~self._is_discrete_mask

    def set_samples(self, x, y, is_int_mask: IsDiscreteMask, is_cat_mask: IsDiscreteMask, is_active: np.ndarray = None):
        self._train_is_active = is_active
        if isinstance(self.metric, Distance):
            self.metric.set_samples(x, y, is_int_mask, is_cat_mask, is_active=is_active)

    def predict_set_is_active(self, is_active: np.ndarray):
        self._predict_is_active = is_active
        if isinstance(self.metric, Distance):
            self.metric.predict_set_is_active(is_active)

    @property
    def hyperparameters(self):
        hyperparameters = super(CustomDistanceKernel, self).hyperparameters
        hyperparameters += self._get_metric_hyperparameters()
        return hyperparameters

    def _get_metric_hyperparameters(self) -> List[Hyperparameter]:
        if self.__metric_hp is not None:
            return self.__metric_hp

        if isinstance(self.metric, Distance):
            metric_hp = self.metric.hyperparameters()
            if metric_hp is not None:
                self.__metric_hp = [Hyperparameter('d_'+hp[0], *hp[1:]) for hp in metric_hp]
                return self.__metric_hp

        self.__metric_hp = []
        return self.__metric_hp

    def get_params(self, deep=True):
        params = super(CustomDistanceKernel, self).get_params(deep=deep)
        params['_is_discrete_mask'] = self._is_discrete_mask
        params['_train_is_active'] = self._train_is_active
        params['_predict_is_active'] = self._predict_is_active

        if isinstance(self.metric, Distance):
            metric_hp = self._get_metric_hyperparameters()
            for i, value in enumerate(self.metric.get_hyperparameter_values()):
                params[metric_hp[i].name] = value

        return params

    def set_params(self, **params):
        super(CustomDistanceKernel, self).set_params(**params)
        self._set_metric_hp_from_attr()

    def _set_metric_hp_from_attr(self):
        if isinstance(self.metric, Distance):
            values = [getattr(self, hp.name) for hp in self._get_metric_hyperparameters()]
            self.metric.set_hyperparameter_values(values)

    def __call__(self, x, y=None, eval_gradient=False):
        x = np.atleast_2d(x)
        length_scale = _check_length_scale(x, self.length_scale)
        x_norm = x.copy()
        x_norm[:, self._is_cont_mask] = x_norm[:, self._is_cont_mask] / length_scale
        if y is None:
            dists = self._pdist(x_norm)
        else:
            y_norm = y.copy()
            y_norm[:, self._is_cont_mask] = y_norm[:, self._is_cont_mask] / length_scale
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = self._cdist(x_norm, y_norm)

        if self.nu == 0.:  # Directly use distances as the Kernel
            K = dists
        elif self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-dists ** 2 / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            # We need to approximate numerically; the original derivations are only valid for the euclidean metric
            def f(theta):  # helper function
                return self.clone_with_theta(theta)(x, y)
            return K, _approx_fprime(self.theta, f, 1e-10)
        else:
            return K

    def _pdist(self, x: np.ndarray):  # Called during training
        if not isinstance(self.metric, Distance):
            return pdist(x, metric=self.metric)

        x = np.asarray(x, order='c')
        m, n = x.shape
        dm = np.empty((m * (m - 1)) // 2, dtype=np.double)

        is_active = self._train_is_active
        if is_active is None:
            raise ValueError('Training activity flags not set!')

        k = 0
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                dm[k] = self.metric(x[i], x[j], u_is_active=is_active[i], v_is_active=is_active[j])
                k = k + 1
        return dm

    def _cdist(self, x: np.ndarray, y: np.ndarray):  # Called during prediction
        if not isinstance(self.metric, Distance):
            return cdist(x, y, metric=self.metric)

        x = np.asarray(x, order='c')  # Prediction points
        y = np.asarray(y, order='c')  # Training points
        dm = np.empty((x.shape[0], y.shape[0]), dtype=np.double)

        train_is_active = self._train_is_active
        if train_is_active is None:
            raise ValueError('Training activity flags not set!')
        predict_is_active = self._predict_is_active
        if predict_is_active is None:
            raise ValueError('Prediction activity flags not set!')

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dm[i, j] = self.metric(x[i], y[j], u_is_active=predict_is_active[i], v_is_active=train_is_active[j])
        return dm


    def __repr__(self):
        return '%s(metric=%s, nu=%r)' % (self.__class__.__name__, self.metric, self.nu)
