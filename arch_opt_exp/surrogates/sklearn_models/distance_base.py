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
    ConstantKernel

__all__ = ['CustomDistanceKernel', 'MixedIntKernel', 'Distance', 'IsDiscreteMask']


IsDiscreteMask = Union[np.ndarray, List[bool]]


class Distance:
    """Base class for implementing some component-wise distance calculation: d(x, x')"""

    def __init__(self):
        self.xt = None
        self.yt = None
        self.is_int_mask = None
        self.is_cat_mask = None
        self.is_discrete_mask = None
        self.is_cont_mask = None

    def set_samples(self, x, y, is_int_mask: IsDiscreteMask, is_cat_mask: IsDiscreteMask):
        """Set before training the GP model."""
        self.xt = x
        self.yt = y

        self.is_int_mask = MixedIntKernel.get_discrete_mask(is_int_mask)
        self.is_cat_mask = MixedIntKernel.get_discrete_mask(is_cat_mask)
        self.is_discrete_mask = np.bitwise_or(self.is_int_mask, self.is_cat_mask)
        self.is_cont_mask = ~self.is_discrete_mask

        self._process_samples(x, y)

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        pass

    def __call__(self, u: Union[np.ndarray, list], v: Union[np.ndarray], **kwargs) -> float:
        uv = np.array([u, v]).T
        return float(self._call(uv))

    def _call(self, uv: np.ndarray) -> float:
        raise NotImplementedError

    def kernel(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


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
    Same as the Matern kernel, but with a custom distance metric:
    k(x, x') = exp(-theta*d(x, x'))

    The distance metric can be an instance of Distance, or some distance name as available in the scipy.spatial.distance
    package (see https://docs.scipy.org/doc/scipy/reference/spatial.distance.html).
    """

    def __init__(self, metric: Union[str, Distance] = None, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        super(CustomDistanceKernel, self).__init__(
            length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.metric = metric if metric is not None else 'sqeuclidean'

    def __call__(self, x, y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        x = np.atleast_2d(x)
        length_scale = _check_length_scale(x, self.length_scale)
        if y is None:
            dists = pdist(x / length_scale, metric=self.metric)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(x / length_scale, y / length_scale,
                          metric=self.metric)

        if self.nu == 0.5:
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
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((x.shape[0], x.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (x[:, np.newaxis, :] - x[np.newaxis, :, :])**2 \
                    / (length_scale ** 2)
            else:
                D = squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                K_gradient = K[..., np.newaxis] * D \
                    / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K_gradient[~np.isfinite(K_gradient)] = 0
            elif self.nu == 1.5:
                K_gradient = \
                    3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(x, y)
                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        return '%s(metric=%s, nu=%r)' % (self.__class__.__name__, self.metric, self.nu)
