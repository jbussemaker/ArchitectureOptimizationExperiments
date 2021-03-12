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

import numba
import numpy as np
from typing import *
from scipy.spatial.distance import hamming
from arch_opt_exp.surrogates.sklearn_models.distance_base import *

__all__ = ['GowerDistance', 'SymbolicCovarianceDistance', 'HammingDistance', 'CompoundSymmetryKernel',
           'LatentVariablesDistance']


class GowerDistance(WeightedDistance):
    """
    Gower distance. Assumes continuous variables are normalized between [0, 1], and discrete variables have values
    between 0 and n-1 (including).

    The Gower distance is also known as the Heterogeneous Euclidean-Overlap Metric (HEOM).

    Implementation based on:
    Halstrup 2016, "Black-box Optimization of Mixed Discrete-Continuous Optimization Problems", section 6.6
    """

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray) -> float:
        u_cont, v_cont = u[self.is_cont_mask], v[self.is_cont_mask]
        u_dis, v_dis = u[self.is_discrete_mask].astype(np.int), v[self.is_discrete_mask].astype(np.int)
        return _gower(u_cont, v_cont, u_dis, v_dis, self.is_cont_mask, self.is_discrete_mask, self.theta)

    def kernel(self, **kwargs):
        return CustomDistanceKernel(metric=self, length_scale_bounds='fixed', **kwargs)


@numba.jit(nopython=True)
def _gower(u_cont, v_cont, u_dis, v_dis, is_cont_mask, is_discrete_mask, theta) -> float:

    s = np.empty((len(is_cont_mask),))
    s[is_cont_mask] = np.abs(u_cont-v_cont)

    dis_is_same = u_dis == v_dis
    s_dis = np.ones((len(u_dis),))
    s_dis[dis_is_same] = 0
    s[is_discrete_mask] = s_dis

    return np.sum(s*theta)


class SymbolicCovarianceDistance(WeightedDistance):
    """
    Symbolic Covariance (SC).

    Implementation based on:
    - McCane 2008, "Black-box Optimization of Mixed Discrete-Continuous Optimization Problems", section 6.6
    - provided R code
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass
    """

    def __init__(self, int_as_discrete=True):
        super(SymbolicCovarianceDistance, self).__init__()

        self.int_as_discrete = int_as_discrete
        self._inv_cov_matrix = None

    @property
    def use_sc_mask(self):
        return self.is_discrete_mask if self.int_as_discrete else self.is_cat_mask

    @property
    def use_cont_mask(self):
        return self.is_cont_mask if self.int_as_discrete else np.bitwise_or(self.is_cont_mask, self.is_int_mask)

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        """Recalculate the covariance matrix when setting new samples for training."""

        x_cont = x[:, self.use_cont_mask]
        x_dis = x[:, self.use_sc_mask].astype(np.int)

        # Determine X - X_bar for all samples
        x_means = np.empty(x.shape)

        # The continuous case
        x_means[:, self.use_cont_mask] = x_cont-np.mean(x_cont, axis=0)

        # For discrete variables use the symbolic covariance
        x_means_dis = np.empty(x_dis.shape)
        for i in range(x_dis.shape[1]):
            n_x_levels_i = np.max(x_dis[:, i])+1
            x_means_dis[:, i] = self._symbolic_covariance_x_means(x_dis[:, i], n_x_levels_i)
        x_means[:, self.use_sc_mask] = x_means_dis

        # Calculate covariance matrix
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass
        n_x = x.shape[1]
        cov_matrix = np.ones((n_x, n_x))
        for i in range(n_x):
            for j in range(i):
                cov_matrix[i, j] = cov_matrix[j, i] = np.mean(x_means[:, i]*x_means[:, j])

        # Invert the covariance matrix
        self._inv_cov_matrix = np.linalg.inv(cov_matrix)

    @staticmethod
    def _symbolic_covariance_x_means(x: np.ndarray, n_x: int):
        """Calculates X - X_bar. Samples contain values between 0 and n-1 and should be of type int."""

        # Calculate marginal probabilities
        a = np.empty((n_x,))
        for i in range(n_x):
            a[i] = np.sum(x == i)

        a /= np.sum(a)

        # Set delta matrix: upper triangle is 1, lower triangle -1 (Eq. 8, 9)
        d_mat = np.tri(n_x, n_x, -1).T - np.tri(n_x, n_x, -1)

        # Calculate distances to the mean (Eq. 7)
        d_mean = np.sum(a*d_mat, axis=1)  # A - A_bar

        # Assign distances to the samples
        x_means = np.empty((len(x),))
        for i in range(n_x):
            x_means[x == i] = d_mean[i]

        return x_means

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray) -> float:
        """Calculate the distance function using the inverse of the symbolic covariance matrix."""
        return _sc_dist(self._inv_cov_matrix, u, v, self.use_sc_mask, self.theta)

    def kernel(self, **kwargs):
        return CustomDistanceKernel(metric=self, length_scale_bounds='fixed', **kwargs)


@numba.jit(nopython=True, cache=True)
def _sc_dist(inv_cov_matrix: np.ndarray, u: np.ndarray, v: np.ndarray, is_discrete_mask: np.ndarray, theta):
    delta_uv = u-v

    # The delta between symbolic variables is 1 if i < j, -1 if i > j (compare with d_mat)
    delta_uv[np.bitwise_and(is_discrete_mask, u < v)] = 1.
    delta_uv[np.bitwise_and(is_discrete_mask, u > v)] = -1.

    # Calculate Mahalanobis-like distance (Eq. 13)
    return np.dot(theta*delta_uv, np.dot(inv_cov_matrix, delta_uv))


class HammingDistance(WeightedDistance):
    """
    Hamming distance for the discrete terms. Based on the principle by Roustant et al. that the kernel value is the
    Hadamard product of the continuous kernel and the discrete kernel. See for example Eq. 8 in:
    Pelamatti 2019: "Surrogate Model Based Optimization of Constrained Mixed Variable Problems"
    """

    def __call__(self, u: Union[np.ndarray, list], v: Union[np.ndarray, list], **kwargs) -> float:
        return hamming(u, v, w=self.theta)

    def _call(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def kernel(self, **kwargs):
        cont_kernel = MixedIntKernel.get_cont_kernel()
        discrete_kernel = CustomDistanceKernel(self, length_scale_bounds='fixed')
        return MixedIntKernel(cont_kernel, discrete_kernel)


class CompoundSymmetryKernel(Distance):
    """
    Compount Symmetry (CS) kernel, based on:
    Roustant 2018, "Group Kernels for Gaussian Process Metamodels with Categorical Inputs"
    Pelamatti 2019, "Surrogate Model Based Optimization of Constrained Mixed Variable Problems"
    """

    def __init__(self, v0=1., fix_v=False, cf0=.5, fix_cf=False):
        self.v = [v0]
        self.fix_v = fix_v
        self.cf = [10**cf0]
        self.fix_cf = fix_cf
        super(CompoundSymmetryKernel, self).__init__()

        self.cf_x = None
        self.cf_l = None
        self.c = None
        self._n_dis_values = None

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        self._n_dis_values = n_dis_values = np.max(x, axis=0)+1
        self._cv_l = -1/(n_dis_values-1)

        if len(self.v) == 1:
            self.v = np.ones((self.xt.shape[1],))*self.v[0]
        if len(self.cf) == 1:
            self.cf = np.ones((self.xt.shape[1],))*self.cf[0]
            self._set_cf_x()

        if self.xt.shape[1] == 0:
            self.fix_v = self.fix_cf = True

    def _set_cf_x(self):
        self.cf_x = np.log10(self.cf)
        self.c = self.v*(self.cf_x*(1-self._cv_l)+self._cv_l)

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray) -> float:
        if len(u) == 0:
            return 1.
        return _cs(u, v, self.c, self.v)

    def kernel(self, **kwargs):
        cont_kernel = MixedIntKernel.get_cont_kernel()

        # nu = 0 means we are directly outputting the kernel values
        discrete_kernel = CustomDistanceKernel(self, length_scale_bounds='fixed', nu=0)
        return MixedIntKernel(cont_kernel, discrete_kernel)

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        return [
            Hyperparameter('v', 'numeric', 'fixed' if self.fix_v else (1e-2, 1.), n_elements=self.xt.shape[1]),
            Hyperparameter('cf', 'numeric', 'fixed' if self.fix_cf else (1e0, 1e1), n_elements=self.xt.shape[1]),
        ]

    def get_hyperparameter_values(self) -> list:
        return [self.v, self.cf]

    def set_hyperparameter_values(self, values: list):
        self.v, self.cf = values
        self._set_cf_x()


@numba.jit(nopython=True)
def _cs(u, v, cs_c, cs_v):
    d = np.ones((len(u),))*cs_c

    is_same_mask = u == v
    d[is_same_mask] = cs_v[is_same_mask]

    return np.prod(d)


class LatentVariablesDistance(Distance):
    """
    Latent Variables (LV) distance, based on:
    Zhang 2019, "A Latent Variable Approach to Gaussian Process Modeling with Qualitative and Quantitative Factors"
    Pelamatti 2020, "Bayesian Optimization of Variable-Size Design Space Problems"
    """

    def __init__(self, theta0=0., fix_theta=False):
        self.theta = [10**theta0]
        self.fix_theta = fix_theta
        super(LatentVariablesDistance, self).__init__()

        self.theta_x = None
        self._n_dis_values = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if state['theta_x'] is not None:
            state['theta_x'] = True
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if state['theta_x'] is True:
            self._set_theta_x()

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        self._n_dis_values = n_dis_values = [int(n) for n in np.max(x, axis=0)+1]

        if len(self.theta) == 1:
            if self.xt.shape[1] == 0:
                self.fix_theta = True
                self.theta = []
            else:
                theta = []
                for i in range(self.xt.shape[1]):
                    theta += [self.theta[0]]*(n_dis_values[i]*2-3)
                self.theta = 10**np.array(theta)
            self._set_theta_x()

    def _set_theta_x(self):
        theta_values = np.log10(self.theta)

        # Map each level of each discrete variable to its latent variable coordinates determine by theta
        # The first level is at (0,0), the second level is at the x-axis (0,...); therefore 3 theta's are skipped
        n_dis_values = self._n_dis_values
        self.theta_x = theta_x = numba.typed.List()
        for i in range(len(n_dis_values)):
            theta_x_i = np.zeros((n_dis_values[i]*2,))
            n_theta_x = len(theta_x_i)-3
            theta_x_i[3:] = theta_values[:n_theta_x]
            theta_x.append(theta_x_i.reshape((n_dis_values[i], 2)))

            theta_values = theta_values[n_theta_x:]

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray) -> float:
        if len(u) == 0:
            return 0.
        return _lv(u, v, self.theta_x)

    def kernel(self, **kwargs):
        cont_kernel = MixedIntKernel.get_cont_kernel()

        # nu = 0 means we are directly outputting the kernel values
        discrete_kernel = CustomDistanceKernel(self, length_scale_bounds='fixed', nu=.5)
        return MixedIntKernel(cont_kernel, discrete_kernel)

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        return [
            Hyperparameter('theta', 'numeric', 'fixed' if self.fix_theta else (1e-2, 1e2),
                           n_elements=len(self.theta)),
        ]

    def get_hyperparameter_values(self) -> list:
        return [self.theta]

    def set_hyperparameter_values(self, values: list):
        self.theta = values[0]
        self._set_theta_x()


@numba.jit(nopython=True)
def _lv(u, v, theta_x):
    u_latent = np.empty((len(theta_x), 2))
    v_latent = np.empty((len(theta_x), 2))
    for i in range(len(theta_x)):
        u_latent[i, :] = theta_x[i][u[i], :]
        v_latent[i, :] = theta_x[i][v[i], :]

    d_latent = u_latent-v_latent
    return np.sum(d_latent**2)


if __name__ == '__main__':
    from arch_opt_exp.surrogates.validation import *
    from arch_opt_exp.surrogates.sklearn_models.gp import *

    from arch_opt_exp.problems.discrete import *
    # problem = MixedIntBraninProblem()
    problem = MixedIntGoldsteinProblem()

    # kernel = None
    # kernel = GowerDistance().kernel()
    kernel = SymbolicCovarianceDistance(int_as_discrete=True).kernel()
    # kernel = HammingDistance().kernel()
    # kernel = CompoundSymmetryKernel().kernel()
    # kernel = LatentVariablesDistance().kernel()

    sm = SKLearnGPSurrogateModel(kernel=kernel, alpha=1e-6, int_as_discrete=True)
    LOOCrossValidation.check_sample_sizes(sm, problem, show=True, print_progress=True)
