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
from arch_opt_exp.surrogates.sklearn_models.distance_base import *

__all__ = ['GowerDistance', 'SymbolicCovarianceDistance']


class GowerDistance(Distance):
    """
    Gower distance. Assumes continuous variables are normalized between [0, 1], and discrete variables have values
    between 0 and n-1 (including).

    The Gower distance is also known as the Heterogeneous Euclidean-Overlap Metric (HEOM).

    Implementation based on:
    Halstrup 2016, "Black-box Optimization of Mixed Discrete-Continuous Optimization Problems", section 6.6
    """

    def __init__(self, is_int_mask: IsIntMask):
        self.is_int_mask = MixedIntKernel.get_int_mask(is_int_mask)
        self.is_cont_mask = ~self.is_int_mask
        super(GowerDistance, self).__init__()

    def _call(self, uv: np.ndarray) -> float:
        uv_cont = uv[self.is_cont_mask, :]
        uv_int = uv[self.is_int_mask, :].astype(np.int)
        return _gower(uv_cont, uv_int, self.is_cont_mask, self.is_int_mask)

    def kernel(self, **kwargs):
        return CustomDistanceKernel(metric=self, **kwargs)


@numba.jit(nopython=True, cache=True)
def _gower(uv_cont: np.ndarray, uv_int: np.ndarray, is_cont_mask: np.ndarray, is_int_mask: np.ndarray) -> float:

    s = np.empty((len(is_cont_mask),))
    s[is_cont_mask] = np.abs(uv_cont[:, 0]-uv_cont[:, 1])

    int_is_same = uv_int[:, 0] == uv_int[:, 1]
    s_int = np.ones((uv_int.shape[0],))
    s_int[int_is_same] = 0
    s[is_int_mask] = s_int

    return np.mean(s)  # Eq. 6.36


class SymbolicCovarianceDistance(Distance):
    """
    Symbolic Covariance (SC).

    Implementation based on:
    - McCane 2008, "Black-box Optimization of Mixed Discrete-Continuous Optimization Problems", section 6.6
    - provided R code
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass
    """

    def __init__(self, is_int_mask: IsIntMask):
        super(SymbolicCovarianceDistance, self).__init__()
        self.is_int_mask = MixedIntKernel.get_int_mask(is_int_mask)
        self.is_cont_mask = ~self.is_int_mask

        self._inv_cov_matrix = None

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        x_cont = x[:, self.is_cont_mask]
        x_int = x[:, self.is_int_mask].astype(np.int)

        # Determine X - X_bar for all samples
        x_means = np.empty(x.shape)
        x_means[:, self.is_cont_mask] = x_cont-np.mean(x_cont, axis=0)

        x_means_int = np.empty(x_int.shape)
        for i in range(x_int.shape[1]):
            n_x_levels_i = np.max(x_int[:, i])+1
            x_means_int[:, i] = self._symbolic_covariance_x_means(x_int[:, i], n_x_levels_i)
        x_means[:, self.is_int_mask] = x_means_int

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

    def _call(self, uv: np.ndarray) -> float:
        """Calculate the distance function using the inverse of the symbolic covariance matrix."""
        return _sc_dist(self._inv_cov_matrix, uv, self.is_int_mask)

    def kernel(self, **kwargs):
        return CustomDistanceKernel(metric=self, **kwargs)


@numba.jit(nopython=True, cache=True)
def _sc_dist(inv_cov_matrix: np.ndarray, uv: np.ndarray, is_int_mask: np.ndarray):
    delta_uv = uv[:, 0]-uv[:, 1]

    # The delta between symbolic variables is 1 if i < j, -1 if i > j (compare with d_mat)
    delta_uv[np.bitwise_and(is_int_mask, uv[:, 0] < uv[:, 1])] = 1.
    delta_uv[np.bitwise_and(is_int_mask, uv[:, 0] > uv[:, 1])] = -1.

    # Calculate Mahalanobis-like distance (Eq. 13)
    return np.dot(delta_uv, np.dot(inv_cov_matrix, delta_uv))


if __name__ == '__main__':
    from arch_opt_exp.surrogates.validation import *
    from arch_opt_exp.problems.discrete_branin import *
    from arch_opt_exp.surrogates.sklearn_models.gp import *

    problem = MixedIntBraninProblem()

    # kernel = None
    # kernel = GowerDistance(problem.is_int_mask).kernel()
    kernel = SymbolicCovarianceDistance(problem.is_int_mask).kernel()

    sm = SKLearnGPSurrogateModel(kernel=kernel, alpha=1e-6)
    LOOCrossValidation.check_sample_sizes(sm, problem, repair=problem.get_repair(), show=True)
