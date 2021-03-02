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
from arch_opt_exp.surrogates.sklearn_models.mixed_int_dist import *


def test_symbolic_covariance():
    # McCane 2008, Table 1
    a_sam = np.array([0, 1, 0, 1], dtype=np.int)
    b_sam = np.array([1, 0, 1, 0], dtype=np.int)
    c_sam = np.array([0, 1, 1, 0], dtype=np.int)
    d_sam = np.array([0, 1, 0, 0], dtype=np.int)

    sigma_aa, d_a, d_a2, d_a_mean, d_a_mean2, a_means, _ = \
        SymbolicCovarianceDistance._symbolic_covariance(a_sam, a_sam, 2, 2)
    assert sigma_aa == 1.

    assert np.all(d_a == d_a2)
    assert np.all(np.diag(d_a) == 0.)
    assert d_a[0, 1] == 1.
    assert d_a[1, 0] == -1.

    assert np.all(d_a_mean == [.5, -.5])
    assert np.all(d_a_mean == d_a_mean2)

    assert np.all(a_means == [.5, -.5, .5, -.5])

    sigma_ab, d_a2, d_b, d_a_mean2, d_b_mean, _, _ = SymbolicCovarianceDistance._symbolic_covariance(a_sam, b_sam, 2, 2)
    assert sigma_ab == -1.

    assert np.all(d_a2 == d_a)
    assert d_b[0, 1] == 1.
    assert d_b[1, 0] == -1.

    assert np.all(d_a_mean == d_a_mean2)
    assert np.all(d_b_mean == [.5, -.5])

    sigma_ac, d_a2, d_c, d_a_mean2, d_c_mean, _, _ = SymbolicCovarianceDistance._symbolic_covariance(a_sam, c_sam, 2, 2)
    assert sigma_ac == 0.

    assert np.all(d_a2 == d_a)
    assert d_c[0, 1] == 1.
    assert d_c[1, 0] == -1.

    assert np.all(d_a_mean == d_a_mean2)
    assert np.all(d_c_mean == [.5, -.5])

    sigma_ad, d_a2, d_d, d_a_mean2, d_d_mean, _, d_means = \
        SymbolicCovarianceDistance._symbolic_covariance(a_sam, d_sam, 2, 2)
    assert sigma_ad == .5

    assert np.all(d_a2 == d_a)
    assert d_d[0, 1] == 1.
    assert d_d[1, 0] == -1.

    assert np.all(d_a_mean == d_a_mean2)
    assert np.all(d_d_mean == [.25, -.75])
    assert np.all(d_means == [.25, -.75, .25, .25])

    sigma_bc, d_b2, d_c2, d_b_mean2, d_c_mean2, _, _ = \
        SymbolicCovarianceDistance._symbolic_covariance(b_sam, c_sam, 2, 2)
    assert sigma_bc == 0.

    assert np.all(d_b2 == d_b)
    assert np.all(d_c2 == d_c)
    assert np.all(d_b_mean2 == d_b_mean)
    assert np.all(d_c_mean2 == d_c_mean)


def test_symbolic_covariance_x_means():
    # McCane 2008, Table 1
    a_sam = np.array([0, 1, 0, 1], dtype=np.int)
    b_sam = np.array([1, 0, 1, 0], dtype=np.int)
    c_sam = np.array([0, 1, 1, 0], dtype=np.int)
    d_sam = np.array([0, 1, 0, 0], dtype=np.int)

    assert np.all(SymbolicCovarianceDistance._symbolic_covariance_x_means(a_sam, 2) == [.5, -.5, .5, -.5])
    assert np.all(SymbolicCovarianceDistance._symbolic_covariance_x_means(b_sam, 2) == [-.5, .5, -.5, .5])
    assert np.all(SymbolicCovarianceDistance._symbolic_covariance_x_means(c_sam, 2) == [.5, -.5, -.5, .5])
    assert np.all(SymbolicCovarianceDistance._symbolic_covariance_x_means(d_sam, 2) == [.25, -.75, .25, .25])
