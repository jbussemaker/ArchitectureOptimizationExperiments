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
