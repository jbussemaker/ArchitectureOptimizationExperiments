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

__all__ = ['GowerDistance']


class GowerDistance(Distance):
    """
    Gower distance. Assumes a normalized input space.

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


if __name__ == '__main__':
    from arch_opt_exp.surrogates.validation import *
    from arch_opt_exp.problems.discrete_branin import *
    from arch_opt_exp.surrogates.sklearn_models.gp import *

    problem = MixedIntBraninProblem()

    kernel = GowerDistance(problem.is_int_mask).kernel()

    sm = SKLearnGPSurrogateModel(kernel=kernel, alpha=1e-6)
    LOOCrossValidation.check_sample_sizes(sm, problem, repair=problem.get_repair(), show=True)
