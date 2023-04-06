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
from arch_opt_exp.hc_strategies.sbo_with_hc import *

__all__ = ['RejectionHCStrategy']


class RejectionHCStrategy(HiddenConstraintStrategy):
    """Strategy that simply rejects failed points before training the model"""

    def mod_xy_train(self, x_norm: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Remove failed points form the training set
        is_not_failed = ~self.is_failed(y)
        return x_norm[is_not_failed, :], y[is_not_failed, :]

    def __str__(self):
        return 'Rejection'

    def __repr__(self):
        return f'{self.__class__.__name__}()'
