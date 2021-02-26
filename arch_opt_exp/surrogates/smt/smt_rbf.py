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

from smt.surrogate_models.rbf import RBF
from arch_opt_exp.surrogates.smt.smt_surrogate_model import SMTSurrogateModel

__all__ = ['SMTRBFSurrogateModel']


class SMTRBFSurrogateModel(SMTSurrogateModel):
    """Radial Basis Function wrapper (SMT package)"""

    def __init__(self, d0=1., deg=-1, reg=1e-10):
        """
        :param d0: Basis function scaling parameter
        :param deg: Global polynomial: -1 no polynomial, 0 constant, 1 linear trend
        :param reg: Regularization coefficient
        """
        super(SMTRBFSurrogateModel, self).__init__()
        self.d0 = d0
        self.deg = deg
        self.reg = reg

    def _create_surrogate_model(self):
        return RBF(
            print_global=False,
            d0=self.d0,
            poly_degree=self.deg,
            reg=self.reg,
        )
