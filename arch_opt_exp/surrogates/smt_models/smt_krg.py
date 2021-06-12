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
from smt.surrogate_models.krg import KRG
from smt.surrogate_models.kpls import KPLS
from smt.applications.mixed_integer import MixedIntegerSurrogateModel
from arch_opt_exp.surrogates.smt_models.smt_surrogate_model import SMTSurrogateModel

__all__ = ['SMTKrigingSurrogateModel', 'SMTKPLSSurrogateModel']


class SMTKrigingSurrogateModel(SMTSurrogateModel):
    """Normal Kriging (SMT package)"""

    def __init__(self, theta0=None, auto_wrap_mixed_int=False, **kwargs):
        super(SMTKrigingSurrogateModel, self).__init__(auto_wrap_mixed_int=auto_wrap_mixed_int)
        self._theta0 = theta0
        self._kw = kwargs

    def theta0(self, n_x):
        theta0 = self._theta0
        if theta0 is None:
            theta0 = 1e-2
        if np.isscalar(theta0):
            theta0 = [theta0]*n_x
        return theta0

    def _create_surrogate_model(self):
        return KRG(
            print_global=False,
            **(self._kw or {}),
        )

    def train(self):
        smt = self._smt
        if isinstance(smt, MixedIntegerSurrogateModel):
            smt = smt._surrogate
        if 'theta0' in smt.options:
            smt.options['theta0'] = self.theta0(smt.nx)

        super(SMTKrigingSurrogateModel, self).train()


class SMTKPLSSurrogateModel(SMTKrigingSurrogateModel):
    """Kriging with Partial Least Squares wrapper (SMT package)"""

    def __init__(self, theta0=None, n_comp=5, **kwargs):
        """
        :param theta0: Initial hyperparameter
        :param n_comp: Number of principle components
        """
        super(SMTKPLSSurrogateModel, self).__init__(theta0=theta0, **kwargs)
        self.n_comp = n_comp

    def _create_surrogate_model(self):
        return KPLS(
            print_global=False,
            n_comp=self.n_comp,
            **(self._kw or {}),
        )

    def train(self):
        smt = self._smt
        if isinstance(smt, MixedIntegerSurrogateModel):
            smt = smt._surrogate
        if 'theta0' in smt.options:
            smt.options['theta0'] = self.theta0(self.n_comp)

        super(SMTKPLSSurrogateModel, self).train()
