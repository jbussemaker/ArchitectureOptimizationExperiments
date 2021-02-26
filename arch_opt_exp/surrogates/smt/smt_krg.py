import numpy as np
from smt.surrogate_models.krg import KRG
from smt.surrogate_models.kpls import KPLS
from arch_opt_exp.surrogates.smt.smt_surrogate_model import SMTSurrogateModel

__all__ = ['SMTKrigingSurrogateModel', 'SMTKPLSSurrogateModel']


class SMTKrigingSurrogateModel(SMTSurrogateModel):
    """Normal Kriging (SMT package)"""

    def __init__(self, theta0=None, **kwargs):
        super(SMTKrigingSurrogateModel, self).__init__()
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
        n_x = self._xt_last.shape[1]
        self._smt.options['theta0'] = self.theta0(n_x)
        self._smt.train()


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
        self._smt.options['theta0'] = self.theta0(self.n_comp)
        self._smt.train()
