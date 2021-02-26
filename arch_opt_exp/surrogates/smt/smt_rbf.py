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
