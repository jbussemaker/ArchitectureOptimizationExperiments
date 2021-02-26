import numpy as np
from typing import *
from arch_opt_exp.surrogates import SurrogateModel
from smt.surrogate_models.surrogate_model import SurrogateModel as SMTSurrogateModelBase


class SMTSurrogateModel(SurrogateModel):
    """Adapter for SMT surrogate models."""

    _exclude = ['_smt_model', '_xt_last']

    def __init__(self):
        self._smt_model: Optional[SMTSurrogateModelBase] = None
        self._xt_last = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    @property
    def _smt(self) -> SMTSurrogateModelBase:
        if self._smt_model is None:
            self._smt_model = self._create_surrogate_model()
        return self._smt_model

    def set_samples(self, x: np.ndarray, y: np.ndarray):
        self._xt_last = x
        self._smt.set_training_values(x, y)

    def train(self):
        self._smt.train()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._smt.predict_values(x)

    def supports_variance(self) -> bool:
        return self._smt.supports['variances']

    def predict_variance(self, x: np.ndarray) -> np.ndarray:
        # There is a bug in SMT preventing a vectorized evaluation:
        # https://github.com/SMTorg/smt/issues/160

        variance = np.zeros((x.shape[0], self._smt.ny))
        for i in range(x.shape[0]):
            variance[i, :] = self._smt.predict_variances(x[[i], :])
        return variance

    def _create_surrogate_model(self) -> SMTSurrogateModelBase:
        raise NotImplementedError
