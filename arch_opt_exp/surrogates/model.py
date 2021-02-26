import copy
import numpy as np

__all__ = ['SurrogateModel', 'SurrogateModelFactory']


class SurrogateModel:
    """
    Base class for the surrogate model as used in this package. Should be pickleable.
    """

    def copy(self) -> 'SurrogateModel':
        """Return an uninitialized copy of the surrogate model."""
        return copy.deepcopy(self)

    def set_samples(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def supports_variance(self) -> bool:
        raise NotImplementedError

    def predict_variance(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SurrogateModelFactory:

    def __init__(self, surrogate_model: SurrogateModel):
        self.base: SurrogateModel = surrogate_model.copy()

    def get(self):
        return self.base.copy()
