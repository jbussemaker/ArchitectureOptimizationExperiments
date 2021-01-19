"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2020, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""

import numpy as np
from typing import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

__all__ = ['ModulatedMOInfill']


class ModulatedMOInfill(SurrogateInfill):
    """
    Multi-objective infill criteria themselves might be represented by a single-objective design space: a criteria
    representing the improvement of the Pareto front might have the same value at different points at the Pareto front.
    To help with exploration, the single-objective MO infill criterion might be turned into a multi-objective criterion
    by modulating the criterion for all original objective values. This helps in finding multiple infill points along
    the current Pareto front at each infill iteration.
    """

    def __init__(self, underlying: SurrogateInfill):
        super(ModulatedMOInfill, self).__init__()

        self.underlying = underlying

    @property
    def needs_variance(self):
        return self.underlying.needs_variance

    def initialize(self, *args, **kwargs):
        super(ModulatedMOInfill, self).initialize(*args, **kwargs)
        self.underlying.initialize(*args, **kwargs)

    def set_training_values(self, x_train: np.ndarray, y_train: np.ndarray):
        super(ModulatedMOInfill, self).set_training_values(x_train, y_train)
        self.underlying.set_training_values(x_train, y_train)

    def get_n_infill_objectives(self) -> int:
        return self.underlying.get_n_infill_objectives()*self.problem.n_obj

    def get_n_infill_constraints(self) -> int:
        return self.underlying.get_n_infill_constraints()

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n_f = self.problem.n_obj
        f_underlying, g = self.underlying.evaluate(x)

        f_predicted, _ = self.predict(x)

        f_modulated = np.empty((f_predicted.shape[0], n_f*f_underlying.shape[1]))
        for i_f_underlying in range(f_underlying.shape[1]):
            f_modulated[:, i_f_underlying*n_f:i_f_underlying*n_f+n_f] = f_underlying[:, [i_f_underlying]]*f_predicted

        return f_modulated, g
