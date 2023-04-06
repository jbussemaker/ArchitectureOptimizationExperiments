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
from sb_arch_opt.problem import *
from sb_arch_opt.algo.simple_sbo.algo import *
from arch_opt_exp.experiments.metrics_base import *
from arch_opt_exp.hc_strategies.prediction import *
from arch_opt_exp.hc_strategies.sbo_with_hc import *
from pymoo.core.algorithm import Algorithm

__all__ = ['FailRateMetric', 'PredictorMetric']


class FailRateMetric(Metric):
    """Measure the failure rate in the population"""

    @property
    def name(self):
        return 'fail'

    @property
    def value_names(self) -> List[str]:
        return ['total', 'failed', 'rate']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        pop = self._get_pop(algorithm)
        is_failed = ArchOptProblemBase.get_failed_points(pop)

        n_total = len(is_failed)
        n_failed = int(np.sum(is_failed))
        fail_rate = 0 if n_total == 0 else (n_failed/n_total)
        return [n_total, n_failed, fail_rate]


class PredictorMetric(Metric):
    """
    Measures performance of the predictor if a prediction HC strategy is used.
    More info on terminology: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """

    @property
    def name(self):
        return 'hc_pred'

    @property
    def value_names(self) -> List[str]:
        return ['acc', 'tpr', 'fpr', 'max_acc', 'max_acc_pov']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        strategy = self._get_predictor(algorithm)
        if strategy is None:
            return [np.nan]*len(self.value_names)
        min_pov = strategy.min_pov
        predictor = strategy.predictor

        stats = predictor.get_stats(algorithm.problem, train=False, plot=False, show=False)
        if stats is None:
            return [np.nan]*len(self.value_names)
        fpr, tpr, acc, x_min_pov = stats
        i_selected = np.argmin(np.abs(x_min_pov-min_pov))
        fpr_value, tpr_value, acc_value = fpr[i_selected], tpr[i_selected], acc[i_selected]

        i_max_acc = np.argmax(acc)
        max_acc = acc[i_max_acc]
        max_acc_pov = x_min_pov[i_max_acc]

        return [acc_value, tpr_value, fpr_value, max_acc, max_acc_pov]

    @staticmethod
    def _get_predictor(algorithm: Algorithm) -> Optional[PredictionHCStrategy]:
        if isinstance(algorithm, InfillAlgorithm):
            if isinstance(algorithm.infill_obj, HiddenConstraintsSBO):
                if isinstance(algorithm.infill_obj.hc_strategy, PredictionHCStrategy):
                    return algorithm.infill_obj.hc_strategy
