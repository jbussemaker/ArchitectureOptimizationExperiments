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

from typing import *
from pymoo.model.problem import Problem
from arch_opt_exp.metrics_base import Metric

from arch_opt_exp.metrics.termination import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.algorithms.surrogate.validation import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

__all__ = ['get_exp_metrics']


def get_exp_metrics(problem: Problem, include_loo_cv=True) -> List[Metric]:
    pf = problem.pareto_front()

    return [
        # Algorithm performance
        DeltaHVMetric(pf),
        IGDMetric(pf),
        SpreadMetric(),
        MaxConstraintViolationMetric(),

        # Convergence detection
        MDRTermination().metric,
        MGBMTermination().metric,
        FHITermination().metric,
        GDTermination().metric,
        IGDTermination().metric,
        CRTermination().metric,
        MCDTermination().metric,
        SPITermination().metric,

        # Diagnostics
        InfillMetric(),
        SurrogateQualityMetric(include_loo_cv=include_loo_cv, n_loo_cv=10),
    ]
