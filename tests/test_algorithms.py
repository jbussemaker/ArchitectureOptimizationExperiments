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

import pytest
from pymoo.factory import get_problem
from pymoo.model.problem import Problem

from arch_opt_exp.experimenter import *
from arch_opt_exp.metrics.filters import *
from arch_opt_exp.metrics.convergence import *
from arch_opt_exp.algorithms.random_search import *


@pytest.fixture
def problem() -> Problem:
    return get_problem('zdt1')


def test_random_search(problem):
    cr_metric = ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5)

    algorithm = RandomSearchAlgorithm(pop_size=100)
    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cr_metric])
    res = exp.run_effectiveness(repeat_idx=0)

    cr = res.metrics[cr_metric.name].results()['cr']
    assert cr[-1] > cr[0]
