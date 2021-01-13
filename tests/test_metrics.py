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
import numpy as np
from pymoo.factory import get_problem
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.algorithm import Algorithm

from arch_opt_exp.experimenter import *
from arch_opt_exp.metrics.performance import *


@pytest.fixture
def problem() -> Problem:
    return get_problem('zdt1')


@pytest.fixture
def problem_3obj() -> Problem:
    return get_problem('DTLZ1')


@pytest.fixture
def algorithm() -> Algorithm:
    return NSGA2(pop_size=20)


def test_spread(problem, problem_3obj, algorithm):
    spread = SpreadMetric()

    with pytest.raises(ValueError):
        Experimenter(problem_3obj, algorithm, n_eval_max=100, metrics=[spread]).run_effectiveness()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[spread])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)
    assert spread.name in result.metrics

    values = result.metrics[spread.name].results()['delta']
    assert len(values) == 50

    result.metrics[spread.name].plot(show=False)


def test_delta_hv(problem, algorithm):
    delta_hv = DeltaHVMetric(problem.pareto_front())
    assert delta_hv.hv_true == pytest.approx(.661, abs=1e-3)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[delta_hv])
    result = exp.run_effectiveness(repeat_idx=0, seed=1)
    assert delta_hv.name in result.metrics

    values = result.metrics[delta_hv.name].results()
    assert 'delta_hv' in values
    assert len(values['delta_hv']) == 50
    assert values['delta_hv'][-1] < values['delta_hv'][0]

    result.metrics[delta_hv.name].plot(show=False)


def test_igd(problem, algorithm):
    igd = IGDMetric(problem.pareto_front())
    igd_plus = IGDPlusMetric(problem.pareto_front())

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[igd, igd_plus])
    result = exp.run_effectiveness(repeat_idx=0, seed=1)
    assert igd.name in result.metrics
    assert igd_plus.name in result.metrics

    values = result.metrics[igd.name].results()['indicator']
    assert values[-1] < values[0]

    result.metrics[igd.name].plot(show=False)
    result.metrics[igd_plus.name].plot(show=False)

    results = exp.run_effectiveness_parallel(n_repeat=5)
    result = ExperimenterResult.aggregate_results(results)
    result.metrics[igd.name].plot(show=False)
    result.metrics[igd_plus.name].plot(show=False)


def test_max_cv(problem, algorithm):
    max_cv = MaxConstraintViolationMetric()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[max_cv])
    result = exp.run_effectiveness(repeat_idx=0)
    assert max_cv.name in result.metrics
    assert np.all(result.metrics[max_cv.name].results()['max_cv'] == 0.)

    constrained_problem = get_problem('C1DTLZ1')
    exp = Experimenter(constrained_problem, algorithm, n_eval_max=1000, metrics=[max_cv])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)
    assert max_cv.name in result.metrics

    values = result.metrics[max_cv.name].results()['max_cv']
    assert len(values) == 50

    result.metrics[max_cv.name].plot(show=False)
