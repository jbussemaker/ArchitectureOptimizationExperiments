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
from pymoo.model.population import Population
from pymoo.performance_indicator.gd import GD
from pymoo.performance_indicator.igd import IGD

from arch_opt_exp.metrics import *
from arch_opt_exp.experimenter import *


@pytest.fixture
def problem() -> Problem:
    return get_problem('zdt1')


@pytest.fixture
def algorithm() -> Algorithm:
    return NSGA2(pop_size=20)


def test_instantiate(problem, algorithm):
    exp = Experimenter(problem, algorithm, n_eval_max=2000)
    assert exp.algorithm_name

    assert exp._get_effectiveness_result_path(repeat_idx=0)


def test_run_effectiveness(problem, algorithm):
    assert problem.n_var == 30
    exp = Experimenter(problem, algorithm, n_eval_max=2000)

    res = exp.run_effectiveness(repeat_idx=0, seed=0)
    assert isinstance(res, ExperimenterResult)
    assert res.problem.name() == problem.name()
    assert res.exec_time > 0

    assert len(res.pop) == 20
    assert len(res.history) == 100

    assert res.X.shape == (20, 30)
    assert np.min(res.X) >= 0
    assert np.max(res.X) <= 1

    assert res.F.shape == (20, 2)
    assert np.max(res.F) <= 1.72
    assert np.min(res.F) <= 4e-7

    assert res.G.shape == (20,)
    assert res.CV.shape == (20, 1)  # Constraint violation
    assert np.all(res.CV == 0)

    assert isinstance(res.history[0], algorithm.__class__)
    assert isinstance(res.history[-1].pop, Population)
    assert np.all(res.history[-1].pop.get('X') == res.X)
    assert np.all(res.history[-1].pop.get('F') == res.F)
    assert np.all(res.history[-1].pop.get('G') == res.G)
    assert np.all(res.history[-1].pop.get('CV') == res.CV)

    res2 = exp.get_effectiveness_result(repeat_idx=0)
    assert res2 is not res
    assert np.all(res.X == res2.X)

    assert exp.get_effectiveness_result(repeat_idx=1) is None


def test_run_effectiveness_parallel(problem, algorithm):
    exp = Experimenter(problem, algorithm, n_eval_max=2000)

    result = exp.run_effectiveness_parallel(n_repeat=4)
    assert len(result) == 4

    for i in range(6):
        res = exp.get_effectiveness_result(repeat_idx=i)
        if i >= 4:
            assert res is None
            continue

        assert isinstance(res, ExperimenterResult)
        assert np.all(res.X == result[i].X)


def test_metrics(problem, algorithm):
    gd_metric = IndicatorMetric(GD(problem.pareto_front()))
    igd_metric = IndicatorMetric(IGD(problem.pareto_front()))

    exp = Experimenter(problem, algorithm, n_eval_max=2000, metrics=[gd_metric, igd_metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)
    assert isinstance(result, ExperimenterResult)

    assert len(result.metrics) == 2
    met_names = list(result.metrics.keys())
    assert gd_metric.name in result.metrics
    assert igd_metric.name in result.metrics

    assert len(result.metrics[gd_metric.name].results()['indicator']) == 100

    result = exp.get_effectiveness_result(repeat_idx=0)
    assert isinstance(result, ExperimenterResult)
    assert len(result.metrics) == 2
    for name, metric in result.metrics.items():
        assert name in met_names

        assert isinstance(metric, IndicatorMetric)
        assert isinstance(metric.indicator, (GD, IGD))

        met_results = metric.results()
        assert 'indicator' in met_results
        assert len(met_results['indicator']) == 100
        assert met_results['indicator'][0] > met_results['indicator'][-1]


def test_run_efficiency(problem, algorithm):
    igd_metric = IndicatorMetric(IGD(problem.pareto_front()))
    assert igd_metric.name == 'IGD'

    exp = Experimenter(problem, algorithm, n_eval_max=2000, metrics=[igd_metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)
    assert exp.get_effectiveness_result(repeat_idx=0) is not None

    assert len(result.history) == 100
    assert result.metrics['IGD'].results()['indicator'][-1] == pytest.approx(.347, abs=1e-2)

    metric_termination_converged = MetricTermination(igd_metric, lower_limit=.6)
    assert exp.run_efficiency(metric_termination_converged, repeat_idx=1) is None

    result_converged = exp.run_efficiency(metric_termination_converged, repeat_idx=0)
    assert result_converged.metric_converged
    assert len(result_converged.history) == 54
    assert result_converged.metrics['IGD'].results()['indicator'][-1] == pytest.approx(.595, abs=1e-2)

    assert isinstance(result_converged.termination, MetricTermination)
    assert result_converged.termination is not metric_termination_converged
    assert len(result_converged.termination.metric.results()['indicator']) == 54

    result_pkl = exp.get_efficiency_result(metric_termination_converged, repeat_idx=0)
    assert result_pkl.metric_converged
    assert result_pkl.metrics['IGD'].results()['indicator'][-1] == pytest.approx(.595, abs=1e-2)
    assert exp.get_efficiency_result(metric_termination_converged, repeat_idx=1) is None

    results = exp.run_efficiency_repeated(metric_termination_converged)
    assert len(results) == 1

    metric_termination_not_converged = MetricTermination(igd_metric, lower_limit=.3)
    result_not_converged = exp.run_efficiency(metric_termination_not_converged, repeat_idx=0)
    assert not result_not_converged.metric_converged
    assert len(result_not_converged.history) == 100
    assert result_not_converged.metrics['IGD'].results()['indicator'][-1] == pytest.approx(.347, abs=1e-2)
    assert len(result_not_converged.termination.metric.results()['indicator']) == 100
