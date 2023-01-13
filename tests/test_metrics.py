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

import pytest
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.algorithm import Algorithm
from pymoo.factory import get_problem

from arch_opt_exp.experiments.experimenter import *
from arch_opt_exp.metrics.filters import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.metrics.convergence import *
from arch_opt_exp.metrics.termination import *


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
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert spread.name in result.metrics

    values = result.metrics[spread.name].results()['delta']
    assert len(values) == 50

    # result.metrics[spread.name].plot(show=False)

    spread_termination = SpreadTermination(limit=1e-3)
    eff_res = exp.run_efficiency(spread_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    assert len(eff_res.termination._get_check_values()) == len(eff_res.history)
    eff_res.termination.plot(show=False)

    spread_termination_n = SpreadTermination(limit=1e-3, n_eval_check=40)
    eff_res2 = exp.run_efficiency(spread_termination_n, repeat_idx=0)
    assert len(eff_res2.termination._get_check_values()) < 50
    eff_res2.termination.plot(show=False)


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

    # result.metrics[delta_hv.name].plot(show=False)


def test_igd(problem, algorithm):
    igd = IGDMetric(problem.pareto_front())
    igd_plus = IGDPlusMetric(problem.pareto_front())

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[igd, igd_plus])
    result = exp.run_effectiveness(repeat_idx=0, seed=1, keep_history=True)
    assert igd.name in result.metrics
    assert igd_plus.name in result.metrics

    values = result.metrics[igd.name].results()['indicator']
    assert values[-1] < values[0]

    # result.metrics[igd.name].plot(show=False)
    # result.metrics[igd_plus.name].plot(show=False)

    exp.run_effectiveness_parallel(n_repeat=5, keep_history=True)
    results = exp.get_effectiveness_results()
    result = ExperimenterResult.aggregate_results(results)
    result.metrics[igd.name].plot(show=False)
    result.metrics[igd_plus.name].plot(show=False)


def test_max_cv(problem, algorithm):
    max_cv = MaxConstraintViolationMetric()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[max_cv])
    result = exp.run_effectiveness(repeat_idx=0, keep_history=True)
    assert max_cv.name in result.metrics
    assert np.all(result.metrics[max_cv.name].results()['max_cv'] == 0.)

    constrained_problem = get_problem('C1DTLZ1')
    exp = Experimenter(constrained_problem, algorithm, n_eval_max=1000, metrics=[max_cv])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert max_cv.name in result.metrics

    values = result.metrics[max_cv.name].results()['max_cv']
    assert len(values) == 50

    # result.metrics[max_cv.name].plot(show=False)


def test_nr_evaluations_metric(problem, algorithm):
    nr_eval = NrEvaluationsMetric()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[nr_eval])
    result = exp.run_effectiveness(repeat_idx=0, seed=1, keep_history=True)
    assert nr_eval.name in result.metrics

    values = result.metrics[nr_eval.name].results()
    assert 'n_eval' in values
    assert len(values['n_eval']) == 50
    assert values['n_eval'][-1] > values['n_eval'][0]
    assert values['n_eval'][-1] == 1000

    # result.metrics[nr_eval.name].plot(show=False)


def test_hv(problem, algorithm):
    hv = HVMetric()
    filtered_hv = MovingAverageFilter(HVMetric(), n=5)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[hv, filtered_hv])
    result = exp.run_effectiveness(repeat_idx=0, seed=1, keep_history=True)
    assert hv.name in result.metrics

    values = result.metrics[hv.name].results()['hv']
    assert values[-1] > values[0]

    # result.metrics[hv.name].plot(show=False)

    hv_termination = HVTermination(limit=5e-2)
    eff_res = exp.run_efficiency(hv_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)

    algorithm2 = NSGA2(pop_size=30)
    exp2 = Experimenter(problem, algorithm2, n_eval_max=1000, metrics=[hv, filtered_hv])
    result2 = exp2.run_effectiveness(repeat_idx=0, seed=2, keep_history=True)

    ExperimenterResult.plot_compare_metrics([result, result2], hv.name, titles=['20', '30'], show=False)
    ExperimenterResult.plot_compare_metrics([result, result2], filtered_hv.name, titles=['20', '30'], show=False)


def test_distance_metrics(problem, algorithm):
    metrics = [
        GDConvergenceMetric(),
        IGDConvergenceMetric(),
    ]

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=metrics)
    result = exp.run_effectiveness(repeat_idx=0, seed=1, keep_history=True)
    for metric in metrics:
        assert metric.name in result.metrics
        values = result.metrics[metric.name].results()['d']
        assert values[-1] > values[0]
        # result.metrics[metric.name].plot(show=False)

    # gd_termination = GDTermination(limit=1e-4)
    # eff_res = exp.run_efficiency(gd_termination, repeat_idx=0)
    # assert len(eff_res.history) < 50
    # # eff_res.termination.plot(show=False)

    # igd_termination = IGDTermination(limit=1e-4)
    # eff_res = exp.run_efficiency(igd_termination, repeat_idx=0)
    # assert len(eff_res.history) < 50
    # # eff_res.termination.plot(show=False)


def test_crowding_distance_metric(problem, algorithm):
    cd = CrowdingDistanceMetric()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cd])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert cd.name in result.metrics
    max_values = result.metrics[cd.name].results()['max']
    assert max_values[-1] < max_values[0]

    # result.metrics[cd.name].plot(show=False)

    mcd_termination = MCDTermination(limit=7e-3)
    eff_res = exp.run_efficiency(mcd_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)


# def test_crowding_distance_metric_non_nsga2(problem):
#     cd = CrowdingDistanceMetric()
#
#     algorithm = MOEAD(get_reference_directions('das-dennis', 3, n_partitions=12), n_neighbors=15, decomposition='pbi',
#                       prob_neighbor_mating=.7)
#     exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cd])
#     result = exp.run_effectiveness(repeat_idx=0, seed=0)
#     assert cd.name in result.metrics
#     max_values = result.metrics[cd.name].results()['max']
#     assert max_values[-1] < max_values[0]
#
#     # result.metrics[cd.name].plot(show=False)


def test_steady_performance_indicator(problem, algorithm):
    spi = SteadyPerformanceIndicator(n_last_steps=10)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[spi])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert spi.name in result.metrics

    max_values = result.metrics[spi.name].results()['std']
    assert len(max_values) == 50
    assert np.isnan(max_values[0])
    assert not np.isnan(max_values[9])
    assert max_values[-1] < max_values[9]

    # result.metrics[spi.name].plot(show=False)

    spi_termination = SPITermination(n=10, limit=.03)
    eff_res = exp.run_efficiency(spi_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)


def test_fh_indicator(problem, algorithm):
    fhi = FitnessHomogeneityIndicator()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[fhi])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert fhi.name in result.metrics

    max_values = result.metrics[fhi.name].results()['fhi']
    assert len(max_values) == 50
    assert max_values[-1] > max_values[0]

    # result.metrics[fhi.name].plot(show=False)

    fhi_termination = FHITermination(limit=6e-4)
    eff_res = exp.run_efficiency(fhi_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)


def test_consolidation_ratio_metric(problem, algorithm):
    cr = ConsolidationRatioMetric(n_delta=1)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cr])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert cr.name in result.metrics

    max_values = result.metrics[cr.name].results()['cr']
    assert len(max_values) == 50
    assert max_values[-1] > max_values[0]

    # result.metrics[cr.name].plot(show=False)

    cr_termination = CRTermination(limit=.68)
    eff_res = exp.run_efficiency(cr_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)


def test_mutual_domination_rate_metric(problem, algorithm):
    mdr = MutualDominationRateMetric()

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[mdr])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert mdr.name in result.metrics

    max_values = result.metrics[mdr.name].results()['mdr']
    assert len(max_values) == 50
    assert max_values[-1] < max_values[0]

    # result.metrics[mdr.name].plot(show=False)

    mdr_termination = MDRTermination(limit=.35)
    eff_res = exp.run_efficiency(mdr_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)

    mgbm_termination = MGBMTermination(limit=.3)
    eff_res = exp.run_efficiency(mgbm_termination, repeat_idx=0)
    assert len(eff_res.history) < 50
    # eff_res.termination.plot(show=False)


def test_moving_average_filter(problem, algorithm):
    igd_filtered = MovingAverageFilter(IGDConvergenceMetric(), n=5)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[igd_filtered])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert igd_filtered.name in result.metrics
    values = result.metrics[igd_filtered.name].results()['d']
    assert np.isnan(values[0])
    assert not np.isnan(values[4])
    assert values[-1] > values[4]

    # result.metrics[igd_filtered.name].plot(show=False)


def test_exp_moving_average_filter(problem, algorithm):
    igd_filtered = ExpMovingAverageFilter(IGDConvergenceMetric(), n=3)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[igd_filtered])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert igd_filtered.name in result.metrics
    values = result.metrics[igd_filtered.name].results()['d']
    assert values[-1] > values[0]

    # result.metrics[igd_filtered.name].plot(show=False)


def test_kalman_filter(problem, algorithm):
    igd_filtered = KalmanFilter(IGDConvergenceMetric(), r=.1, q=.1)

    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[igd_filtered])
    result = exp.run_effectiveness(repeat_idx=0, seed=0, keep_history=True)
    assert igd_filtered.name in result.metrics
    values = result.metrics[igd_filtered.name].results()['d']
    assert values[-1] > values[0]

    # result.metrics[igd_filtered.name].plot(show=False)
