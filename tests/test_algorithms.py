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
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.util.dominator import Dominator
from pymoo.problems.many.dtlz import DTLZ2
from pymoo.factory import get_problem, get_reference_directions
from pymoo.operators.selection.random_selection import RandomSelection

from arch_opt_exp.experimenter import *
from arch_opt_exp.algorithms.mdr import *
from arch_opt_exp.metrics.filters import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.metrics.convergence import *
from arch_opt_exp.algorithms.infill_based import *
from arch_opt_exp.algorithms.random_search import *
from arch_opt_exp.algorithms.hill_climbing import *


@pytest.fixture
def problem() -> Problem:
    return get_problem('zdt1')


def test_random_search(problem):
    cr_metric = ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5)
    igd_metric = IGDMetric(problem.pareto_front())

    algorithm = RandomSearchAlgorithm(pop_size=100)
    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cr_metric, igd_metric])
    res = exp.run_effectiveness(repeat_idx=0)

    cr = res.metrics[cr_metric.name].results()['cr']
    assert cr[-1] > cr[0]

    igd = res.metrics[igd_metric.name].results()['indicator']
    assert igd[0] != 0


def test_hill_climbing(problem):
    cr_metric = ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5)
    igd_metric = IGDMetric(problem.pareto_front())

    algorithm = HillClimbingAlgorithm(pop_size=100)
    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cr_metric, igd_metric])
    res = exp.run_effectiveness(repeat_idx=0)

    cr = res.metrics[cr_metric.name].results()['cr']
    assert cr[-1] > cr[0]

    igd = res.metrics[igd_metric.name].results()['indicator']
    assert igd[0] != 0


def test_simulated_annealing(problem):
    cr_metric = ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5)
    igd_metric = IGDMetric(problem.pareto_front())

    algorithm = SimulatedAnnealingAlgorithm(pop_size=100)
    exp = Experimenter(problem, algorithm, n_eval_max=1000, metrics=[cr_metric, igd_metric])
    res = exp.run_effectiveness(repeat_idx=0)

    cr = res.metrics[cr_metric.name].results()['cr']
    assert cr[-1] > cr[0]

    igd = res.metrics[igd_metric.name].results()['indicator']
    assert igd[0] != 0


def test_mdr():
    mdr = MultipleDominanceRelations([
        ((MetricType.G, 0), (MetricType.G, 1)),  # First check dominance among the constraints
        ((MetricType.F, 0), (MetricType.F, 1)),  # Then among the first two objectives
        ((MetricType.F, 2), (MetricType.F, 3)),  # THen among the last two objectives
    ])

    assert mdr.get_relation(
        a=np.array([0, 0, 0, 0]), b=np.array([0, 0, 0, 0]), ga=np.array([0, 0]), gb=np.array([0, 0])) == 0

    assert mdr.get_relation(
        a=np.array([0, 0, 0, 0]), b=np.array([0, 0, 0, 0]), ga=np.array([-1, 0]), gb=np.array([0, -1])) == 0

    assert mdr.get_relation(
        a=np.array([0, 0, 0, 0]), b=np.array([0, 0, 0, 0]), ga=np.array([1, 0]), gb=np.array([0, 0])) == -1

    assert mdr.get_relation(
        a=np.array([0, 0, 0, 0]), b=np.array([0, 0, 0, 0]), ga=np.array([1, 0]), gb=np.array([0, 1])) == 0

    assert mdr.get_relation(
        a=np.array([0, 0, 0, 0]), b=np.array([0, 0, 0, 0]), ga=np.array([1, 0]), gb=np.array([2, 1])) == 1

    assert mdr.get_relation(
        a=np.array([0, 0, 0, 0]), b=np.array([1, 1, 0, 0]), ga=np.array([0, 0]), gb=np.array([0, 0])) == 1

    assert mdr.get_relation(
        a=np.array([1, 1, 0, 0]), b=np.array([0, 0, 0, 0]), ga=np.array([0, 0]), gb=np.array([0, 0])) == -1

    assert mdr.get_relation(
        a=np.array([1, 0, 0, 0]), b=np.array([0, 1, 0, 0]), ga=np.array([0, 0]), gb=np.array([0, 0])) == 0

    assert mdr.get_relation(
        a=np.array([1, 0, 0, 0]), b=np.array([0, 1, 1, 1]), ga=np.array([0, 0]), gb=np.array([0, 0])) == 1

    assert mdr.get_relation(
        a=np.array([1, 0, 1, 1]), b=np.array([0, 1, 0, 0]), ga=np.array([0, 0]), gb=np.array([0, 0])) == -1

    assert mdr.get_relation(
        a=np.array([1, 0, 1, 0]), b=np.array([0, 1, 0, 1]), ga=np.array([0, 0]), gb=np.array([0, 0])) == 0

    assert mdr.get_relation(
        a=np.array([0, 0, 1, 1]), b=np.array([1, 1, 0, 0]), ga=np.array([0, 0]), gb=np.array([0, 0])) == 1

    # Regular behavior: first check the constraint violation, then the objectives
    mdr_reg = MultipleDominanceRelations([
        ((MetricType.CV, 0),),
        ((MetricType.F, 0), (MetricType.F, 1), (MetricType.F, 2), (MetricType.F, 3)),
    ])

    for _ in range(10):
        f = np.random.random((10, 4))
        g = np.random.random((10, 2))-.5

        m_nds = Dominator.calc_domination_matrix_loop(f, g)
        m_mdr_reg = mdr_reg.calc_domination_matrix_loop(f, g)
        assert np.all(m_nds == m_mdr_reg)

        m_mdr = mdr.calc_domination_matrix_loop(f, g)
        assert not np.all(m_nds == m_mdr)

    # Check the direct matrix methos
    mdr = MultipleDominanceRelations([((MetricType.F, 0), (MetricType.F, 1))])
    f = np.random.random((20, 2))
    m_nds = Dominator.calc_domination_matrix(f)
    m_mdr = mdr.calc_domination_matrix(f)
    assert np.all(m_nds == m_mdr)


def test_inject_dominator():
    mdr = MultipleDominanceRelations([
        ((MetricType.G, 0), (MetricType.G, 1)),
        ((MetricType.F, 0), (MetricType.F, 1)),
        ((MetricType.F, 2), (MetricType.F, 3)),
    ])

    f = np.random.random((20, 4))
    g = np.random.random((20, 2)) - .5
    m_nds = Dominator.calc_domination_matrix_loop(f, g)

    with inject_dominator(mdr):
        m_mdr = Dominator.calc_domination_matrix_loop(f, g)

    assert not np.all(m_nds == m_mdr)


def test_nsga2_mdr():
    problem = DTLZ2(n_var=13, n_obj=4)
    ref_dirs = get_reference_directions('das-dennis', 4, n_partitions=12)
    metric = DeltaHVMetric(problem.pareto_front(ref_dirs))

    mdr = MultipleDominanceRelations([
        ((MetricType.F, 0), (MetricType.F, 1)),
        ((MetricType.F, 2), (MetricType.F, 3)),
    ])

    nsga2 = NSGA2(pop_size=50)
    exp = Experimenter(problem, nsga2, n_eval_max=1000, metrics=[metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)

    nsga2_mdr = NSGA2MultiDR(mdr, pop_size=50)
    exp_mdr = Experimenter(problem, nsga2_mdr, n_eval_max=1000, metrics=[metric])
    result_mdr = exp_mdr.run_effectiveness(repeat_idx=0, seed=0)

    values = result.metrics[metric.name].values['delta_hv']
    values_mdr = result_mdr.metrics[metric.name].values['delta_hv']
    assert values[0] == values_mdr[0]
    assert values[-1] != values_mdr[-1]

    ExperimenterResult.plot_compare_metrics(
        [result, result_mdr], metric.name, titles=['MDR', 'ND'], plot_value_names=['delta_hv'], show=False)


def test_infill_based_algorithm(problem):
    mating_infill = NSGA2().mating
    mating_infill.selection = RandomSelection()
    algorithm = InfillBasedAlgorithm(mating_infill, infill_size=5, init_size=20)

    metric = DeltaHVMetric(problem.pareto_front())
    exp = Experimenter(problem, algorithm, n_eval_max=500, metrics=[metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)

    values = result.metrics[metric.name].values['delta_hv']
    assert len(values) == 97
    # assert values[-1] < values[0]
