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
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.algorithm import Algorithm
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from arch_opt_exp.experimenter import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.problems.discretization import *


@pytest.fixture
def problem() -> Problem:
    return get_problem('zdt1')


@pytest.fixture
def algorithm() -> Algorithm:
    return NSGA2(pop_size=100)


def test_mixed_int_problem(problem, algorithm):
    mixed_int_problem = MixedIntProblem(problem, n_choices=11, n_vars_mixed_int=15)

    hv_metric = DeltaHVMetric(mixed_int_problem.pareto_front())
    exp = Experimenter(mixed_int_problem, algorithm, n_eval_max=2000, metrics=[hv_metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)

    hv_values = result.metrics[hv_metric.name].results()['delta_hv']
    assert hv_values[-1] < hv_values[0]
    # result.metrics[hv_metric.name].plot()


def test_mixed_int_problem_repair(problem):
    mixed_int_problem = MixedIntProblem(problem, n_choices=11, n_vars_mixed_int=15)

    algorithm = NSGA2(
        pop_size=100,
        repair=mixed_int_problem.get_repair(),
    )

    hv_metric = DeltaHVMetric(mixed_int_problem.pareto_front())
    exp = Experimenter(mixed_int_problem, algorithm, n_eval_max=2000, metrics=[hv_metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)

    hv_values = result.metrics[hv_metric.name].results()['delta_hv']
    assert hv_values[-1] < hv_values[0]
    # result.metrics[hv_metric.name].plot()


def test_mixed_int_problem_masked(problem):
    mixed_int_problem = MixedIntProblem(problem, n_choices=11, n_vars_mixed_int=15)

    sampling = MixedVariableSampling(mixed_int_problem.mask, {
        'real': get_sampling('real_random'),
        'int': get_sampling('int_random'),
    })

    crossover = MixedVariableCrossover(mixed_int_problem.mask, {
        'real': get_crossover('real_sbx', prob=.9, eta=3.),
        'int': get_crossover('int_ux', prob=.9),
    })

    mutation = MixedVariableMutation(mixed_int_problem.mask, {
        'real': get_mutation('real_pm', eta=3.),
        'int': get_mutation('bin_bitflip'),
    })

    algorithm = NSGA2(
        pop_size=100,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        repair=mixed_int_problem.get_repair(),
    )

    hv_metric = DeltaHVMetric(mixed_int_problem.pareto_front())
    exp = Experimenter(mixed_int_problem, algorithm, n_eval_max=2000, metrics=[hv_metric])
    result = exp.run_effectiveness(repeat_idx=0, seed=0)

    hv_values = result.metrics[hv_metric.name].results()['delta_hv']
    assert hv_values[-1] < hv_values[0]
    # result.metrics[hv_metric.name].plot()
