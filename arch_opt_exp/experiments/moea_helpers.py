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

from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from arch_opt_exp.problems.discretization import *
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

__all__ = ['get_algo', 'get_evolutionary_ops']


def get_algo(problem: Problem, add_ops=True):
    kwargs = get_evolutionary_ops(problem) if add_ops else {}
    repair = MixedIntProblemHelper.get_repair(problem)
    return NSGA2(pop_size=100, n_offsprings=25, repair=repair, **kwargs)

def get_evolutionary_ops(problem: Problem, crs_rp=.8, crs_ip=.8, crs_re=3, mut_re=3, mut_ie=3):
    is_discrete_mask = MixedIntProblemHelper.get_is_discrete_mask(problem)
    return {
        'sampling': MixedVariableSampling(is_discrete_mask, {
            False: get_sampling('real_lhs'),
            True: get_sampling('int_lhs'),
        }),
        'crossover': MixedVariableCrossover(is_discrete_mask, {
            False: get_crossover('real_sbx', prob=crs_rp, eta=crs_re),
            True: get_crossover('int_ux', prob=crs_ip),
        }),
        'mutation': MixedVariableMutation(is_discrete_mask, {
            False: get_mutation('real_pm', eta=mut_re),
            True: get_mutation('int_pm', eta=mut_ie),
        }),
    }
