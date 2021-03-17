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

from pymoo.algorithms.nsga2 import NSGA2
from arch_opt_exp.experiments import runner
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.problems.hierarchical import *
from arch_opt_exp.experiments.moea_helpers import *


def tune_pop_offspring_size(do_run=True):
    problem, metrics, plot_metric_values = get_problem()

    # results_key, pop_sizes, n_offsprings = 'tune_moea', [50, 100, 200], [25, 50, 100]
    results_key, pop_sizes, n_offsprings = 'tune_moea2', [50, 100], [10, 25, 50, 100]

    algorithms = [NSGA2(pop_size=pop_size, n_offsprings=n_off)
                  for pop_size in pop_sizes for n_off in n_offsprings if n_off <= pop_size]
    algorithm_names = [('NSGA2(%d, %d)' % (algo.pop_size, algo.n_offsprings)) for algo in algorithms]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, do_run=do_run)


def tune_evolutionary_ops(do_run=True):
    problem, metrics, plot_metric_values = get_problem()

    # results_key = 'tune_moea_ops'
    # algo = get_algo(problem, add_ops=False)
    # kwargs = {'pop_size': algo.pop_size, 'n_offsprings': algo.n_offsprings, 'repair': algo.repair}
    # ev_ops = get_evolutionary_ops(problem)
    # algorithms = [
    #     algo,
    #     NSGA2(sampling=get_sampling('real_lhs'), **kwargs),
    #     NSGA2(**{**kwargs, **get_evolutionary_ops(problem)}),
    #     NSGA2(sampling=ev_ops['sampling'], **kwargs),
    #     NSGA2(mutation=ev_ops['mutation'], **kwargs),
    #     NSGA2(crossover=ev_ops['crossover'], **kwargs),
    # ]
    # algorithm_names = ['NSGA2', 'NSGA2_lhs', 'NSGA2_MI', 'NSGA2_MI_sam', 'NSGA2_MI_mut', 'NSGA2_MI_crs']

    results_key = 'tune_moea_ops2'
    algo = get_algo(problem, add_ops=False)
    kwargs = {'pop_size': algo.pop_size, 'n_offsprings': algo.n_offsprings, 'repair': algo.repair}
    cp_values = [.6, .8, .9, .95]
    eta_values = [3, 15, 30]
    algorithms = [
        NSGA2(**{**kwargs, **get_evolutionary_ops(problem, crs_rp=cp, crs_ip=cp, crs_re=eta, mut_re=eta, mut_ie=eta)})
        for cp in cp_values for eta in eta_values
    ]
    algorithm_names = ['NSGA2_MI_%02d_%d' % (cp*100, eta) for cp in cp_values for eta in eta_values]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, do_run=do_run)


def get_problem():
    problem = MOHierarchicalTestProblem()

    pf = problem.pareto_front()
    metrics = [
        DeltaHVMetric(pf),
        IGDMetric(pf),
        SpreadMetric(),
        MaxConstraintViolationMetric(),
    ]
    plot_metric_values = {
        'delta_hv': ['delta_hv'],
        'IGD': None,
        'spread': ['delta'],
        'max_cv': ['max_cv'],
    }
    return problem, metrics, plot_metric_values


def run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=12, do_run=True):
    runner.set_results_folder(results_key)
    exp = runner.get_experimenters(problem, algorithms, metrics, n_eval_max=5000, algorithm_names=algorithm_names)

    if do_run:
        runner.run_effectiveness_multi(exp, n_repeat=n_repeat)
    runner.plot_effectiveness_results(exp, plot_metric_values=plot_metric_values, save=True, show=False)


if __name__ == '__main__':
    """
    The goal of this script is to tune "hyperparameters" of the NSGA2 algorithm
    to make sure we are effective at finding the Pareto front.
    """
    # tune_pop_offspring_size(
    #     # run=False,
    # )
    tune_evolutionary_ops(
        do_run=False,
    )
