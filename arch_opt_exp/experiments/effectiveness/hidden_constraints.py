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
from arch_opt_exp.experiments.moea_helpers import *

from arch_opt_exp.experiments import runner
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.problems.hierarchical import *
from arch_opt_exp.algorithms.surrogate.validation import *
from arch_opt_exp.algorithms.surrogate.mo.min_var_pf import *
from arch_opt_exp.algorithms.surrogate.func_estimate import *
from arch_opt_exp.algorithms.surrogate.mo.enhanced_poi import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *
from arch_opt_exp.algorithms.surrogate.hidden_constraints import *

from arch_opt_exp.surrogates.smt_models.smt_krg import *
from arch_opt_exp.surrogates.smt_models.smt_rbf import *


def run_compare_hc_strategies(do_run=True, return_exp=False):
    problem = HCMOHierarchicalTestProblem()
    return run_compare(problem, 'compare_hc', do_run=do_run, return_exp=return_exp)


def run_nsga2(do_run=True, return_exp=False):
    problem = HCMOHierarchicalTestProblem()
    results_key = 'compare_hc_nsga2'

    pop_size = 5*problem.n_var
    nsga2 = get_algo(problem, n_init=pop_size, n_offspring=pop_size)
    n_eval_max = 4000

    metrics, plot_metric_values = get_metrics(problem, include_loo_cv=False)

    algorithms = [nsga2]
    algorithm_names = ['NSGA2']
    return run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_eval_max=n_eval_max,
               do_run=do_run, return_exp=return_exp)


def run_compare(problem: Problem, results_key, do_run=True, return_exp=False):
    metrics, plot_metric_values = get_metrics(problem, include_loo_cv=False)

    n_init = 5*problem.n_var
    n_term = 100

    n_infill = 1

    nsga2 = get_algo(problem, n_init=n_init)
    nsga2_name = 'NSGA2'

    rbf_sm = SMTRBFSurrogateModel(d0=1., deg=-1, reg=1e-10)
    krg_sm = SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, theta0=1.)
    sb_infills = [
        lambda hc: (SurrogateBasedInfill(infill=FunctionEstimateInfill(), hc_strategy=hc, surrogate_model=rbf_sm,
                                         termination=n_term, verbose=True), 'RBF(y)'),
        lambda hc: (SurrogateBasedInfill(infill=FunctionEstimateInfill(), hc_strategy=hc, surrogate_model=rbf_sm,
                                         termination=n_term, verbose=True), 'RBF(y-Dist)'),
        lambda hc: (SurrogateBasedInfill(infill=FunctionEstimatePoFInfill(), surrogate_model=krg_sm,
                                         termination=n_term, verbose=True), 'Krg(y)'),
        lambda hc: (SurrogateBasedInfill(infill=MinimumPOIInfill(), surrogate_model=krg_sm,
                                         termination=n_term, verbose=True), 'Krg(mpoi)'),
        lambda hc: (SurrogateBasedInfill(infill=MinVariancePFInfill(), surrogate_model=krg_sm,
                                         termination=n_term, verbose=True), 'Krg(mvpf)'),
    ]

    hc_strategies = [
        (lambda: MaxValueHCStrategy(), 'max'),

        (lambda: AvoidHCStrategy(d_avoid=.05), 'd005'),
        # (lambda: AvoidHCStrategy(d_avoid=.20), 'd020'),
        (lambda: AvoidHCStrategy(d_avoid=.50), 'd050'),
        (lambda: AvoidHCStrategy(d_avoid=2.), 'd200'),

        # (lambda: PredictHCStrategy(rbf_sm, hc_predict_max=.2), 'pred20'),
        # (lambda: PredictHCStrategy(rbf_sm, hc_predict_max=.5), 'pred50'),
        # (lambda: PredictHCStrategy(rbf_sm, hc_predict_max=.8), 'pred80'),
    ]

    algorithms = [nsga2]
    algorithm_names = [nsga2_name]
    for sb_infill_factory in sb_infills:
        for hc_factory, hc_strat_name in hc_strategies:
            sb_infill, sbo_name = sb_infill_factory(hc_factory())
            sbo_algo = sb_infill.algorithm(infill_size=n_infill, init_size=n_init)

            algorithms.append(sbo_algo)
            algorithm_names.append('%s+%s' % (sbo_name, hc_strat_name))

    return run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values,
               do_run=do_run, return_exp=return_exp)


def get_metrics(problem: Problem, include_loo_cv=True, include_spread=True):
    pf = problem.pareto_front()
    metrics = [
        DeltaHVMetric(pf),
        IGDMetric(pf),
        SpreadMetric(),
        MaxConstraintViolationMetric(),
        SurrogateQualityMetric(include_loo_cv=include_loo_cv, n_loo_cv=4),
        TrainingMetric(),
        InfillMetric(),
    ]
    plot_metric_values = {
        'delta_hv': ['delta_hv'],
        'IGD': None,
        'spread': ['delta'],
        'max_cv': None,
        'sm_quality': ['rmse', 'loo_cv'] if include_loo_cv else ['rmse'],
        'training': ['n_train', 'n_samples', 'time_train'],
        'infill': ['time_infill'],
    }

    if not include_spread:
        metrics.pop(2)
        del plot_metric_values['spread']

    return metrics, plot_metric_values


def run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=8, n_eval_max=400,
        do_run=True, return_exp=False):
    runner.set_results_folder(results_key)
    exp = runner.get_experimenters(problem, algorithms, metrics, n_eval_max=n_eval_max, algorithm_names=algorithm_names)
    if return_exp:
        return exp

    if do_run:
        runner.run_effectiveness_multi(exp, n_repeat=n_repeat)
    runner.plot_effectiveness_results(exp, plot_metric_values=plot_metric_values, save=True, show=False)


if __name__ == '__main__':
    # run_nsga2(
    #     # do_run=False,
    # )
    run_compare_hc_strategies(
        # do_run=False,
    )
