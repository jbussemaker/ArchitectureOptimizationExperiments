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

from arch_opt_exp.experiments import runner
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.problems.hierarchical import *
from arch_opt_exp.algorithms.surrogate.validation import *
from arch_opt_exp.algorithms.surrogate.mo.min_var_pf import *
from arch_opt_exp.algorithms.surrogate.mo.enhanced_poi import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
from arch_opt_exp.algorithms.surrogate.so_probabilities import *

from arch_opt_exp.surrogates.smt_models.smt_krg import *
from arch_opt_exp.surrogates.sklearn_models.gp import *
from arch_opt_exp.surrogates.sklearn_models.distance_base import *
from arch_opt_exp.surrogates.sklearn_models.mixed_int_dist import *
from arch_opt_exp.surrogates.sklearn_models.hierarchical_dist import *
from arch_opt_exp.surrogates.sklearn_models.hierarchical_decomp_kernel import *


def select_kriging_doe_size(do_run=True):
    problem, metrics, plot_metric_values = get_problem(include_loo_cv=False)

    n_infill = 1

    def _get_algo(n_init):
        infill = ModMinimumPOIInfill() if n_infill > 1 else MinimumPOIInfill()
        return SurrogateBasedInfill(infill=infill, surrogate_model=sm, termination=100, verbose=True)\
            .algorithm(infill_size=n_infill, init_size=n_init)

    # sm, suf = SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, theta0=1.), 'cont_relax'
    sm, suf = SMTKrigingSurrogateModel(auto_wrap_mixed_int=True, theta0=1.), 'dummy'
    # sm, suf = SKLearnGPSurrogateModel(kernel=HammingDistance().kernel(), alpha=1e-6, int_as_discrete=True), 'ham'
    # sm, suf = SKLearnGPSurrogateModel(kernel=GowerDistance().kernel(), alpha=1e-6, int_as_discrete=True), 'gow'

    results_key = 'eff_select_kriging_doe_size_%s_%d' % (suf, n_infill)
    n_init_test = [25, 50, 100, 150, 200]
    algorithms = [_get_algo(n) for n in n_init_test]
    algorithm_names = [('SBO(%d)' % n) for n in n_init_test]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=8, n_eval_max=300,
        do_run=do_run)


def select_kriging_surrogate_mi_h_pre(do_run=True):
    problem, metrics, plot_metric_values = get_problem(include_loo_cv=False)

    results_key = 'eff_select_kriging_mi_h_pre'
    n_init, n_infill = 5*problem.n_var, 1

    infills = {
        'mpoi': (MinimumPOIInfill, ModMinimumPOIInfill),
        'ei': (ExpectedImprovementInfill, ExpectedImprovementInfill),
    }

    def _get_algo(sm, infill_key):
        infill = infills[infill_key][0 if n_infill == 1 else 1]()
        return SurrogateBasedInfill(infill=infill, surrogate_model=sm, termination=100, verbose=True)\
            .algorithm(infill_size=n_infill, init_size=n_init)

    smt_kwargs = {'theta0': 1.}
    sklearn_kwargs = {'alpha': 1e-6, 'int_as_discrete': True}
    sms = [
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, **smt_kwargs), 'cont_relax'),
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=True, **smt_kwargs), 'dummy_coding'),
        (SKLearnGPSurrogateModel(kernel=HammingDistance().kernel(), **sklearn_kwargs), 'MI: Ham'),
        # (SKLearnGPSurrogateModel(kernel=GowerDistance().kernel(), **sklearn_kwargs), 'MI: Gow'),
        (SKLearnGPSurrogateModel(kernel=IndefiniteConditionalDistance().kernel(), **sklearn_kwargs), 'MI+H: Ico'),
        (SKLearnGPSurrogateModel(kernel=SPWDecompositionKernel(CompoundSymmetryKernel().kernel()),
                                 **sklearn_kwargs), 'MI+H: SPW+CS'),
    ]
    infill_keys = ['ei', 'mpoi']

    algorithms = [_get_algo(sm, key) for key in infill_keys for sm, _ in sms]
    algorithm_names = [('SBO(%s, %s)' % (name, key.upper())) for key in infill_keys for _, name in sms]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=4, do_run=do_run)


def select_kriging_surrogate(do_run=True):
    problem, metrics, plot_metric_values = get_problem(include_loo_cv=False)

    quick = True
    results_key = 'eff_select_kriging'
    n_init, n_infill = 5*problem.n_var, 1
    # n_init, n_infill = 200, 10

    n_rep = 16
    if quick:
        results_key += '_q'
        n_rep = 4

    def _get_algo(sm):
        infill = MinimumPOIInfill() if n_infill == 1 else ModMinimumPOIInfill()
        return SurrogateBasedInfill(infill=infill, surrogate_model=sm, termination=100, verbose=True)\
            .algorithm(infill_size=n_infill, init_size=n_init)

    smt_kwargs = {'theta0': 1.}
    sklearn_kwargs = {'alpha': 1e-6, 'int_as_discrete': True}
    sms = [
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, **smt_kwargs), 'cont_relax'),
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=True, **smt_kwargs), 'dummy_coding'),

        (SKLearnGPSurrogateModel(kernel=HammingDistance().kernel(), **sklearn_kwargs), 'MI: Ham'),
        (SKLearnGPSurrogateModel(kernel=GowerDistance().kernel(), **sklearn_kwargs), 'MI: Gow'),
        (SKLearnGPSurrogateModel(kernel=SymbolicCovarianceDistance().kernel(), **sklearn_kwargs), 'MI: SC'),
        (SKLearnGPSurrogateModel(kernel=CompoundSymmetryKernel().kernel(), **sklearn_kwargs), 'MI: CS'),
        (SKLearnGPSurrogateModel(kernel=LatentVariablesDistance().kernel(), **sklearn_kwargs), 'MI: LV'),

        (SKLearnGPSurrogateModel(kernel=ArcDistance().kernel(), **sklearn_kwargs), 'MI+H: Arc'),
        (SKLearnGPSurrogateModel(kernel=IndefiniteConditionalDistance().kernel(), **sklearn_kwargs), 'MI+H: Ico'),
        (SKLearnGPSurrogateModel(kernel=ImputationDistance().kernel(), **sklearn_kwargs), 'MI+H: Imp'),
        (SKLearnGPSurrogateModel(kernel=WedgeDistance().kernel(), **sklearn_kwargs), 'MI+H: Wed'),

        (SKLearnGPSurrogateModel(kernel=SPWDecompositionKernel(CompoundSymmetryKernel().kernel()),
                                 **sklearn_kwargs), 'MI+H: SPW+CS'),
        (SKLearnGPSurrogateModel(kernel=SPWDecompositionKernel(LatentVariablesDistance().kernel()),
                                 **sklearn_kwargs), 'MI+H: SPW+LV'),
        (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(CompoundSymmetryKernel().kernel()),
                                 **sklearn_kwargs), 'MI+H: DVW+CS'),
        (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(LatentVariablesDistance().kernel()),
                                 **sklearn_kwargs), 'MI+H: DVW+LV'),

        (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(MixedIntKernel.get_cont_kernel()),
                                 **sklearn_kwargs), 'MI+H: DVW+cr'),
        (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(HammingDistance().kernel()),
                                 **sklearn_kwargs), 'MI+H: DVW+Ham'),
        (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(SymbolicCovarianceDistance().kernel()),
                                 **sklearn_kwargs), 'MI+H: DVW+SC'),
    ]

    algorithms = [_get_algo(sm) for sm, _ in sms]
    algorithm_names = [('SBO(%s)' % name) for _, name in sms]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=n_rep, do_run=do_run)


def select_infill_size(do_run=True):
    problem, metrics, plot_metric_values = get_problem(include_loo_cv=False)

    results_key = 'eff_select_kriging_infill'
    n_init = 5*problem.n_var

    infills = {
        'mpoi': (MinimumPOIInfill, ModMinimumPOIInfill),
        'mvfp': (MinVariancePFInfill, MinVariancePFInfill),
        'ei': (ExpectedImprovementInfill, ExpectedImprovementInfill),
    }

    def _get_algo(sm, n_infill_, infill_key):
        infill = infills[infill_key][0 if n_infill_ == 1 else 1]()
        return SurrogateBasedInfill(infill=infill, surrogate_model=sm, termination=100, verbose=True)\
            .algorithm(infill_size=n_infill_, init_size=n_init)

    smt_kwargs = {'theta0': 1.}
    sms = [
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, **smt_kwargs), 'cont_relax'),
        # (SMTKrigingSurrogateModel(auto_wrap_mixed_int=True, **smt_kwargs), 'dummy_coding'),
    ]
    n_infills = [1, 5, 10, 20, 50]
    infill_keys = ['ei', 'mvfp', 'mpoi']

    algorithms = [_get_algo(sm, n_infill, key) for sm, _ in sms for key in infill_keys for n_infill in n_infills]
    algorithm_names = [('SBO(%s, %s, %d)' % (name, key.upper(), n_infill))
                       for _, name in sms for key in infill_keys for n_infill in n_infills]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, do_run=do_run)


def get_problem(include_loo_cv=True):
    problem = MOHierarchicalTestProblem()

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
        'max_cv': ['max_cv'],
        'sm_quality': ['rmse', 'loo_cv'] if include_loo_cv else ['rmse'],
        'training': ['n_train', 'n_samples', 'time_train'],
        'infill': ['time_infill'],
    }
    return problem, metrics, plot_metric_values


def run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=16, n_eval_max=300,
        do_run=True):
    runner.set_results_folder(results_key)
    exp = runner.get_experimenters(problem, algorithms, metrics, n_eval_max=n_eval_max, algorithm_names=algorithm_names)

    if do_run:
        runner.run_effectiveness_multi(exp, n_repeat=n_repeat)
    runner.plot_effectiveness_results(exp, plot_metric_values=plot_metric_values, save=True, show=False)


if __name__ == '__main__':
    """
    The goal of this script is to answer the question: which Kriging kernel performs best for system architecture
    optimization problems? This is done by comparing different Kriging surrogates on the analytical test problem, using
    the Minimum Probability of Improvement (MPoI) infill criterion.
    """
    # select_kriging_doe_size(
    #     # do_run=False,
    # )
    select_kriging_surrogate_mi_h_pre(
        # do_run=False,
    )
    # select_kriging_surrogate(
    #     # do_run=False,
    # )
    # select_infill_size(
    #     # do_run=False,
    # )
