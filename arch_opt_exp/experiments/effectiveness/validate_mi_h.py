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
from arch_opt_exp.problems.so_mo import *
from arch_opt_exp.experiments import runner
from arch_opt_exp.problems.discrete import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.problems.hierarchical import *
from arch_opt_exp.algorithms.surrogate.validation import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
from arch_opt_exp.algorithms.surrogate.so_probabilities import *

from arch_opt_exp.surrogates.smt_models.smt_krg import *
from arch_opt_exp.surrogates.sklearn_models.gp import *
from arch_opt_exp.surrogates.sklearn_models.mixed_int_dist import *
from arch_opt_exp.surrogates.sklearn_models.hierarchical_dist import *
from arch_opt_exp.surrogates.sklearn_models.hierarchical_decomp_kernel import *


def run_goldstein(do_run=True):
    run_mi_h_algo(HierarchicalGoldsteinProblem(), 'goldstein', 104, 208, do_run=do_run)


def run_rosenbrock(do_run=True):
    run_mi_h_algo(HierarchicalRosenbrockProblem(), 'rosenbrock', 104, 208, do_run=do_run)


def run_munoz_zuniga(do_run=True):
    run_mi_h_algo(MunozZunigaToyProblem(), 'munoz_zuniga', 5, 20, do_run=do_run)


def run_halstrup_4(do_run=True):
    run_mi_h_algo(Halstrup04(), 'halstrup_04', 80, 120, do_run=do_run)


def run_halstrup_4_fc(do_run=True):
    run_mi_h_infill_force_cont(Halstrup04(), 'halstrup_04', 80, 120, do_run=do_run)


def run_mimo_himmelblau(do_run=True):
    run_mi_h_algo(MIMOHimmelblau(), 'mimo_himm', 10, 25, do_run=do_run)


def run_zaefferer(do_run=True):
    for problem, name in [
        (ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI), 'zaef_a'),
        (ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.B_OPT_INACT_IMP_UNPR_UNI), 'zaef_b'),
        (ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.C_OPT_ACT_IMP_PROF_BI), 'zaef_c'),
        (ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.D_OPT_ACT_IMP_UNPR_BI), 'zaef_d'),
        (ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI), 'zaef_e'),
    ]:
        run_mi_h_algo(problem, name, 3, 10, do_run=do_run)


def run_mi_h_infill_force_cont(problem: Problem, name: str, n_init: int, n_max: int, do_run=True):
    metrics, plot_metric_values = get_metrics(problem, include_loo_cv=False)

    results_key = 'eff_validate_mi_h_fc_%s' % name

    def _get_algo(sm, force_cont):
        infill = ExpectedImprovementInfill()
        return SurrogateBasedInfill(infill=infill, surrogate_model=sm, termination=100, verbose=True,
                                    infill_force_cont=force_cont).algorithm(infill_size=1, init_size=n_init)

    smt_kwargs = {'theta0': 1.}
    sms = [
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, **smt_kwargs), 'cont_relax'),
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=True, **smt_kwargs), 'dummy_coding'),
    ]

    algorithms = [_get_algo(sm, fc) for sm, _ in sms for fc in [False, True]]
    algorithm_names = [('SBO(%s, force_cont=%r)' % (name, fc)) for _, name in sms for fc in [False, True]]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_eval_max=n_max, do_run=do_run)


def run_mi_h_algo(problem: Problem, name: str, n_init: int, n_max: int, do_run=True):
    metrics, plot_metric_values = get_metrics(problem, include_loo_cv=False)

    results_key = 'eff_validate_mi_h_%s' % name

    def _get_algo(sm):
        infill = ExpectedImprovementInfill()
        return SurrogateBasedInfill(infill=infill, surrogate_model=sm, termination=100, verbose=True)\
            .algorithm(infill_size=1, init_size=n_init)

    smt_kwargs = {'theta0': 1.}
    sklearn_kwargs = {'alpha': 1e-6, 'int_as_discrete': True}
    sms = [
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=False, **smt_kwargs), 'cont_relax'),
        (SMTKrigingSurrogateModel(auto_wrap_mixed_int=True, **smt_kwargs), 'dummy_coding'),

        (SKLearnGPSurrogateModel(kernel=HammingDistance().kernel(), **sklearn_kwargs), 'MI: Ham'),
        # (SKLearnGPSurrogateModel(kernel=GowerDistance().kernel(), **sklearn_kwargs), 'MI: Gow'),  # Slow
        # (SKLearnGPSurrogateModel(kernel=SymbolicCovarianceDistance().kernel(), **sklearn_kwargs), 'MI: SC'),  # Unstab

        # (SKLearnGPSurrogateModel(kernel=ArcDistance().kernel(), **sklearn_kwargs), 'MI+H: Arc'),  # Slow
        (SKLearnGPSurrogateModel(kernel=IndefiniteConditionalDistance().kernel(), **sklearn_kwargs), 'MI+H: Ico'),
        # (SKLearnGPSurrogateModel(kernel=ImputationDistance().kernel(), **sklearn_kwargs), 'MI+H: Imp'),  # Slow
        (SKLearnGPSurrogateModel(kernel=WedgeDistance().kernel(), **sklearn_kwargs), 'MI+H: Wed'),

        (SKLearnGPSurrogateModel(kernel=CompoundSymmetryKernel().kernel(), **sklearn_kwargs), 'MI: CS'),
        # (SKLearnGPSurrogateModel(kernel=LatentVariablesDistance().kernel(), **sklearn_kwargs), 'MI: LV'),  # Slow

        (SKLearnGPSurrogateModel(kernel=SPWDecompositionKernel(CompoundSymmetryKernel().kernel()),
                                 **sklearn_kwargs), 'MI+H: SPW+CS'),
        # (SKLearnGPSurrogateModel(kernel=SPWDecompositionKernel(LatentVariablesDistance().kernel()),
        #                          **sklearn_kwargs), 'MI+H: SPW+LV'),
        (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(CompoundSymmetryKernel().kernel()),
                                 **sklearn_kwargs), 'MI+H: DVW+CS'),
        # (SKLearnGPSurrogateModel(kernel=DVWDecompositionKernel(LatentVariablesDistance().kernel()),
        #                          **sklearn_kwargs), 'MI+H: DVW+LV'),
    ]

    algorithms = [_get_algo(sm) for sm, _ in sms ]
    algorithm_names = [('SBO(%s)' % name) for _, name in sms]

    run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_eval_max=n_max, do_run=do_run)


def get_metrics(_: Problem, include_loo_cv=True):
    metrics = [
        BestObjMetric(),
        MaxConstraintViolationMetric(),
        SurrogateQualityMetric(include_loo_cv=include_loo_cv, n_loo_cv=4),
        TrainingMetric(),
        InfillMetric(),
    ]
    plot_metric_values = {
        'f_best': None,
        'max_cv': ['max_cv'],
        'sm_quality': ['rmse', 'loo_cv'] if include_loo_cv else ['rmse'],
        'training': ['n_train', 'n_samples', 'time_train'],
        'infill': ['time_infill'],
    }
    return metrics, plot_metric_values


def run(results_key, problem, algorithms, algorithm_names, metrics, plot_metric_values, n_repeat=8, n_eval_max=500,
        do_run=True):
    runner.set_results_folder(results_key)
    exp = runner.get_experimenters(problem, algorithms, metrics, n_eval_max=n_eval_max, algorithm_names=algorithm_names)

    if do_run:
        runner.run_effectiveness_multi(exp, n_repeat=n_repeat)
    runner.plot_effectiveness_results(exp, plot_metric_values=plot_metric_values, save=True, show=False)


if __name__ == '__main__':
    run_goldstein(
        # do_run=False,
    )
    run_rosenbrock(
        # do_run=False,
    )
    run_munoz_zuniga(
        # do_run=False,
    )
    run_halstrup_4(
        # do_run=False,
    )
    run_halstrup_4_fc(
        # do_run=False,
    )
    run_zaefferer(
        # do_run=False,
    )
    run_mimo_himmelblau(
        # do_run=False,
    )
