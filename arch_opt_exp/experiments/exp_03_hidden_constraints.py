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

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
import os
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from scipy.spatial import distance
from smt.surrogate_models.krg import KRG
from werkzeug.utils import secure_filename
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from arch_opt_exp.experiments.runner import *
from pymoo.core.initialization import Initialization
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sb_arch_opt.sampling import *
from sb_arch_opt.problem import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.problems.turbofan_arch import *
from sb_arch_opt.algo.simple_sbo.infill import *
from sb_arch_opt.problems.hidden_constraints import *

from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.hc_strategies.rejection import *
from arch_opt_exp.hc_strategies.prediction import *
from arch_opt_exp.hc_strategies.replacement import *
from arch_opt_exp.hc_strategies.sbo_with_hc import *
from arch_opt_exp.experiments.metrics import get_exp_metrics

log = logging.getLogger('arch_opt_exp.03_hc')
capture_log()

_exp_03_01_folder = '03_hc_01_hc_area'
_exp_03_02_folder = '03_hc_02_hc_test_area'
_exp_03_03_folder = '03_hc_03_hc_predictors'
_exp_03_04_folder = '03_hc_04_simple_optimization'
_exp_03_05_folder = '03_hc_05_optimization'

_test_problems = lambda: [
    # HFR = high failure rate; G = constrained
    (HCBranin(), '00_far'),
    (Mueller02(), '01'),
    (HCSphere(), '01'),
    (Mueller01(), '02_HFR'),
    (Mueller08(), '02_HFR'),
    (CantileveredBeamHC(), '03_G_HFR'),
    (MOMueller08(), '04_MO_HFR'),
    (CarsideHCLess(), '05_MO_G'),
    (CarsideHC(), '05_MO_G_HFR'),
    (MDMueller02(), '06_MD'),
    (MDMueller08(), '06_MD_HFR'),
    (MDCantileveredBeamHC(), '07_MD_G_HFR'),
    (MDMOMueller08(), '08_MD_MO_HFR'),
    (MDCarsideHC(), '09_MD_MO_G_HFR'),
    (HierMueller02(), '10_HIER'),
    (HierMueller08(), '10_HIER_HFR'),
    (HierarchicalRosenbrockHC(), '11_HIER_G'),
    (MOHierMueller08(), '12_HIER_MO_HFR'),
    (MOHierarchicalRosenbrockHC(), '13_HIER_MO_G_HFR'),
]


def exp_03_01_hc_area():
    """
    Investigate where areas with hidden constraints (i.e. failed areas) typically lie.

    Hypothesis:
    As opposed to some test problems with hidden constraints, areas with violated hidden constraints are usually not
    distributed "randomly" throughout the design space. Instead, they are usually found far away from (Pareto-) optimal
    points, because the hidden constraints stem from simulations unable to converge [Forrester2006, Dupont2019].

    Forrester, Alexander IJ, András Sóbester, and Andy J. Keane. "Optimization with missing data." Proceedings of the
    Royal Society A: Mathematical, Physical and Engineering Sciences 462, no. 2067 (2006): 935-945.

    Dupont, C., A. Tromba, and S. Missonnier. "New strategy to preliminary design space launch vehicle based on a
    dedicated MDO platform." Acta Astronautica 158 (2019): 103-110.

    Conclusion:
    - Failed points are in general found at a slightly larger distance to feasible and/or Pareto points, however not by
      a large margin
    - Hidden constraint strategies should both be tried on test problems with distributed and far-away failed areas
    """
    folder = set_results_folder(_exp_03_01_folder)
    problems = [
        (SimpleTurbofanArch(), 350),
        # (RealisticTurbofanArch(), 20),
    ]

    # Sample the design space of the turbofan arch problems
    for problem, n_samples in problems:
        name = problem.__class__.__name__
        results_folder = f'{folder}/{name}'
        os.makedirs(results_folder, exist_ok=True)

        problem.verbose = True
        problem.n_parallel = 4
        problem.results_folder = results_folder
        problem.set_max_iter(30)

    #     log.info(f'Evaluating {name} with {n_samples} points')
    #     doe_algo = get_doe_algo(doe_size=n_samples, results_folder=results_folder)
    #     doe_algo.setup(problem)
    #     doe_algo.run()
    #     pop = doe_algo.pop
    #
    #     is_failed = ArchOptProblemBase.get_failed_points(pop)
    #     fail_rate = np.sum(is_failed)/len(is_failed)
    #     log.info(f'Evaluation finished, failure rate: {fail_rate*100:.0f}%')

    # Post-process results
    for problem, _ in problems:
        name = problem.__class__.__name__
        results_folder = f'{folder}/{name}'
        pop = load_from_previous_results(problem, results_folder)
        assert len(pop) > 0

        is_failed = ArchOptProblemBase.get_failed_points(pop)
        fail_rate = np.sum(is_failed)/len(is_failed)
        log.info(f'Evaluation finished, failure rate: {fail_rate*100:.0f}%')

        # # Verify that failures are deterministic
        # from pymoo.core.evaluator import Evaluator
        # from pymoo.core.population import Population
        # ok_pop = pop[~is_failed]
        # ok_pop_test = ok_pop[np.random.choice(len(ok_pop), 8)]
        # out_pop = Evaluator().eval(problem, Population.new(X=ok_pop_test.get('X')))
        # n_failed = np.sum(ArchOptProblemBase.get_failed_points(out_pop))
        # print(f'Verification 1: {n_failed}/{len(out_pop)} failed for non-failed ref population')
        #
        # failed_pop = pop[is_failed]
        # failed_pop_test = failed_pop[np.random.choice(len(failed_pop), 8)]
        # out_pop = Evaluator().eval(problem, Population.new(X=failed_pop_test.get('X')))
        # n_failed = np.sum(ArchOptProblemBase.get_failed_points(out_pop))
        # print(f'Verification 2: {n_failed}/{len(out_pop)} failed for failed ref population')

        # Plot distances to feasible and pareto front
        plot_distance_distributions(
            problem, f'{folder}/plot_{problem.__class__.__name__}', problem.__class__.__name__, pop=pop)


def plot_distance_distributions(problem: ArchOptProblemBase, save_path: str, name: str, pop: Population = None, n=500):
    if pop is None:
        pop = HierarchicalRandomSampling().do(problem, n)
        pop = Evaluator().eval(problem, pop)

    # Get Pareto front
    f, x = pop.get('F'), pop.get('X')
    feas = pop.get('feas')
    is_failed = ArchOptProblemBase.get_failed_points(pop)
    feas[is_failed] = False
    i_feas = np.where(feas)[0]
    f_feas, x_feas = f[feas, :], x[feas, :]
    i_pf = NonDominatedSorting().do(f_feas, only_non_dominated_front=True)

    n_feas, n_failed, n = np.sum(feas), np.sum(is_failed), len(pop)
    log.info(f'{name}: {n} points, {n_feas} feasible ({(n_feas/n)*100:.0f}%), '
             f'{n_failed} failed ({(n_failed/n)*100:.0f}%)')
    has_infeasible = np.any(~feas)

    # Get and plot distance to feasible and pareto sets
    d = {}
    x_norm = (x-problem.xl)/(problem.xu-problem.xl)
    for (from_name, from_mask), (to_name, to_mask) in itertools.product([
        ('failed', is_failed),  # ('valid', ~is_failed),
        ('infeasible', (~feas) & (~is_failed)), ('feasible', feas),
    ], [('feasible', feas), ('pareto', i_feas[i_pf])]):
        if to_name == 'feasible' and not has_infeasible:
            continue
        x_from, x_to = x_norm[from_mask], x_norm[to_mask]
        if to_name not in d:
            d[to_name] = {}
        d[to_name][from_name] = np.min(distance.cdist(x_from, x_to, metric='cityblock'), axis=1)

    fig, ax = plt.subplots(1, len(d), figsize=(8, 4), sharex='all')
    fig.suptitle(f'{name}\n{(n_failed/n)*100:.0f}% failed ({((n-n_failed)/n)*100:.0f}% valid), '
                 f'{(n_feas/n)*100:.0f}% feasible, n = {n}')

    max_dist = max(max(d_values) if len(d_values) > 0 else 0 for d_ in d.values() for d_values in d_.values())
    bins = np.linspace(0, max_dist, 50)
    for i, to_name in enumerate(d):
        ax[i].set_xlabel(f'Normalized Manhattan dist to {to_name}' if i == 0 else f'... to {to_name}')
        for from_name in d[to_name]:
            if from_name == to_name:
                continue
            dist = d[to_name][from_name]
            if len(dist) > 0:
                ax[i].hist(dist, bins, alpha=.25, label=from_name)

    plt.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False)
    plt.tight_layout()
    plt.savefig(f'{save_path}.png')


def exp_03_02_hc_test_area():
    """
    Plot distance distributions of test problems.

    Hypothesis:
    Test problems designed for randomized failed areas show more overlap in distance distribution than test problems
    with failed areas far away from the optimum.

    Conclusions: hypothesis confirmed
    """
    folder = set_results_folder(_exp_03_02_folder)
    for problem, category in _test_problems():
        name = f'{category} {problem.__class__.__name__}'
        log.info(f'Plotting {problem.__class__.__name__}')
        plot_distance_distributions(problem, f'{folder}/plot_{category}_{problem.__class__.__name__}', name)


_predictors: List[PredictorInterface] = [
    RandomForestClassifier(n=100),
    KNNClassifier(k=5),
    GPClassifier(nu=2.5),
    SVMClassifier(),
    LinearRBFRegressor(),
    GPRegressor(),
    VariationalGP(),
]


def exp_03_03_hc_predictors():
    """
    Plot different predictors applied to a very simple test problem.

    Hypothesis:
    Some predictors are well able to capture the hidden constraint areas.

    Conclusions:
    - GP regressor/classifier, Linear RBF and Random Forest Classifier seem most promising
    """
    folder = set_results_folder(_exp_03_03_folder)

    ref_saved = False
    for n in [50]:
        curves = []
        max_acc = []
        predictor_names = []
        for i, predictor in enumerate(_predictors):
            log.info(f'Testing {n} samples for predictor ({i+1}/{len(_predictors)}): {predictor!s}')
            predictor_names.append(str(predictor))

            save_path = f'{folder}/{n}_{i}_{secure_filename(str(predictor))}'
            fpr, tpr, acc, _ = predictor.get_stats(
                Alimo(), n=n, plot=True, save_path=save_path, save_ref=not ref_saved, show=False)
            curves.append((fpr, tpr))
            max_acc.append(max(acc))
            plt.close('all')
            ref_saved = True

        plt.figure(figsize=(10, 4)), plt.title(f'ROC Curves ({n} samples)')
        for i, name in enumerate(predictor_names):
            label = f'{name} ({max_acc[i]*100:.1f}% acc)'
            plt.plot(curves[i][0], curves[i][1], linewidth=1, label=label)
        plt.xlim([0, 1]), plt.ylim([0, 1]), plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
        plt.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False)
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{folder}/{n}_roc.png')


_strategies: List[HiddenConstraintStrategy] = [
    RejectionHCStrategy(),

    GlobalWorstReplacement(),
    LocalReplacement(n=1),
    LocalReplacement(n=5, mean=False),
    LocalReplacement(n=5, mean=True),
    PredictedWorstReplacement(mul=1.),
    PredictedWorstReplacement(mul=2.),

    PredictionHCStrategy(RandomForestClassifier(n=100)),
    PredictionHCStrategy(RandomForestClassifier(n=100), constraint=False),
    PredictionHCStrategy(GPClassifier(nu=2.5)),
    PredictionHCStrategy(GPClassifier(nu=2.5), constraint=False),
    PredictionHCStrategy(LinearRBFRegressor()),
    PredictionHCStrategy(LinearRBFRegressor(), constraint=False),
    PredictionHCStrategy(GPRegressor()),
    PredictionHCStrategy(GPRegressor(), constraint=False),
]


def _get_sbo(problem: ArchOptProblemBase, strategy: HiddenConstraintStrategy, doe_pop: Population):
    model = KRG(print_global=False)
    if problem.n_obj == 1:
        infill = ExpectedImprovementInfill()
    else:
        infill = MinVariancePFInfill()

    sbo = HiddenConstraintsSBO(model, infill, init_size=len(doe_pop), hc_strategy=strategy)\
        .algorithm(infill_size=1, init_size=len(doe_pop))
    sbo.initialization = Initialization(doe_pop)
    return sbo


def exp_03_04_simple_optimization():
    """
    Plot different predictors applied to a very simple test problem.

    Hypothesis:
    Some predictors are well able to capture the hidden constraint areas.

    Conclusions:
    - The test problem is simple so most strategies are able to approximate the best point closely
    - The rejection strategy get stuck at a failed point in the design space with high EI after about 10 iterations, it
      is not able to find any valid points after that
    - The replacement strategies struggle to approximate the best point, and generate many failed points
      - Local 5 mean replacement seems to work best
      - Predicted worst replacement is able to relatively quickly find the best point without generating failed points;
        after that, it start generating many failed points
    - The tested prediction strategies all work very well, they all manage to find the best point in 3 to 4 iterations
      - Constraint-based infill reduces the amount of failed points after finding the best point
      - Objective-based infill on the other hand leads to more exploration after finding the best point
    """
    folder = set_results_folder(_exp_03_04_folder)
    n_init = 30
    n_infill = 30
    for problem, f_known_best in [(Alimo(), -1.0474), (AlimoEdge(), -1.0468)]:
        prob_name = problem.__class__.__name__
        doe = get_doe_algo(n_init)
        doe.setup(problem)
        doe.run()
        doe_pop = doe.pop
        log.info(f'Best of initial DOE: {np.nanmin(doe_pop.get("F")[:, 0])} (best: {f_known_best})')

        for i, strategy in enumerate(_strategies):
            log.info(f'Strategy {i+1}/{len(_strategies)}: {strategy!s}')
            strategy_folder = f'{folder}/{prob_name}_{i:02d}_{secure_filename(str(strategy))}'
            os.makedirs(strategy_folder, exist_ok=True)

            sbo = _get_sbo(problem, strategy, doe_pop)
            sbo.setup(problem)
            sbo.ask()  # Once to initialize the infill search using the DOE
            sbo_infill: HiddenConstraintsSBO = sbo.infill_obj
            n_pop, n_fail = [len(doe_pop)], [np.sum(ArchOptProblemBase.get_failed_points(doe_pop))]
            f_best = [np.nanmin(doe_pop.get('F')[:, 0])]
            for i_infill in range(n_infill):
                # Do the last infill using the mean prediction
                if i_infill == n_infill-1:
                    sbo_infill.infill = inf = FunctionEstimatePoFInfill()
                    inf.initialize(sbo_infill.problem, sbo_infill.surrogate_model)

                log.info(f'Infill {i_infill+1}/{n_infill}')
                infills = sbo.ask()
                assert len(infills) == 1
                sbo.evaluator.eval(problem, infills)

                if i_infill == 0:
                    sbo_infill.plot_state(save_path=f'{strategy_folder}/doe', show=False)
                sbo_infill.plot_state(x_infill=infills.get('X')[0, :], plot_std=False,
                                      save_path=f'{strategy_folder}/infill_{i_infill}', show=False)
                if isinstance(strategy, PredictionHCStrategy):
                    strategy.predictor.get_stats(problem, train=False, save_ref=False,
                                                 save_path=f'{strategy_folder}/predictor_{i_infill}', show=False)

                sbo.tell(infills=infills)

                n_pop.append(len(sbo.pop))
                n_fail.append(np.sum(ArchOptProblemBase.get_failed_points(sbo.pop)))
                f_best.append(np.nanmin(sbo.pop.get('F')[:, 0]))

            n_pop, n_fail, f_best = np.array(n_pop), np.array(n_fail), np.array(f_best)
            plt.figure(), plt.title(f'{strategy!s} on {prob_name}')
            plt.plot(n_fail/n_pop, 'k', linewidth=2, label='Failure rate')
            plt.plot(n_fail[0]/n_pop, '--k', linewidth=.5, label='Perfect failure rate')
            plt.legend(frameon=False), plt.xlabel('Iteration'), plt.ylabel('Failure rate')
            plt.tight_layout()
            plt.savefig(f'{strategy_folder}/failure_rate.png')

            plt.figure(), plt.title(f'{strategy!s} on {prob_name}')
            plt.plot(f_best, 'k', linewidth=2)
            plt.plot([f_known_best]*len(f_best), '--k', linewidth=.5)
            plt.xlabel('Iteration'), plt.ylabel('Best $f$')
            plt.tight_layout()
            plt.savefig(f'{strategy_folder}/f.png')


def exp_03_05_optimization():
    """
    Apply the different strategies to all test problems.

    For the DOE size the following rule of thumb is used:
    5 times the nr of dimensions, corrected for the expected failure rate --> 5*n_dim/(1-expected_fail_rate)
    This is the ensure there are enough valid points to start the optimization with.

    Hypothesis:
    Some predictors are well able to capture the hidden constraint areas.

    Conclusions:
    """
    folder = set_results_folder(_exp_03_05_folder)
    expected_fail_rate = .6
    n_infill = 30
    n_repeat = 20

    problems = _test_problems()
    problem_paths = []
    problem_names = []
    for i, (problem, category) in enumerate(problems):
        name = f'{category} {problem.__class__.__name__}'
        problem_names.append(name)
        problem_path = f'{folder}/{secure_filename(name)}'
        problem_paths.append(problem_path)

        # Rule of thumb: 5*n_dim --> corrected for expected fail rate (unknown before running a problem, of course)
        n_init = int(np.ceil(5*problem.n_var/(1-expected_fail_rate)))

        log.info(f'Running optimizations for {i+1}/{len(problems)}: {name} (n_init = {n_init})')
        problem.pareto_front()

        doe = {}
        for i_rep in range(n_repeat):
            doe_algo = get_doe_algo(n_init)
            doe_algo.setup(problem)
            doe_algo.run()
            doe[i_rep] = doe_algo.pop

        metrics = get_exp_metrics(problem, including_convergence=False)
        metrics += [
            FailRateMetric(),
            PredictorMetric(),
        ]
        additional_plot = {
            'fail': ['rate'],
            'hc_pred': ['acc', 'max_acc', 'max_acc_pov'],
        }

        algorithms = []
        algo_names = []
        for j, strategy in enumerate(_strategies):
            sbo = _get_sbo(problem, strategy, doe[0])
            algorithms.append(sbo)
            algo_names.append(str(strategy))

        do_run = True
        # do_run = False
        exps = run(_exp_03_05_folder, problem, algorithms, algo_names, doe=doe, n_repeat=n_repeat, n_eval_max=n_infill,
                   metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run)

        df_data = []
        for exp in exps:
            with open(exp.get_problem_algo_results_path('result_agg_df.pkl'), 'rb') as fp:
                df_strat = pickle.load(fp)
                df_data.append(pd.Series(df_strat.iloc[-1, :], name=exp.algorithm_name))

        df_prob = pd.concat(df_data, axis=1).T
        df_prob.to_pickle(f'{problem_path}/df_strategies.pkl')
        with pd.ExcelWriter(f'{problem_path}/df_strategies.xlsx') as writer:
            df_prob.to_excel(writer)

    df_probs = []
    for i, problem_name in enumerate(problem_names):
        problem_path = problem_paths[i]
        try:
            with open(f'{problem_path}/df_strategies.pkl', 'rb') as fp:
                df_prob = pickle.load(fp)
        except FileNotFoundError:
            continue

        df_prob = df_prob.set_index(pd.MultiIndex.from_tuples([(problem_name, val) for val in df_prob.index]))
        df_probs.append(df_prob)

    with pd.ExcelWriter(f'{folder}/results.xlsx') as writer:
        df_agg = pd.concat(df_probs, axis=0)
        df_agg.to_excel(writer)


if __name__ == '__main__':
    # exp_03_01_hc_area()
    # exp_03_02_hc_test_area()
    # exp_03_03_hc_predictors()
    # exp_03_04_simple_optimization()
    exp_03_05_optimization()
