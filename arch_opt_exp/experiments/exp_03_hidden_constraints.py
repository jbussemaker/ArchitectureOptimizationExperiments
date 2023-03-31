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
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from arch_opt_exp.experiments.runner import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sb_arch_opt.sampling import *
from sb_arch_opt.problem import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.problems.turbofan_arch import *
from sb_arch_opt.problems.hidden_constraints import *

log = logging.getLogger('arch_opt_exp.03_hc')
capture_log()

_exp_03_01_folder = '03_hc_01_hc_area'
_exp_03_02_folder = '03_hc_02_hc_test_area'

_test_problems = [
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
    for problem, category in _test_problems:
        name = f'{category} {problem.__class__.__name__}'
        log.info(f'Plotting {problem.__class__.__name__}')
        plot_distance_distributions(problem, f'{folder}/plot_{category}_{problem.__class__.__name__}', name)


if __name__ == '__main__':
    # exp_03_01_hc_area()
    exp_03_02_hc_test_area()
