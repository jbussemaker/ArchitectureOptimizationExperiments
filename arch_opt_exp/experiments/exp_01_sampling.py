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
import pickle
import logging
import numpy as np
import pandas as pd
from typing import *
import concurrent.futures
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from arch_opt_exp.experiments.runner import *

from pymoo.core.mixed import MixedVariableSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.turbofan_arch import *

log = logging.getLogger('arch_opt_exp.01_sampling')
capture_log()

_exp_01_01_folder = '01_sampling_01_dv_opt_occurrence'
_exp_01_02_folder = '01_sampling_02_sampling_similarity'
_exp_01_03_folder = '01_sampling_03_doe_accuracy'

_all_problems = [
    SimpleTurbofanArch(),  # Realistic hierarchical problem

    HierarchicalGoldstein(),  # Hierarchical test problem by Pelamatti
    HierarchicalRosenbrock(),  # Hierarchical test problem by Pelamatti

    MOHierarchicalTestProblem(),  # Hierarchical test problem by Bussemaker (2021)

    MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
    CombHierBranin(),  # More realistic hierarchical test problem
    CombHierMO(),  # More realistic multi-objective hierarchical test problem
    CombHierDMO(),  # More realistic multi-objective discrete hierarchical test problem
]
_problems = [
    SimpleTurbofanArch(),  # Realistic hierarchical problem
    MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
    CombHierBranin(),  # More realistic hierarchical test problem
    CombHierMO(),  # More realistic multi-objective hierarchical test problem
    CombHierDMO(),  # More realistic multi-objective discrete hierarchical test problem
]


class HierarchicalUniformRandomSampling(HierarchicalRandomSampling):

    def __init__(self):
        super().__init__(sobol=False)


class HierarchicalSobolSampling(HierarchicalRandomSampling):

    def __init__(self):
        super().__init__(sobol=True)


_samplers = [
    (False, FloatRandomSampling()),
    (False, MixedVariableSampling()),
    (False, LatinHypercubeSampling()),
    (True, HierarchicalUniformRandomSampling()),
    (True, HierarchicalSobolSampling()),
    (True, HierarchicalLatinHypercubeSampling()),
]


def exp_01_01_dv_opt_occurrence():
    """
    Investigate the spread of discrete DV option occurrences for different types of problems.

    Hypothesis:
    Realistically-behaving hierarchical problems have several discrete variables where there is a large discrepancy in
    how often their options appear in all possible design variables.

    Conclusions:
    - Realistic architecture optimization problems usually have several discrete design variables with large
      discrepancies in the nr of times their options occur in all the possible design vectors; this can be explained by
      the observation that usually several design variables are the most impactful in terms of architecture, and thereby
      might activate or deactivate many of the downstream design variables: if one of the options then disables most of
      the downstream design variables, its presence is greatly reduced in all available design vectors
    - (Mixed-)discrete problems without hierarchy do not see this effect: the design vectors are simply the Cartesian
      product of design variable options, and therefore each option occurs 1/n_opts times, where n_opts is the number of
      options for the design variable
    - Good hierarchical test problems should have the same behavior
    - The MOHierarchicalTestProblem used in the Aviation 2021 paper doesn't have this behavior
    - Better test problems: CombHierBranin (SO), CombHierMORosenbrock (MO)
    """
    folder = set_results_folder(_exp_01_01_folder)
    with pd.ExcelWriter(f'{folder}/output.xlsx') as writer:

        for i, problem in enumerate(_all_problems):
            # Exhaustively sample the problem
            log.info(f'Sampling {problem!r}')
            x = HierarchicalExhaustiveSampling(n_cont=1).do(problem, 0).get('X')

            # Count appearances of design variable options
            d_mask = problem.is_discrete_mask
            x_rel = _count_appearance(x[:, d_mask], problem.xl[d_mask], problem.xu[d_mask])
            index = [f'opt {i}' for i in range(x_rel.shape[0])]

            # Calculate spread of option appearance
            x_mean = np.nanmean(x_rel, axis=0)
            x_spread = np.sqrt(np.nanmean((x_rel-x_mean)**2, axis=0))
            index.append('rmse')

            x_range = np.nanmax(x_rel, axis=0)-np.nanmin(x_rel, axis=0)
            index.append('range')

            x_rel = np.row_stack([x_rel, x_spread, x_range])

            cols = [f'x{ix}' for ix in range(problem.n_var) if d_mask[ix]]
            df = pd.DataFrame(index=index, data=x_rel, columns=cols)
            df.to_excel(writer, sheet_name=repr(problem))
            df.to_pickle(f'{folder}/df_{secure_filename(repr(problem))}.pkl')


def _count_appearance(x, xl, xu):
    # Round and reduce bin sizes in order to count continuous variables
    x -= np.min(x, axis=0)
    for i in range(x.shape[1]):
        if xu[i]-xl[i] > 6:
            x[:, i] = x[:, i]*(6/(xu[i]-xl[i]))
    x = np.round(x)

    n_opt_max = int(np.max(x)+1)
    x_count = np.zeros((n_opt_max, x.shape[1]))*np.nan
    for ix in range(x.shape[1]):
        values, counts = np.unique(x[:, ix], return_counts=True)
        x_count[:len(counts), ix] = counts
    return x_count/x.shape[0]


def _sample_and_repair(problem, sampler, n_samples, is_repaired=True):
    x = sampler.do(problem, n_samples).get('X')

    if isinstance(sampler, MixedVariableSampling):
        x = np.array([[row[x_name] for x_name in problem.vars.keys()] for i, row in enumerate(x)])

    if not is_repaired:
        x = ArchOptRepair().do(problem, x)
        x = x[~LargeDuplicateElimination.eliminate(x), :]
    return x


def exp_01_02_sampling_similarity():
    """
    Investigates which non-exhaustive sampling algorithms best approximate the occurrence rates of exhaustive sampling.

    Hypothesis:
    For non-hierarchical problems it does not matter (probably LHS is better). For hierarchical problems the
    hierarchical samplers are closer to the exhaustively-sampled occurrence rates.

    Conclusions:
    - Correspondence to exhaustively-sampled rates is measured as max(max(rate-rate_exhaustive per x) over all x)
    - Each sampler was tested with 1000 samples
    - For non-hierarchical problems all samplers perform well: correspondence is < 3%
    - For hierarchical problems:
      - Non-hierarchical samplers always perform bad: 20% < correspondence < 40%
      - Hierarchical samplers perform well: correspondence < 4%
      - They do this by exhaustively-sampling all discrete design vectors and then randomly sampling these
      - A safeguard against time/memory usage is implemented; if triggered, they perform as non-hierarchical samplers
    """
    exp1_folder = set_results_folder(_exp_01_01_folder)
    folder = set_results_folder(_exp_01_02_folder)
    n_samples = 1000

    df_exhaustive: List[pd.DataFrame] = []
    for i, problem in enumerate(_problems):
        path = f'{exp1_folder}/df_{secure_filename(repr(problem))}.pkl'
        with open(path, 'rb') as fp:
            df_exhaustive.append(pickle.load(fp))

    for i, (is_repaired, sampler) in enumerate(_samplers):
        with pd.ExcelWriter(f'{folder}/output_{i}_{sampler.__class__.__name__}.xlsx') as writer:
            for j, problem in enumerate(_problems):
                log.info(f'Sampling {problem!r} ({j+1}/{len(_problems)}) '
                         f'with sampler: {sampler!r} ({i+1}/{len(_samplers)})')
                x = _sample_and_repair(problem, sampler, n_samples, is_repaired)

                # Count appearances of design variable options
                x_rel = _count_appearance(x, problem.xl, problem.xu)
                index = [f'opt {i}' for i in range(x_rel.shape[0])]

                # Calculate difference to exhaustively-sampled rates
                d_mask = problem.is_discrete_mask
                x_rel_diff = x_rel*0
                x_rel_exhaustive = df_exhaustive[j].values[:x_rel.shape[0], :]
                x_diff = x_rel[:x_rel_exhaustive.shape[0], d_mask]-x_rel_exhaustive
                x_rel_diff[:x_diff.shape[0], d_mask] = x_diff
                index += [f'opt diff {i}' for i in range(x_rel.shape[0])]

                x_max_diff = np.nanmax(np.abs(x_rel_diff), axis=0)
                index.append('max diff')

                x_count = np.ones((x.shape[1],), dtype=int)*x.shape[0]
                index.append('count')

                x_rel = np.row_stack([x_rel, x_rel_diff, x_max_diff, x_count])
                cols = [f'x{ix}{" (c)" if problem.is_cont_mask[ix] else ""}' for ix in range(x_rel.shape[1])]

                x_rel = np.column_stack([x_rel, np.max(np.abs(x_rel), axis=1)])
                cols.append('max')

                df = pd.DataFrame(index=index, data=x_rel, columns=cols)
                df.to_excel(writer, sheet_name=problem.__class__.__name__)


def exp_01_03_doe_accuracy():
    """
    Investigates how accurate surrogate models trained on DOE's sampled by the different samplers are.

    Hypothesis:
    Using hierarchical samplers for hierarchical problems results in higher accuracy (lower errors).
    For non-hierarchical problems it does not matter.

    Conclusions:
    - For non-hierarchical test problems, all samplers have the same performance.
    - For mixed-discrete hierarchical test problems, hierarchical samplers perform better.
    - For the discrete hierarchical test problem, hierarchical samplers perform worse.
    - In terms of CPU time, LHS is much more expensive than the other samplers, at a marginal benefit.
    """
    folder = set_results_folder(_exp_01_03_folder)
    n_train_mult = np.array([1, 2, 5, 10])
    n_test_factor = 1
    n_repeat = 20
    problems = _problems[1:]

    def q25(x_):
        return x_.quantile(.25)

    def q75(x_):
        return x_.quantile(.75)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, problem in enumerate(problems):
            df_samplers = []
            for j, (is_repaired, sampler) in enumerate(_samplers):
                log.info(f'Sampling {problem!r} ({i+1}/{len(problems)}) '
                         f'with sampler: {sampler!r} ({j+1}/{len(_samplers)})')

                rep = f'prob {i+1}/{len(problems)}; sampler {j+1}/{len(_samplers)}; rep '
                futures = [executor.submit(_sample_and_train, f'{rep}{k+1}/{n_repeat}', problem, sampler,
                                           is_repaired, n_train_mult, n_test_factor) for k in range(n_repeat)]
                concurrent.futures.wait(futures)
                sampler_data = [fut.result() for fut in futures]

                data = np.array(sampler_data)  # (n_repeat x len(n_train_mult))
                mid, std = np.mean(data, axis=0), np.std(data, axis=0)
                data[np.abs(data-mid) > 1.5*std] = np.nan

                df_sampler = pd.DataFrame(columns=n_train_mult, data=data)
                df_sampler = df_sampler.agg(['mean', 'std', 'median', q25, q75])
                df_sampler = df_sampler.set_index(pd.MultiIndex.from_tuples(
                    [(sampler.__class__.__name__, val) for val in df_sampler.index]))
                df_samplers.append(df_sampler)

            df_agg = pd.concat(df_samplers, axis=0).T
            df_agg.to_pickle(f'{folder}/output_{problem.__class__.__name__}.pkl')

    # Plot results
    with pd.ExcelWriter(f'{folder}/output.xlsx') as writer:
        for problem in problems:
            with open(f'{folder}/output_{problem.__class__.__name__}.pkl', 'rb') as fp:
                df: pd.DataFrame = pickle.load(fp)
            df.to_excel(writer, sheet_name=problem.__class__.__name__)

            plt.figure(figsize=(8, 4))
            plt.title(f'{problem.__class__.__name__}\n$k_{{test}}$ = {n_test_factor:.1f}, '
                      f'$n_{{repeat}}$ = {n_repeat}, $n_{{dim}}$ = {problem.n_var}')

            for sampler in df.columns.get_level_values(0).unique():
                name = sampler.replace('Sampling', '')
                fmt = '--' if 'Repaired' not in name else '-'
                data = df[sampler]
                k, mid = data.index.values, data['median'].values
                l, = plt.plot(k, mid, fmt, linewidth=1, marker='.', label=name)
                q25, q75 = data['q25'].values, data['q75'].values
                plt.fill_between(k, q25, q75, alpha=.05, color=l.get_color(), linewidth=0)

            plt.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False)
            ax = plt.gca()
            ax.set_xscale('log'), ax.set_yscale('log')
            ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
            plt.ylabel('RMSE'), plt.xlabel('$k = n_{train} / n_{dim}$')
            plt.tight_layout()
            plt.savefig(f'{folder}/plot_{problem.__class__.__name__}.png')


def _sample_and_train(rep, problem, sampler, is_repaired, n_train_mult_factors, n_test_factor):
    rmse_values = []
    for n_train_mult_factor in n_train_mult_factors:
        n_train = n_train_mult_factor*problem.n_var
        log.info(f'Repetition {rep}: {n_train_mult_factor}*{problem.n_var} = {n_train} training points')
        x_train = _sample_and_repair(problem, sampler, n_train, is_repaired)
        y_train = problem.evaluate(x_train, return_as_dictionary=True)['F']
        y_norm = np.mean(y_train)
        y_train /= y_norm

        x_test = _sample_and_repair(problem, HierarchicalRandomSampling(), max(1, int(n_test_factor * problem.n_var)))
        y_test = problem.evaluate(x_test, return_as_dictionary=True)['F']/y_norm

        # Normalize inputs
        x_min, x_max = np.min(x_train, axis=0), np.max(x_train, axis=0)
        same = x_max == x_min
        x_max[same] = x_min[same]+1
        x_train = (x_train-x_min)/(x_max-x_min)
        x_test = (x_test-x_min)/(x_max-x_min)

        # Train a surrogate model
        from smt.surrogate_models.krg import KRG
        model = KRG(print_global=False, theta0=[1e-5]*problem.n_var)
        model.set_training_values(x_train, y_train)
        model.train()

        # Get error metric
        y_predict = model.predict_values(x_test)
        rmse = np.sqrt(np.mean((y_predict-y_test)**2))
        rmse_values.append(rmse)
    return rmse_values


if __name__ == '__main__':
    # exp_01_01_dv_opt_occurrence()
    # exp_01_02_sampling_similarity()
    exp_01_03_doe_accuracy()
