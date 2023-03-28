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

from pymoo.problems.multi.omnitest import OmniTest
from pymoo.core.mixed import MixedVariableSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *
from sb_arch_opt.problems.turbofan_arch import *

log = logging.getLogger('arch_opt_exp.01_sampling')
capture_log()

_exp_01_01_folder = '01_sampling_01_dv_opt_occurrence'
_exp_01_02_folder = '01_sampling_02_sampling_similarity'
_exp_01_03_folder = '01_sampling_03_doe_accuracy'
_exp_01_04_folder = '01_sampling_04_activeness_diversity'
_exp_01_05_folder = '01_sampling_05_perf_influence'

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


class HierarchicalDirectRandomSampling(HierarchicalRandomSampling):
    """Directly sample from all available discrete design vectors"""

    def __init__(self):
        super().__init__(sobol=False)

    @classmethod
    def _sample_discrete_x(cls, n_samples: int, is_cont_mask, x_all: np.ndarray, is_act_all: np.ndarray, sobol=False):
        has_x_cont = np.any(is_cont_mask)

        x = x_all
        if n_samples < x.shape[0]:
            i_x = cls._choice(n_samples, x.shape[0], replace=False, sobol=sobol)
        elif has_x_cont:
            # If there are more samples requested than points available, only repeat points if there are continuous vars
            i_x_add = cls._choice(n_samples-x.shape[0], x.shape[0], sobol=sobol)
            i_x = np.sort(np.concatenate([np.arange(x.shape[0]), i_x_add]))
        else:
            i_x = np.arange(x.shape[0])

        x = x[i_x, :]
        is_active = is_act_all[i_x, :]
        return x, is_active


_samplers = [
    (False, FloatRandomSampling()),
    (False, MixedVariableSampling()),
    (False, LatinHypercubeSampling()),
    (True, HierarchicalDirectRandomSampling()),
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


def q25(x_):
    return x_.quantile(.25)


def q75(x_):
    return x_.quantile(.75)


def exp_01_03_doe_accuracy():
    """
    Investigates how accurate surrogate models trained on DOE's sampled by the different samplers are.

    Hypothesis:
    Using hierarchical samplers for hierarchical problems results in higher accuracy (lower errors).
    For non-hierarchical problems it does not matter.

    Conclusions:
    - For non-hierarchical test problems, all samplers have the same performance.
    - For mixed-discrete hierarchical test problems, hierarchical samplers perform better (direct sampler best)
    - For the discrete hierarchical test problem, the hierarchical direct sampler perform worst, other hierarchical
      samplers perform as good or better.
    - In terms of CPU time, LHS is much more expensive than the other samplers, at a marginal benefit.
    """
    folder = set_results_folder(_exp_01_03_folder)
    n_train_mult = np.array([1, 2, 5, 10])
    n_test_factor = 10
    n_repeat = 100
    problems = [
        MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
        CombHierBranin(),  # More realistic hierarchical test problem
        CombHierMO(),  # More realistic multi-objective hierarchical test problem
        CombHierDMO(),  # More realistic multi-objective discrete hierarchical test problem

        HierarchicalGoldstein(),  # Hierarchical test problem by Pelamatti
        HierarchicalRosenbrock(),  # Hierarchical test problem by Pelamatti
        # MOHierarchicalTestProblem(),  # Hierarchical test problem by Bussemaker (2021)
        Jenatton(),
    ]

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
            # plt.title(f'{problem.__class__.__name__}\n$k_{{test}}$ = {n_test_factor:.1f}, '
            #           f'$n_{{repeat}}$ = {n_repeat}, $n_{{dim}}$ = {problem.n_var}')
            plt.title(f'{problem.__class__.__name__}\n$n_{{repeat}}$ = {n_repeat}, $n_{{dim}}$ = {problem.n_var}')

            for sampler in df.columns.get_level_values(0).unique():
                name = sampler.replace('Sampling', '')
                fmt = '--' if 'Hierarchical' not in name else '-'
                data = df[sampler]
                k, mid = data.index.values, data['median'].values
                l, = plt.plot(k, mid, fmt, linewidth=1, marker='.', label=name)
                q25_values, q75_values = data['q25'].values, data['q75'].values
                plt.fill_between(k, q25_values, q75_values, alpha=.05, color=l.get_color(), linewidth=0)

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
        n_train = int(n_train_mult_factor*problem.n_var)
        log.info(f'Repetition {rep}: {n_train_mult_factor:.1f}*{problem.n_var} = {n_train} training points')
        x_train = _sample_and_repair(problem, sampler, n_train, is_repaired)
        y_train = problem.evaluate(x_train, return_as_dictionary=True)['F']
        y_norm = np.mean(y_train)
        y_train /= y_norm

        if isinstance(n_test_factor, np.ndarray) and n_test_factor.shape[1] == x_train.shape[1]:
            x_test = n_test_factor
        else:
            n_test = max(1, int(n_test_factor * problem.n_var))
            # x_test = _sample_and_repair(problem, FloatRandomSampling(), n_test, is_repaired=False)
            # x_test = _sample_and_repair(problem, HierarchicalRandomSampling(), n_test)
            # x_test = _sample_and_repair(problem, sampler, n_test, is_repaired)
            x_test = _sample_and_repair(problem, HierarchicalExhaustiveSampling(n_cont=1), n_test)

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


def exp_01_04_activeness_diversity_ratio():
    """
    Investigate the influence of activeness diversity ratio on hierarchical sampler performance. Activeness diversity
    ratio (ADR) is the ratio between the largest and smallest groups of discrete design vectors separated by the nr of
    active discrete design variables.

    Hypothesis:
    A larger activeness diversity ratio leads to worse performance of the direct hierarchical samplers, and better for
    the activeness-separated hierarchical samplers. This is because if the activeness diversity ratio is high, the
    separation by nr of active discrete variables ensures that also from these design vector groups some samples are
    taken, whereas for direct sampling this is not the case and there is a high chance that only vectors from large
    groups are sampled.

    Conclusions:
    - For the mixed-discrete problems, all samplers perform similar and there is no trend w.r.t. ADR
    - For the discrete problems, direct sampling performs much worse
      - For the other samplers, a higher ADR leads to lower RMSE
    """
    folder = set_results_folder(_exp_01_04_folder)
    n_train = 50
    n_test = 100
    n_repeat = 100

    problems = [
        # Target: n_valid_discr ~ 1000, imp_ratio ~ 5, activeness diversity ratio increasing
        # Note: do not use n_opts=4 for the OmniTest problem!!
        ('Mixed-discrete', [
            CombinatorialHierarchicalMetaProblem(  # 1024, 4.9, 1
                NoHierarchyWrappedProblem(OmniTest(n_var=5)),
                n_parts=4, n_sel_dv=3, sep_power=1.4, target_n_opts_ratio=1., repr_str='ADR_MD_0010'),
            CombinatorialHierarchicalMetaProblem(  # 1024, 6.1, 7.8
                NoHierarchyWrappedProblem(OmniTest(n_var=5)),
                n_parts=4, n_sel_dv=4, sep_power=1.3, target_n_opts_ratio=1., repr_str='ADR_MD_0078'),
            CombinatorialHierarchicalMetaProblem(  # 1024, 4.5, 78
                NoHierarchyWrappedProblem(OmniTest(n_var=5)),
                n_parts=4, n_sel_dv=5, sep_power=1.3, target_n_opts_ratio=5., repr_str='ADR_MD_0780'),
            CombinatorialHierarchicalMetaProblem(  # 1024, 4.3, 113
                NoHierarchyWrappedProblem(OmniTest(n_var=5)),
                n_parts=4, n_sel_dv=6, sep_power=1.28, target_n_opts_ratio=3., repr_str='ADR_MD_1130'),
        ]),
        ('Discrete', [
            CombinatorialHierarchicalMetaProblem(  # 1034, 2.8, 4
                MixedDiscretizerProblemBase(OmniTest(n_var=3), n_opts=12),
                n_parts=6, n_sel_dv=2, sep_power=1., target_n_opts_ratio=1., repr_str='ADR_D_0040'),
            CombinatorialHierarchicalMetaProblem(  # 1034, 5.9, 13.7
                MixedDiscretizerProblemBase(OmniTest(n_var=3), n_opts=12),
                n_parts=6, n_sel_dv=5, sep_power=1.2, target_n_opts_ratio=1., repr_str='ADR_D_0137'),
            CombinatorialHierarchicalMetaProblem(  # 1117, 5.6, 153
                MixedDiscretizerProblemBase(OmniTest(n_var=3), n_opts=11),
                n_parts=7, n_sel_dv=4, sep_power=1.1, target_n_opts_ratio=1., repr_str='ADR_D_1530'),
            CombinatorialHierarchicalMetaProblem(  # 1360, 6.0, 256
                MixedDiscretizerProblemBase(OmniTest(n_var=4), n_opts=8),
                n_parts=2, n_sel_dv=7, sep_power=1.1, target_n_opts_ratio=1., repr_str='ADR_D_2560'),
        ]),
    ]
    # for _, problem_set in problems:
    #     for problem in problem_set:
    #         problem.print_stats()
    # return

    samplers = [
        (False, FloatRandomSampling()),
        (True, HierarchicalDirectRandomSampling()),
        (True, HierarchicalUniformRandomSampling()),
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for set_name, problem_set in problems:
            df_agg = []
            for i, (is_repaired, sampler) in enumerate(samplers):
                df_samplers = []
                for j, problem in enumerate(problem_set):
                    log.info(f'Sampling {problem!r} ({j+1}/{len(problem_set)}) '
                             f'with sampler: {sampler!r} ({i+1}/{len(samplers)})')

                    rep = f'sampler {i+1}/{len(samplers)}; prob {j+1}/{len(problem_set)}; rep '
                    n_train_mult = n_train/problem.n_var
                    n_test_factor = n_test/problem.n_var
                    futures = [executor.submit(_sample_and_train, f'{rep}{k+1}/{n_repeat}', problem, sampler,
                                               is_repaired, [n_train_mult], n_test_factor) for k in range(n_repeat)]
                    concurrent.futures.wait(futures)
                    sampler_data = [fut.result() for fut in futures]

                    data = np.array(sampler_data)  # (n_repeat x len(n_train_mult))
                    mid, std = np.mean(data, axis=0), np.std(data, axis=0)
                    data[np.abs(data-mid) > 1.5*std] = np.nan

                    act_div_ratio = get_activeness_diversity_ratio(problem)
                    df_sampler = pd.DataFrame(columns=[act_div_ratio], data=data)
                    df_sampler = df_sampler.agg(['mean', 'std', 'median', q25, q75])
                    df_sampler = df_sampler.set_index(pd.MultiIndex.from_tuples(
                        [(sampler.__class__.__name__, val) for val in df_sampler.index]))
                    df_samplers.append(df_sampler)

                df_sampler_agg = pd.concat(df_samplers, axis=1).T
                df_agg.append(df_sampler_agg)
                df_sampler_agg.to_pickle(f'{folder}/output_{set_name.lower()}_{sampler.__class__.__name__}.pkl')

            df_agg = pd.concat(df_agg, axis=1)
            df_agg.to_pickle(f'{folder}/output_{set_name.lower()}.pkl')

    # Plot results
    with pd.ExcelWriter(f'{folder}/output.xlsx') as writer:
        for set_name, problem_set in problems:
            with open(f'{folder}/output_{set_name.lower()}.pkl', 'rb') as fp:
                df: pd.DataFrame = pickle.load(fp)
            df.to_excel(writer, sheet_name=set_name)

            plt.figure(figsize=(8, 4))
            # plt.title(f'{set_name}\n$n_{{train}}$ = {n_train}, $n_{{test}}$ = {n_test}, $n_{{repeat}}$ = {n_repeat}')
            plt.title(f'{set_name}\n$n_{{train}}$ = {n_train}, $n_{{repeat}}$ = {n_repeat}')

            for sampler in df.columns.get_level_values(0).unique():
                name = sampler.replace('Sampling', '')
                fmt = '--' if 'Hierarchical' not in name else '-'
                data = df[sampler]
                adr, mid = data.index.values, data['median'].values
                l, = plt.plot(adr, mid, fmt, linewidth=1, marker='.', label=name)
                q25_values, q75_values = data['q25'].values, data['q75'].values
                plt.fill_between(adr, q25_values, q75_values, alpha=.05, color=l.get_color(), linewidth=0)

            plt.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False)
            ax = plt.gca()
            ax.set_xscale('log'), ax.set_yscale('log')
            ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
            plt.ylabel('RMSE'), plt.xlabel('Activeness diversity ratio')
            plt.tight_layout()
            plt.savefig(f'{folder}/plot_{set_name.lower()}.png')


def get_n_opt_max(problem: ArchOptProblemBase):
    is_discrete_mask = problem.is_discrete_mask
    return int(np.max(problem.xu[is_discrete_mask]-problem.xl[is_discrete_mask]+1))


def get_activeness_diversity_ratio(problem: ArchOptProblemBase):
    """Ratio between the largest and smallest activeness groups (groups of discrete design vectors, obtained when
    splitting by nr of active discrete variables)"""
    x_discrete, is_act_discrete = HierarchicalExhaustiveSampling().get_all_x_discrete(problem)
    if x_discrete.shape[0] == 0:
        return 1.

    is_cont_mask = HierarchicalExhaustiveSampling.get_is_cont_mask(problem)
    x_groups, _, _ = HierarchicalRandomSampling.split_by_discrete_n_active(
        x_discrete, is_act_discrete, is_cont_mask)

    group_sizes = [len(group) for group in x_groups]
    return max(group_sizes)/min(group_sizes)


def get_cont_imp_ratio(problem: ArchOptProblemBase):
    is_cont_mask = problem.is_cont_mask
    if np.sum(is_cont_mask) == 0:
        return 1

    _, is_act_all = problem.all_discrete_x
    is_act_cont = is_act_all[:, is_cont_mask]
    n_cont_dim = is_act_cont.shape[0]*is_act_cont.shape[1]
    n_cont_act = np.sum(is_act_cont)
    return n_cont_dim/n_cont_act


def get_cont_sample_ratio(problem: ArchOptProblemBase, n_samples):
    x_discrete, _ = problem.all_discrete_x
    n_discrete = x_discrete.shape[0]
    return n_samples/n_discrete


def get_train_cont_act_ratio(problem: ArchOptProblemBase, sample_func, n=50):
    ratios = []
    is_cont_mask = problem.is_cont_mask
    for _ in range(n):
        _, is_active = problem.correct_x(sample_func())
        is_act_cont = is_active[:, is_cont_mask]
        n_cont_dim = is_act_cont.shape[0]*is_act_cont.shape[1]
        n_cont_act = np.sum(is_act_cont)
        ratios.append(n_cont_dim/n_cont_act)
    return np.mean(ratios)


def get_partial_option_activeness_ratio(problem: ArchOptProblemBase, n_indep_vars: int):
    x_discrete, is_act_discrete = problem.all_discrete_x
    x = x_discrete[:, problem.is_discrete_mask]
    is_active = is_act_discrete[:, problem.is_discrete_mask]
    xl, xu = problem.xl[problem.is_discrete_mask], problem.xu[problem.is_discrete_mask]

    _, ix_indep_unique, counts = np.unique(x[:, :n_indep_vars], axis=0, return_counts=True, return_inverse=True)
    n_active = 0
    n_no_poa = 0
    for i_group in range(len(counts)):
        mask = ix_indep_unique == i_group
        x_dependent = x[mask, :][:, n_indep_vars:]
        is_active_group = is_active[mask, :][:, n_indep_vars:]
        n_active += np.sum(is_active_group)

        x_dep_min, x_dep_max = np.min(x_dependent, axis=0), np.max(x_dependent, axis=0)
        has_poa = (x_dep_min != xl[n_indep_vars:]) | (x_dep_max != xu[n_indep_vars:])
        n_no_poa += np.sum(is_active_group*(~has_poa))

    return n_no_poa / n_active


def exp_01_05_performance_influence():
    """
    Investigate what the main influence on hierarchical sampler performance is.

    Hypothesis:
    There exists some metric that consistently predicts hierarchical sampler performance.

    Conclusions:
    - No defining parameter found
    """
    folder = set_results_folder(_exp_01_05_folder)
    n_train = 50
    n_test = 10000
    n_repeat = 100
    samplers = [
        (False, FloatRandomSampling()),
        (True, HierarchicalDirectRandomSampling()),
        (True, HierarchicalUniformRandomSampling()),
    ]

    problems = [
        # Target: imp_ratio ~ 5; note: do not use n_opts=4 for the OmniTest problem!!
        # ('Mixed-discrete', [
        #     # CombinatorialHierarchicalMetaProblem(  # 256, 1.6
        #     #     NoHierarchyWrappedProblem(OmniTest(n_var=8)),
        #     #     n_parts=2, n_sel_dv=1, sep_power=2, target_n_opts_ratio=1., repr_str='MD_8_2'),
        #     CombinatorialHierarchicalMetaProblem(  # 128, 1.6
        #         NoHierarchyWrappedProblem(OmniTest(n_var=7)),
        #         n_parts=2, n_sel_dv=2, sep_power=2, target_n_opts_ratio=1., repr_str='MD_7_2'),
        #     CombinatorialHierarchicalMetaProblem(  # 64, 4.9
        #         NoHierarchyWrappedProblem(OmniTest(n_var=6)),
        #         n_parts=2, n_sel_dv=3, sep_power=1.7, target_n_opts_ratio=1., repr_str='MD_6_3'),
        #     CombinatorialHierarchicalMetaProblem(  # 160, 4.5
        #         MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=1),
        #         n_parts=2, n_sel_dv=3, sep_power=1.5, target_n_opts_ratio=1., repr_str='MD_5_4'),
        #     CombinatorialHierarchicalMetaProblem(  # 1000, 4.9
        #         MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=3),
        #         n_parts=2, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3'),
        #     CombinatorialHierarchicalMetaProblem(  # 6250, 4.9
        #         MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=5),
        #         n_parts=2, n_sel_dv=3, sep_power=1.1, target_n_opts_ratio=1., repr_str='MD_8_1'),
        #     CombinatorialHierarchicalMetaProblem(  # 5827, 14.0
        #         MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5),
        #         n_parts=2, n_sel_dv=3, sep_power=1., target_n_opts_ratio=1., repr_str='MD_0_9'),
        # ]),
        ('Variations', [
            CombinatorialHierarchicalMetaProblem(  # 1000, 4.9
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=3),
                n_parts=2, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3'),
            CombinatorialHierarchicalMetaProblem(  # 1000, 4.9
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=3),
                n_parts=2, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3_n_opts'),
            CombinatorialHierarchicalMetaProblem(  # 1000, 4.9
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=2),
                n_parts=2, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3_n_vars_int'),
            CombinatorialHierarchicalMetaProblem(  # 1000, 4.9
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=2),
                n_parts=3, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3_n_parts'),
            CombinatorialHierarchicalMetaProblem(  # 1000, 4.9
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=2),
                n_parts=3, n_sel_dv=5, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3_n_sel_dv'),
            CombinatorialHierarchicalMetaProblem(  # CombHierMO
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=2),
                n_parts=3, n_sel_dv=5, sep_power=1.1, target_n_opts_ratio=1., repr_str='CombHierMO'),
        ]),
        ('Variations2', [
            CombinatorialHierarchicalMetaProblem(
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=3),
                n_parts=2, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3'),
            CombinatorialHierarchicalMetaProblem(
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=3),
                n_parts=3, n_sel_dv=3, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3_n_parts'),
            CombinatorialHierarchicalMetaProblem(
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=3),
                n_parts=3, n_sel_dv=5, sep_power=1.3, target_n_opts_ratio=1., repr_str='MD_6_3_n_sel_dv'),
            CombinatorialHierarchicalMetaProblem(
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=3),
                n_parts=3, n_sel_dv=5, sep_power=1.1, target_n_opts_ratio=1., repr_str='MD_6_3_sep_power'),
            CombinatorialHierarchicalMetaProblem(
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5, n_vars_int=2),
                n_parts=3, n_sel_dv=5, sep_power=1.1, target_n_opts_ratio=1., repr_str='MD_6_3_n_vars_int'),
            CombinatorialHierarchicalMetaProblem(  # CombHierMO
                MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=2),
                n_parts=3, n_sel_dv=5, sep_power=1.1, target_n_opts_ratio=1., repr_str='CombHierMO'),
        ]),
    ]
    # for _, problem_set in problems:
    #     for problem in problem_set:
    #         problem.print_stats()
    # return

    x_test_db = {}
    for set_name, problem_set in problems:
        for problem in problem_set:
            if np.sum(problem.is_cont_mask) > 0:
                x_test = HierarchicalDirectRandomSampling().do(problem, n_test).get('X')
            else:
                x_test = HierarchicalExhaustiveSampling().do(problem, 0).get('X')

            log.info(f'Test samples for {repr(problem)}: {x_test.shape[0]}')
            x_test_db[set_name, repr(problem)] = x_test
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for set_name, problem_set in problems:
            df_agg = []
            for i, (is_repaired, sampler) in enumerate(samplers):
                df_samplers = []
                for j, problem in enumerate(problem_set):
                    log.info(f'Sampling {problem!r} ({j+1}/{len(problem_set)}) '
                             f'with sampler: {sampler!r} ({i+1}/{len(samplers)})')

                    rep = f'sampler {i+1}/{len(samplers)}; prob {j+1}/{len(problem_set)}; rep '
                    n_train_mult = n_train/problem.n_var
                    x_test = x_test_db[set_name, repr(problem)]
                    futures = [executor.submit(_sample_and_train, f'{rep}{k+1}/{n_repeat}', problem, sampler,
                                               is_repaired, [n_train_mult], x_test) for k in range(n_repeat)]
                    concurrent.futures.wait(futures)
                    sampler_data = [fut.result() for fut in futures]

                    data = np.array(sampler_data)  # (n_repeat x len(n_train_mult))
                    mid, std = np.mean(data, axis=0), np.std(data, axis=0)
                    data[np.abs(data-mid) > 1.5*std] = np.nan

                    cont_dim_ratio = np.sum(problem.is_cont_mask)/problem.n_var
                    df_sampler = pd.DataFrame(columns=[cont_dim_ratio*100], data=data)
                    df_sampler = df_sampler.agg(['mean', 'std', 'median', q25, q75])
                    df_sampler = df_sampler.set_index(pd.MultiIndex.from_tuples(
                        [(sampler.__class__.__name__, val) for val in df_sampler.index]))
                    df_samplers.append(df_sampler)

                df_sampler_agg = pd.concat(df_samplers, axis=1).T
                df_agg.append(df_sampler_agg)
                df_sampler_agg.to_pickle(f'{folder}/output_{set_name.lower()}_{sampler.__class__.__name__}.pkl')

            df_agg = pd.concat(df_agg, axis=1)
            df_agg.to_pickle(f'{folder}/output_{set_name.lower()}.pkl')

    # Plot results
    show_names = True
    with pd.ExcelWriter(f'{folder}/output.xlsx') as writer:
        for set_name, problem_set in problems:
            with open(f'{folder}/output_{set_name.lower()}.pkl', 'rb') as fp:
                df: pd.DataFrame = pickle.load(fp)

            df['problem', 'name'] = [repr(problem) for problem in problem_set]
            df['problem', 'n_discr'] = [np.sum(problem.is_discrete_mask) for problem in problem_set]
            df['problem', 'n_opt_max'] = [get_n_opt_max(problem) for problem in problem_set]
            df['problem', 'n_cont'] = [np.sum(problem.is_cont_mask) for problem in problem_set]
            df['problem', 'n_dec_discr'] = [problem.get_n_declared_discrete() for problem in problem_set]
            df['problem', 'n_valid_discr'] = [problem.get_n_valid_discrete() for problem in problem_set]
            df['problem', 'imp_ratio'] = [problem.get_imputation_ratio() for problem in problem_set]
            df['problem', 'adr'] = [get_activeness_diversity_ratio(problem) for problem in problem_set]
            df['problem', 'imp_ratio_cont'] = [get_cont_imp_ratio(problem) for problem in problem_set]
            df['problem', 'cont_ratio_train'] = [get_cont_sample_ratio(problem, n_train) for problem in problem_set]
            df['problem', 'cont_ratio_test'] = [get_cont_sample_ratio(problem, n_test) for problem in problem_set]
            df['problem', 'ir_train_float'] = [get_train_cont_act_ratio(problem, lambda: _sample_and_repair(
                problem, FloatRandomSampling(), n_train, is_repaired=False)) for problem in problem_set]
            df['problem', 'ir_train_dir'] = [get_train_cont_act_ratio(problem, lambda: _sample_and_repair(
                problem, HierarchicalDirectRandomSampling(), n_train)) for problem in problem_set]
            df['problem', 'ir_train_uni'] = [get_train_cont_act_ratio(problem, lambda: _sample_and_repair(
                problem, HierarchicalUniformRandomSampling(), n_train)) for problem in problem_set]
            df['problem', 'poa_ratio'] = [get_partial_option_activeness_ratio(problem, problem._x_sel.shape[1])
                                          for problem in problem_set]

            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                   'display.expand_frame_repr', False):
                print(df['problem'])

            df.to_excel(writer, sheet_name=set_name)

            for x_col, x_name in [('cdr', 'Continuous dimensions [%]'), ('idx', 'Problem variant')]:
                plt.figure(figsize=(8, 4))
                plt.title(f'{set_name}\n$n_{{train}}$ = {n_train}, $n_{{repeat}}$ = {n_repeat}')

                y_values = []
                x_values = None
                for sampler in df.columns.get_level_values(0).unique():
                    if sampler == 'problem':
                        continue
                    name = sampler.replace('Sampling', '')
                    fmt = '--' if 'Hierarchical' not in name else '-'
                    data = df[sampler]
                    cdr, mid = data.index.values, data['median'].values
                    if x_values is None:
                        if x_col == 'cdr':
                            x_values = cdr
                        elif x_col == 'idx':
                            x_values = np.arange(len(mid))
                    l, = plt.plot(x_values, mid, fmt, linewidth=1, marker='.', label=name)
                    y_values.append(mid)
                    q25_values, q75_values = data['q25'].values, data['q75'].values
                    plt.fill_between(x_values, q25_values, q75_values, alpha=.05, color=l.get_color(), linewidth=0)

                if show_names:
                    prob_names = df['problem', 'name']
                    y_values = np.max(y_values, axis=0)
                    names = prob_names.values
                    for i, prob_name in enumerate(names):
                        plt.text(x_values[i], y_values[i], '  '+prob_name, horizontalalignment='left',
                                 verticalalignment='center', rotation=60, rotation_mode='anchor')

                plt.legend(loc='center left', bbox_to_anchor=(1, .5), frameon=False)
                ax = plt.gca()
                ax.set_yscale('log')
                ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
                plt.ylabel('RMSE'), plt.xlabel(x_name)
                plt.tight_layout()
                plt.savefig(f'{folder}/plot_{set_name.lower()}_{x_col}.png')


if __name__ == '__main__':
    # exp_01_01_dv_opt_occurrence()
    # exp_01_02_sampling_similarity()
    exp_01_03_doe_accuracy()
    # exp_01_04_activeness_diversity_ratio()
    # exp_01_05_performance_influence()
