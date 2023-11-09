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
import matplotlib
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from arch_opt_exp.experiments.runner import *
from arch_opt_exp.experiments.metrics import *
from arch_opt_exp.experiments.plotting import *
from arch_opt_exp.md_mo_hier.sampling import *
from arch_opt_exp.hc_strategies.metrics import *
from arch_opt_exp.md_mo_hier.correction import *
from arch_opt_exp.md_mo_hier.hier_problems import *
from arch_opt_exp.md_mo_hier.hierarchical_comb import *

from pymoo.problems.multi.omnitest import OmniTest
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *
from sb_arch_opt.problems.turbofan_arch import *
from sb_arch_opt.algo.pymoo_interface.api import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.arch_sbo.models import *

log = logging.getLogger('arch_opt_exp.01_sampling')
capture_log()

_exp_01_01_folder = '01_sampling_01_dv_opt_occurrence'
_exp_01_02_folder = '01_sampling_02_sampling_similarity'
_exp_01_03_folder = '01_sampling_03_doe_accuracy'
_exp_01_04_folder = '01_sampling_04_activeness_diversity'
_exp_01_05_folder = '01_sampling_05_correction'
_exp_01_06_folder = '01_sampling_06_optimization'

_all_problems = lambda: [
    SimpleTurbofanArch(),  # Realistic hierarchical problem

    HierarchicalGoldstein(),  # Hierarchical test problem by Pelamatti
    HierarchicalRosenbrock(),  # Hierarchical test problem by Pelamatti

    MOHierarchicalTestProblem(),  # Hierarchical test problem by Bussemaker (2021)

    MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
    HierBranin(),  # More realistic hierarchical test problem
    HierZDT1(),  # More realistic multi-objective hierarchical test problem
    HierDiscreteZDT1(),  # More realistic multi-objective discrete hierarchical test problem
]
_problems = lambda: [
    SimpleTurbofanArch(),  # Realistic hierarchical problem
    MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
    HierBranin(),  # More realistic hierarchical test problem
    HierZDT1(),  # More realistic multi-objective hierarchical test problem
    HierDiscreteZDT1(),  # More realistic multi-objective discrete hierarchical test problem
]


_samplers = [
    (RepairedSampler(FloatRandomSampling()), 'Rnd'),
    (RepairedSampler(LatinHypercubeSampling()), 'LHS'),
    (NoGroupingHierarchicalSampling(), 'HierNoGroup'),
    (NrActiveHierarchicalSampling(), 'HierNrAct'),
    (NrActiveHierarchicalSampling(weight_by_nr_active=True), 'HierNrActWt'),
    (ActiveVarHierarchicalSampling(), 'HierAct'),
    (ActiveVarHierarchicalSampling(weight_by_nr_active=True), 'HierActWt'),
    # (HierarchicalCoveringSampling(), 'Covering'),
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

        for i, problem in enumerate(_all_problems()):
            # Exhaustively sample the problem
            log.info(f'Sampling {problem!r}')

            # Count appearances of design variable options
            df = problem.get_discrete_rates(force=True).iloc[:, problem.is_discrete_mask]
            df.to_excel(writer, sheet_name=repr(problem))
            df.to_pickle(f'{folder}/df_{secure_filename(repr(problem))}.pkl')


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
    raise RuntimeError('Experiment code is broken --> update to use problem.get_discrete_rates!')
    exp1_folder = set_results_folder(_exp_01_01_folder)
    folder = set_results_folder(_exp_01_02_folder)
    n_samples = 1000

    problems = _problems()
    df_exhaustive: List[pd.DataFrame] = []
    for i, problem in enumerate(problems):
        path = f'{exp1_folder}/df_{secure_filename(repr(problem))}.pkl'
        with open(path, 'rb') as fp:
            df_exhaustive.append(pickle.load(fp))

    for i, (sampler, sampler_name) in enumerate(_samplers):
        with pd.ExcelWriter(f'{folder}/output_{i}_{sampler_name}.xlsx') as writer:
            for j, problem in enumerate(problems):
                log.info(f'Sampling {problem!r} ({j+1}/{len(problems)}) '
                         f'with sampler: {sampler_name} ({i+1}/{len(_samplers)})')
                x = sampler.do(problem, n_samples).get('X')

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
        HierBranin(),  # More realistic hierarchical test problem
        HierZDT1(),  # More realistic multi-objective hierarchical test problem
        HierDiscreteZDT1(),  # More realistic multi-objective discrete hierarchical test problem

        HierarchicalGoldstein(),  # Hierarchical test problem by Pelamatti
        HierarchicalRosenbrock(),  # Hierarchical test problem by Pelamatti
        # MOHierarchicalTestProblem(),  # Hierarchical test problem by Bussemaker (2021)
        Jenatton(),
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, problem in enumerate(problems):
            df_samplers = []
            for j, (sampler, sampler_name) in enumerate(_samplers):
                log.info(f'Sampling {problem!r} ({i+1}/{len(problems)}) '
                         f'with sampler: {sampler_name} ({j+1}/{len(_samplers)})')

                rep = f'prob {i+1}/{len(problems)}; sampler {j+1}/{len(_samplers)}; rep '
                futures = [executor.submit(_sample_and_train, f'{rep}{k+1}/{n_repeat}', problem, sampler,
                                           n_train_mult, n_test_factor) for k in range(n_repeat)]
                concurrent.futures.wait(futures)
                sampler_data = [fut.result() for fut in futures]

                data = np.array(sampler_data)  # (n_repeat x len(n_train_mult))
                mid, std = np.mean(data, axis=0), np.std(data, axis=0)
                data[np.abs(data-mid) > 1.5*std] = np.nan

                df_sampler = pd.DataFrame(columns=n_train_mult, data=data)
                df_sampler = df_sampler.agg(['mean', 'std', 'median', q25, q75])
                df_sampler = df_sampler.set_index(pd.MultiIndex.from_tuples(
                    [(sampler_name, val) for val in df_sampler.index]))
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


def _sample_and_train(rep, problem, sampler, n_train_mult_factors, n_test_factor):
    rmse_values = []
    for n_train_mult_factor in n_train_mult_factors:
        n_train = int(n_train_mult_factor*problem.n_var)
        log.info(f'Repetition {rep}: {n_train_mult_factor:.1f}*{problem.n_var} = {n_train} training points')
        x_train = sampler.do(problem, n_train).get('X')
        y_train = problem.evaluate(x_train, return_as_dictionary=True)['F']
        y_norm = np.mean(y_train)
        y_train /= y_norm

        if isinstance(n_test_factor, np.ndarray) and n_test_factor.shape[1] == x_train.shape[1]:
            x_test = n_test_factor
        else:
            n_test = max(1, int(n_test_factor * problem.n_var))
            x_test = HierarchicalExhaustiveSampling(n_cont=1).do(problem, n_test).get('X')

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
        RepairedSampler(FloatRandomSampling()),
        NoGroupingHierarchicalSampling(),
        NrActiveHierarchicalSampling(),
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for set_name, problem_set in problems:
            df_agg = []
            for i, sampler in enumerate(samplers):
                df_samplers = []
                for j, problem in enumerate(problem_set):
                    log.info(f'Sampling {problem!r} ({j+1}/{len(problem_set)}) '
                             f'with sampler: {sampler!r} ({i+1}/{len(samplers)})')

                    rep = f'sampler {i+1}/{len(samplers)}; prob {j+1}/{len(problem_set)}; rep '
                    n_train_mult = n_train/problem.n_var
                    n_test_factor = n_test/problem.n_var
                    futures = [executor.submit(_sample_and_train, f'{rep}{k+1}/{n_repeat}', problem, sampler,
                                               [n_train_mult], n_test_factor) for k in range(n_repeat)]
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
    x_groups = NrActiveHierarchicalSampling().group_design_vectors(x_discrete, is_act_discrete, is_cont_mask)

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
    raise NotImplementedError
    folder = set_results_folder(_exp_01_05_folder)
    n_train = 50
    n_test = 10000
    n_repeat = 100
    samplers = [
        RepairedSampler(FloatRandomSampling()),
        NoGroupingHierarchicalSampling(),
        NrActiveHierarchicalSampling(),
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
                x_test = NoGroupingHierarchicalSampling().do(problem, n_test).get('X')
            else:
                x_test = HierarchicalExhaustiveSampling().do(problem, 0).get('X')

            log.info(f'Test samples for {repr(problem)}: {x_test.shape[0]}')
            x_test_db[set_name, repr(problem)] = x_test
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for set_name, problem_set in problems:
            df_agg = []
            for i, sampler in enumerate(samplers):
                df_samplers = []
                for j, problem in enumerate(problem_set):
                    log.info(f'Sampling {problem!r} ({j+1}/{len(problem_set)}) '
                             f'with sampler: {sampler!r} ({i+1}/{len(samplers)})')

                    rep = f'sampler {i+1}/{len(samplers)}; prob {j+1}/{len(problem_set)}; rep '
                    n_train_mult = n_train/problem.n_var
                    x_test = x_test_db[set_name, repr(problem)]
                    futures = [executor.submit(_sample_and_train, f'{rep}{k+1}/{n_repeat}', problem, sampler,
                                               [n_train_mult], x_test) for k in range(n_repeat)]
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


def _get_metrics(problem):
    metrics = get_exp_metrics(problem, including_convergence=False)+[SBOTimesMetric()]
    additional_plot = {
        'time': ['train', 'infill'],
    }
    return metrics, additional_plot


class CorrectorFactory:

    def __init__(self, klass, **kwargs):
        self.klass = klass
        self.kw = kwargs

    def __call__(self, ds, is_valid):
        if issubclass(self.klass, LazyCorrectorBase):
            return self.klass(ds, is_valid, **self.kw)
        return self.klass(ds, **self.kw)


def exp_01_05_correction(sbo=True, post_process=False):
    """
    Run optimizations with different correction strategies for different sub-problem properties and optimum locations.
    """
    folder_post = '' if sbo else '_nsga2'
    folder = set_results_folder(_exp_01_05_folder+folder_post)
    n_infill = 100
    n_gen = 25
    n_repeat = 8 if sbo else 100
    doe_k = 10
    n_sub, n_opts = 9, 3
    i_opt_test = [0, n_sub-1]

    eager_samplers = [
        (RepairedSampler(LatinHypercubeSampling()), 'LHS'),
        (NoGroupingHierarchicalSampling(), 'HierNoGroup'),
        (NrActiveHierarchicalSampling(), 'HierNrAct'),
        (NrActiveHierarchicalSampling(weight_by_nr_active=True), 'HierNrActWt'),
        (ActiveVarHierarchicalSampling(), 'HierAct'),
        (ActiveVarHierarchicalSampling(weight_by_nr_active=True), 'HierActWt'),
    ]
    lazy_samplers = [
        (RepairedSampler(LatinHypercubeSampling()), 'LHS'),
    ]

    # if sbo:
    #     correctors = [
    #         (CorrectorFactory(AnyEagerCorrector, correct_valid_x=False, random_if_multiple=True), 'Eager Rnd', eager_samplers),
    #         (CorrectorFactory(GreedyEagerCorrector, correct_valid_x=False, random_if_multiple=True), 'Eager Greedy Rnd', eager_samplers),
    #         (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=False), 'Lazy Closest', lazy_samplers),
    #     ]
    # else:
    correctors = [
        (CorrectorFactory(AnyEagerCorrector, correct_valid_x=False, random_if_multiple=True), 'Eager Rnd', eager_samplers),  # 0
        (CorrectorFactory(AnyEagerCorrector, correct_valid_x=True, random_if_multiple=True), 'Eager Rnd Cval', eager_samplers),
        (CorrectorFactory(GreedyEagerCorrector, correct_valid_x=False, random_if_multiple=False), 'Eager Greedy', eager_samplers),  # 2
        (CorrectorFactory(GreedyEagerCorrector, correct_valid_x=False, random_if_multiple=True), 'Eager Greedy Rnd', eager_samplers),
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=False, random_if_multiple=False, euclidean=False), 'Eager Closest', eager_samplers),  # 4
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=False, random_if_multiple=False, euclidean=True), 'Eager Closest Euc', eager_samplers),
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=False, random_if_multiple=True, euclidean=False), 'Eager Closest Rnd', eager_samplers),
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=False, random_if_multiple=True, euclidean=True), 'Eager Closest Rnd Euc', eager_samplers),
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=True, random_if_multiple=False, euclidean=False), 'Eager Closest Cval', eager_samplers),  # 8
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=True, random_if_multiple=False, euclidean=True), 'Eager Closest Cval Euc', eager_samplers),
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=True, random_if_multiple=True, euclidean=False), 'Eager Closest Cval Rnd', eager_samplers),
        (CorrectorFactory(ClosestEagerCorrector, correct_valid_x=True, random_if_multiple=True, euclidean=True), 'Eager Closest Cval Rnd Euc', eager_samplers),

        (CorrectorFactory(RandomLazyCorrector, correct_valid_x=False), 'Lazy Rnd', lazy_samplers),  # 12
        (CorrectorFactory(RandomLazyCorrector, correct_valid_x=True), 'Lazy Rnd Cval', lazy_samplers),
        (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=False, by_dist=False), 'Lazy Closest', lazy_samplers),  # 14
        (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=True, by_dist=False), 'Lazy Closest Cval', lazy_samplers),
        (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=False, by_dist=True, euclidean=False), 'Lazy Closest Dist', lazy_samplers),  # 16
        (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=False, by_dist=True, euclidean=True), 'Lazy Closest Dist Euc', lazy_samplers),
        (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=True, by_dist=True, euclidean=False), 'Lazy Closest Cval Dist', lazy_samplers),
        (CorrectorFactory(ClosestLazyCorrector, correct_valid_x=True, by_dist=True, euclidean=True), 'Lazy Closest Cval Dist Euc', lazy_samplers),
    ]
    if sbo:
        sbo_eager_samplers = [
            (RepairedSampler(LatinHypercubeSampling()), 'LHS'),
            # (NoGroupingHierarchicalSampling(), 'HierNoGroup'),
            (NrActiveHierarchicalSampling(), 'HierNrAct'),
            # (NrActiveHierarchicalSampling(weight_by_nr_active=True), 'HierNrActWt'),
            # (ActiveVarHierarchicalSampling(), 'HierAct'),
            (ActiveVarHierarchicalSampling(weight_by_nr_active=True), 'HierActWt'),
        ]
        sbo_corr = {
            'Eager Rnd': sbo_eager_samplers,  # Best eager
            'Eager Greedy': [(RepairedSampler(LatinHypercubeSampling()), 'LHS')],  # Greedy LHS --> custom correct_x
            # 'Eager Closest': sbo_eager_samplers,  # Not selected because HierActWt also has other good correctors
            'Eager Closest Euc': sbo_eager_samplers,  # Best eager
            # 'Eager Closest Rnd Euc': sbo_eager_samplers,  # Not selected because samplers also have better correctors
            'Lazy Closest': lazy_samplers,  # Best lazy
        }
        correctors = [(factory, name, sbo_corr[name]) for factory, name, _ in enumerate(correctors) if name in sbo_corr]

    prob_data = {}

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['corr'] = algo_name
        strat_data_['corr_cls'] = algo_name.split(' ')[0]
        strat_data_['corr_type'] = algo_name.split(' ')[1]
        strat_data_['corr_config'] = ' '.join(algo_name.split(' ')[2:-1])
        strat_data_['sampler'] = algo_name.split(' ')[-1]

        data_key = name
        if data_key in prob_data:
            for key, value in prob_data[data_key].items():
                strat_data_[key] = value
            return

        discrete_rates = problem.get_discrete_rates(force=True)

        prob_data[data_key] = data = {
            'is_mo': problem.n_obj > 1,
            'imp_ratio': problem.get_imputation_ratio(),
            'imp_ratio_d': problem.get_discrete_imputation_ratio(),
            'imp_ratio_c': problem.get_continuous_imputation_ratio(),
            'n_discr': problem.get_n_valid_discrete(),
            'n_sub': n_sub,
            'n_doe': n_init,
            'opt_in_small_sub': i_opt > .5*n_sub,
            'max_dr': discrete_rates.loc['diversity'].max(),
            'max_adr': discrete_rates.loc['active-diversity'].max(),
        }
        for key, value in data.items():
            strat_data_[key] = value

    problems = [
        (lambda i_opt_: SelectableTunableBranin(
            n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_, imp_ratio=1., diversity_range=0), '00_SO_NO_HIER', 'Branin ('),
        (lambda i_opt_: SelectableTunableBranin(n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_, diversity_range=0), '01_SO_LDR', 'Branin (H/'),
        (lambda i_opt_: SelectableTunableBranin(n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_), '02_SO_HDR', 'Branin (H/MRD/'),  # High diversity range
        (lambda i_opt_: SelectableTunableZDT1(n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_), '03_MO_HDR', 'ZDT1 (H/MRD/'),
    ]
    # for i, (problem_factory, _, _) in enumerate(problems):
    #     problem_factory(0).print_stats()
    # exit()
    problem_paths = []
    problem_names = []
    prob_name_map = {}
    i_prob = 0
    problem: ArchOptProblemBase
    for i, (problem_factory, category, title) in enumerate(problems):
        for i_opt in i_opt_test:
            problem = problem_factory(i_opt)
            name = f'{category} {problem.__class__.__name__} opt={i_opt}'
            problem_names.append(name)
            prob_name_map[name] = f'{title}{"L" if i_opt == 0 else "S"})'
            problem_path = f'{folder}/{secure_filename(name)}'
            problem_paths.append(problem_path)
            if post_process:
                continue

            n_init = int(np.ceil(doe_k*problem.n_var))
            n_kpls = None
            # n_kpls = n_kpls if problem.n_var > n_kpls else None
            i_prob += 1
            log.info(f'Running optimizations for {i_prob}/{len(problems)*len(i_opt_test)}: {name} '
                     f'(n_init = {n_init}, n_kpls = {n_kpls})')
            problem.pareto_front()

            metrics, additional_plot = _get_metrics(problem)
            additional_plot['delta_hv'] = ['ratio', 'regret', 'delta_hv', 'abs_regret']
            metrics.append(CorrectionTimeMetric())
            additional_plot['corr_time'] = ['mean']

            problems = []
            algorithms = []
            algo_names = []
            for corrector_factory, corr_name, samplers in correctors:
                for cls_sampler, sampler_name in samplers:
                    problem: SelectableTunableMetaProblem = problem_factory(i_opt)
                    problem.corrector_factory = corrector_factory
                    problems.append(problem)

                    if sbo:
                        model, norm = ModelFactory(problem).get_md_kriging_model(
                            kpls_n_comp=n_kpls, ignore_hierarchy=True)
                        infill, n_batch = get_default_infill(problem)
                        sbo_algo = SBOInfill(
                            model, infill, pop_size=100, termination=100, normalization=norm, verbose=True)
                        sbo_algo = sbo_algo.algorithm(infill_size=1, sampler=cls_sampler, init_size=n_init)
                        algorithms.append(sbo_algo)
                    else:
                        algorithms.append(ArchOptNSGA2(pop_size=n_init, sampling=cls_sampler))
                    algo_names.append(f'{corr_name} {sampler_name}')

            do_run = not post_process
            n_eval_max = (n_init+n_infill) if sbo else ((n_gen-1)*n_init)
            exps = run(folder, problems, algorithms, algo_names, n_repeat=n_repeat, n_eval_max=n_eval_max,
                       metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                       run_if_exists=False)
            agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)
            plt.close('all')

    def _add_cols(df_agg_):
        df_agg_['cls_sampler'] = [f'{df_agg_["corr_cls"].values[ii]} {samp}' for ii, samp in enumerate(df_agg_.sampler)]

        analyze_perf_rank(df_agg_, 'delta_hv_abs_regret', n_repeat)
        for corr_cls_ in df_agg_['corr_cls'].unique():
            analyze_perf_rank(df_agg_, 'delta_hv_abs_regret', n_repeat, prefix=corr_cls_,
                              df_subset=df_agg_.corr_cls == corr_cls_)
        for cls_sampler_ in df_agg_.cls_sampler.unique():
            analyze_perf_rank(df_agg_, 'delta_hv_abs_regret', n_repeat, prefix=cls_sampler_,
                              df_subset=df_agg_.cls_sampler == cls_sampler_)

        return df_agg_

    df_agg = agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    # cat_names = [
    #     # 'Random',
    #     'LHS',
    #     'Hier.: No Grouping',
    #     # 'Hier.: By $n_{act}$', 'Hier.: By $n_{act}$ (wt.)',
    #     'Hier.: By $x_{act}$', 'Hier.: By $x_{act}$ (wt.)',
    # ]
    # cat_name_map = {sampler: cat_names[i] for i, (_, sampler) in enumerate(_samplers)}
    cat_name_map = {}
    plot_perf_rank(df_agg, 'corr', cat_name_map=cat_name_map, idx_name_map=prob_name_map,
                   save_path=f'{folder}/rank{folder_post}')
    for corr_cls in df_agg['corr_cls'].unique():
        plot_perf_rank(df_agg[df_agg.corr_cls == corr_cls], 'corr', cat_name_map=cat_name_map,
                       idx_name_map=prob_name_map, save_path=f'{folder}/rank_{corr_cls}{folder_post}',
                       prefix=corr_cls)

    i_best_eager = []
    i_best_all = []
    for cls_sampler in df_agg.cls_sampler.unique():
        i_best_ = plot_perf_rank(df_agg[df_agg.cls_sampler == cls_sampler], 'corr', cat_name_map=cat_name_map,
                                 idx_name_map=prob_name_map, save_path=f'{folder}/rank_{cls_sampler}{folder_post}',
                                 prefix=cls_sampler)

        i_best_glob, = np.where(df_agg.index.get_level_values(1).isin(i_best_))
        i_best_all += list(i_best_glob)
        if cls_sampler.startswith('Eager'):
            i_best_eager += list(i_best_glob)

    best_eager_selector = pd.Series(index=df_agg.index, data=np.in1d(np.arange(len(df_agg)), i_best_eager))
    analyze_perf_rank(df_agg, 'delta_hv_abs_regret', n_repeat, prefix='best_eager',
                      df_subset=best_eager_selector)
    best_eager = plot_perf_rank(df_agg[best_eager_selector], 'corr', cat_name_map=cat_name_map,
                                idx_name_map=prob_name_map, save_path=f'{folder}/rank_best_eager{folder_post}',
                                prefix='best_eager', h_factor=.5)

    best_all_selector = pd.Series(index=df_agg.index, data=np.in1d(np.arange(len(df_agg)), i_best_all))
    analyze_perf_rank(df_agg, 'delta_hv_abs_regret', n_repeat, prefix='best_all',
                      df_subset=best_all_selector)
    plot_perf_rank(df_agg[best_all_selector], 'corr', cat_name_map=cat_name_map,
                   idx_name_map=prob_name_map, save_path=f'{folder}/rank_best_all{folder_post}',
                   prefix='best_all', h_factor=.5)

    ref_df = df_agg[df_agg.index.get_level_values(1) == best_eager[0]]
    df_rel_stats: pd.DataFrame = _sampling_rel_stats_table(
        None, df_agg[best_all_selector], None, ref_df=ref_df, incl_q=True)
    df_rel_stats.to_csv(f'{folder}/best_rel_perf.csv')
    df_rel_stats.to_excel(f'{folder}/best_rel_perf.xlsx')

    df_corr_times = df_agg[best_all_selector]
    df_corr_times['col'] = df_corr_times.index.get_level_values(0)
    df_corr_times['idx'] = df_corr_times.index.get_level_values(1)
    df_corr_times = np.log10(df_corr_times.pivot(columns='col', index='idx', values='corr_time_mean').astype(float))

    def _split_prob_name(p_name):
        prob, spec = p_name[:-3].strip(), p_name[-2:-1]
        if '(' in prob:
            prob += ')'
        return prob, spec

    df_corr_times.columns = pd.MultiIndex.from_tuples(
        [_split_prob_name(prob_name_map.get(col, col)) for col in df_corr_times.columns])
    df_corr_times = df_corr_times.groupby(level=0, axis=1).mean()
    df_corr_times.index = [cat_name_map.get(val, val) for val in df_corr_times.index]

    styler = df_corr_times.style
    styler.format(formatter=lambda v: f'{v:.2f}')
    styler.background_gradient(cmap='Reds', vmin=df_corr_times.min().min(), vmax=df_corr_times.max().max())
    styler.to_latex(f'{folder}/best_rel_perf_corr_time.tex', hrules=True, convert_css=True,
                    column_format='ll'+'c'*len(df_corr_times.columns))

    plt.close('all')


def _sampling_rel_stats_table(folder, df_agg: pd.DataFrame, cat_name_map, ref_df, incl_q=False):
    col_rel_analyze = {'delta_hv_abs_regret': '$\\Delta HV$ regret', 'corr_time_mean': 'Correction time'}
    select_post = ['', '_q25', '_q75'] if incl_q else ['']
    col_rel_sel = [col+post for col in col_rel_analyze for post in select_post]

    df_agg = df_agg[col_rel_sel].astype(float)
    ref_values = ref_df[col_rel_sel].astype(float).values
    rejection_ref = np.repeat(ref_values, len(np.unique(df_agg.index.get_level_values(1))), axis=0)
    col_idx = {col: i for i, col in enumerate(df_agg.columns)}
    for col in df_agg.columns:
        if '_q25' in col or '_q75' in col:
            continue
        if col+'_q25' in col_idx:
            rejection_ref[:, col_idx[col+'_q25']] = rejection_ref[:, col_idx[col]]
            rejection_ref[:, col_idx[col+'_q75']] = rejection_ref[:, col_idx[col]]
    rejection_ref[rejection_ref == 0] = np.nan
    is_num = np.array([df_agg.dtypes[col] == float for col in df_agg.columns])
    rel_values = df_agg.values.copy()
    rel_values[:, is_num] /= rejection_ref[:, is_num]
    df_rel = pd.DataFrame(index=df_agg.index, columns=df_agg.columns, data=rel_values)

    df_rel_agg = df_rel.groupby(level=1).mean()
    df_rel_agg = (df_rel_agg-1)*100
    if cat_name_map is not None:
        df_rel_agg['names'] = [cat_name_map.get(cat, cat) for cat in df_rel_agg.index]
        df_rel_agg = df_rel_agg.set_index('names').loc[[cat for cat in cat_name_map.values()]]
    if incl_q:
        df_rel_agg.columns = pd.MultiIndex.from_tuples([(name, pn) for col, name in col_rel_analyze.items()
                                                        for post, pn in [('', 'mean'), ('_q25', 'min'), ('_q75', 'max')]])
    else:
        df_rel_agg.columns = [col_name for col_name in col_rel_analyze.values()]

    if folder is not None:
        styler = df_rel_agg.style
        styler.format(formatter=lambda v: f'{v:+.0f}\\%')
        styler.background_gradient(cmap='Greens_r', subset=col_rel_analyze['delta_hv_regret'], vmin=-100, vmax=0)
        styler.background_gradient(cmap='Greens_r', subset=col_rel_analyze['fail_rate'], vmin=-100, vmax=0)
        styler.background_gradient(cmap='Reds', subset=col_rel_analyze['time_train'], vmin=0, vmax=200)
        styler.background_gradient(cmap='Reds', subset=col_rel_analyze['time_infill'], vmin=0, vmax=200)
        styler.background_gradient(cmap='Reds', subset=col_rel_analyze['time_train_infill'], vmin=0, vmax=200)
        styler.to_latex(f'{folder}/rel_perf.tex', hrules=True, convert_css=True,
                        column_format='ll'+'c'*len(df_rel_agg.columns))

    return df_rel_agg


def exp_01_06_opt(sbo=True, post_process=False):
    """
    Run optimizations with different sampling strategies for different sub-problem properties and optimum locations:
    - Single-objective, no hierarchy
    - Single-objective, imputation ratio ~= 8, no activeness diversity ratio
    - Single-objective, imputation ratio ~= 8, high activeness diversity ratio
    - Multi-objective, imputation ratio ~= 8, high activeness diversity ratio

    Each is tested for two scenarios:
    - The optimum lies in the largest subproblem
    - The optimum lies in the smallest subproblem

    Conclusions:
    - For non-hierarchical and low-adr problems, the sampler choice is not relevant
    - For high-adr hierarchical problems:
      - The hierarchical samplers have better starting points due to higher chances of sampling smaller subproblems
      - The hierarchical grouped-by-active weighted sampler performs best, most consistently
    """
    folder_post = '' if sbo else '_nsga2'
    folder = set_results_folder(_exp_01_06_folder+folder_post)
    n_infill = 100
    n_repeat = 20 if sbo else 100
    # doe_k, n_gen = 10, 25
    doe_k, n_gen = 5, 50
    # doe_k, n_gen = 2, 100
    # n_sub, n_opts = 8, 2
    n_sub, n_opts = 9, 3
    i_opt_test = [0, n_sub-1]
    opt_offset = .25
    prob_data = {}

    def prob_add_cols(strat_data_, df_strat, algo_name):
        strat_data_['sampler_hier'] = is_hier_sampler = algo_name.startswith('Hier')
        strat_data_['sampler_strat'] = algo_name[4:].replace('Wt', '') if is_hier_sampler else 'NoHier'
        strat_data_['sampler_grp_wt'] = 'Wt' in algo_name

        data_key = name
        if data_key in prob_data:
            for key, value in prob_data[data_key].items():
                strat_data_[key] = value
            return

        discrete_rates = problem.get_discrete_rates(force=True)

        prob_data[data_key] = data = {
            'is_mo': problem.n_obj > 1,
            'imp_ratio': problem.get_imputation_ratio(),
            'imp_ratio_d': problem.get_discrete_imputation_ratio(),
            'imp_ratio_c': problem.get_continuous_imputation_ratio(),
            'n_discr': problem.get_n_valid_discrete(),
            'n_sub': n_sub,
            'n_doe': n_init,
            'opt_in_small_sub': i_opt > .5*n_sub,
            'max_dr': discrete_rates.loc['diversity'].max(),
            'max_adr': discrete_rates.loc['active-diversity'].max(),
        }
        for key, value in data.items():
            strat_data_[key] = value

    problems = [
        (lambda i_opt_: SelectableTunableBranin(
            n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_, imp_ratio=1., diversity_range=0, offset=opt_offset),
         '00_SO_NO_HIER', 'Branin ('),
        (lambda i_opt_: SelectableTunableBranin(
            n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_, diversity_range=0, offset=opt_offset), '01_SO_LDR', 'Branin (H/'),
        (lambda i_opt_: SelectableTunableBranin(
            n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_, offset=opt_offset), '02_SO_HDR', 'Branin (H/MRD/'),  # High diversity range
        (lambda i_opt_: SelectableTunableZDT1(
            n_sub=n_sub, n_opts=n_opts, i_sub_opt=i_opt_, offset=opt_offset), '03_MO_HDR', 'ZDT1 (H/MRD/'),
        # (lambda i_opt_: HierarchicalGoldstein(), '04_OTHER', 'Goldst. (H/MRD/'),
        # (lambda i_opt_: MOHierarchicalGoldstein(), '04_OTHER', 'MO Goldst. (H/MRD/'),
    ]
    # for i, (problem_factory, _, _) in enumerate(problems):
    #     problem_factory(0).print_stats()
    #     problem_factory(0).plot_transformation(show=False)
    #     problem_factory(i_opt_test[-1]).plot_transformation()
    # exit()
    problem_paths = []
    problem_names = []
    prob_name_map = {}
    i_prob = 0
    problem: ArchOptProblemBase
    for i, (problem_factory, category, title) in enumerate(problems):
        for i_opt in i_opt_test:
            problem = problem_factory(i_opt)
            name = f'{category} {problem.__class__.__name__} opt={i_opt}'
            problem_names.append(name)
            prob_name_map[name] = f'{title}{"L" if i_opt == 0 else "S"})'
            problem_path = f'{folder}/{secure_filename(name)}'
            problem_paths.append(problem_path)
            if post_process:
                continue

            n_init = int(np.ceil(doe_k*problem.n_var))
            n_kpls = None
            # n_kpls = n_kpls if problem.n_var > n_kpls else None
            i_prob += 1
            log.info(f'Running optimizations for {i_prob}/{len(problems)*len(i_opt_test)}: {name} '
                     f'(n_init = {n_init}, n_kpls = {n_kpls})')
            problem.pareto_front()

            metrics, additional_plot = _get_metrics(problem)
            additional_plot['delta_hv'] = ['ratio', 'regret', 'delta_hv', 'abs_regret']
            model, norm = ModelFactory(problem).get_md_kriging_model(kpls_n_comp=n_kpls, ignore_hierarchy=True)
            infill, n_batch = get_default_infill(problem)

            algorithms = []
            algo_names = []
            for sampler, sampler_name in _samplers:
                if sbo:
                    sbo_algo = SBOInfill(
                        model, infill, pop_size=100, termination=100, normalization=norm, verbose=True)
                    sbo_algo = sbo_algo.algorithm(infill_size=1, init_sampling=sampler, init_size=n_init)
                    algorithms.append(sbo_algo)
                else:
                    algorithms.append(ArchOptNSGA2(pop_size=n_init, sampling=sampler))
                algo_names.append(sampler_name)

            do_run = not post_process
            n_eval_max = (n_init+n_infill) if sbo else ((n_gen-1)*n_init)
            exps = run(folder, problem, algorithms, algo_names, n_repeat=n_repeat, n_eval_max=n_eval_max,
                       metrics=metrics, additional_plot=additional_plot, problem_name=name, do_run=do_run,
                       run_if_exists=False)
            agg_prob_exp(problem, problem_path, exps, add_cols_callback=prob_add_cols)
            plt.close('all')

    def _add_cols(df_agg_):
        # df_agg_['is_mo'] = ['_MO' in val[0] for val in df_agg_.index]
        df_agg_['strategy'] = [val[1] for val in df_agg_.index]
        analyze_perf_rank(df_agg_, 'delta_hv_abs_regret', n_repeat)
        return df_agg_

    df_agg = agg_opt_exp(problem_names, problem_paths, folder, _add_cols)

    cat_names = [
        'Random',
        'LHS',
        'Hier.: No Grouping',
        'Hier.: By $n_{act}$', 'Hier.: By $n_{act}$ (wt.)',
        'Hier.: By $x_{act}$', 'Hier.: By $x_{act}$ (wt.)',
    ]
    cat_name_map = {sampler: cat_names[i] for i, (_, sampler) in enumerate(_samplers)}
    plot_perf_rank(df_agg, 'strategy', cat_name_map=cat_name_map, idx_name_map=prob_name_map,
                   save_path=f'{folder}/rank{folder_post}', n_col_split=6, h_factor=.6)

    green = matplotlib.cm.get_cmap('Greens')
    blue = matplotlib.cm.get_cmap('Blues')
    orange = matplotlib.cm.get_cmap('Oranges')
    cat_colors = [
        orange(.33), orange(.66),
        blue(.25),
        blue(.5), green(.5),
        blue(.75), green(.75),
    ]
    plot_problem_bars(df_agg, folder, 'strategy', 'delta_hv_abs_regret', prob_name_map=prob_name_map,
                      cat_colors=cat_colors, label_rot=20, label_i=3, cat_names=cat_names, y_log=False, rel=True)

    plt.close('all')


def agg_prob_exp(problem, problem_path, exps, add_cols_callback=None):
    df_data = []
    for exp in exps:
        with open(exp.get_problem_algo_results_path('result_agg_df.pkl'), 'rb') as fp:
            df_strat = pickle.load(fp)
            strat_data = pd.Series(df_strat.iloc[-1, :], name=exp.algorithm_name)
            if add_cols_callback is not None:
                strat_data_ = add_cols_callback(strat_data, df_strat, exp.algorithm_name)
                if strat_data_ is not None:
                    strat_data = strat_data_
            df_data.append(strat_data)

    df_prob = pd.concat(df_data, axis=1).T
    df_prob.to_pickle(f'{problem_path}/df_problem.pkl')
    with pd.ExcelWriter(f'{problem_path}/df_problem.xlsx') as writer:
        df_prob.to_excel(writer)


def agg_opt_exp(problem_names, problem_paths, folder, add_cols_callback):
    df_probs = []
    for i, problem_name in enumerate(problem_names):
        problem_path = problem_paths[i]
        try:
            with open(f'{problem_path}/df_problem.pkl', 'rb') as fp:
                df_prob = pickle.load(fp)
        except FileNotFoundError:
            continue

        df_prob = df_prob.set_index(pd.MultiIndex.from_tuples([(problem_name, val) for val in df_prob.index]))
        df_probs.append(df_prob)

    df_agg = pd.concat(df_probs, axis=0)
    df_agg_ = add_cols_callback(df_agg)
    if df_agg_ is not None:
        df_agg = df_agg_

    try:
        with pd.ExcelWriter(f'{folder}/results.xlsx') as writer:
            df_agg.to_excel(writer)
    except PermissionError:
        pass
    return df_agg


if __name__ == '__main__':
    # exp_01_01_dv_opt_occurrence()
    # exp_01_02_sampling_similarity()
    # exp_01_03_doe_accuracy()
    # exp_01_04_activeness_diversity_ratio()
    # exp_01_05_performance_influence()
    # exp_01_05_correction(sbo=False)
    exp_01_05_correction()
    # exp_01_06_opt()
    # exp_01_06_opt(sbo=False)
