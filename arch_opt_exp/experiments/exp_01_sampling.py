import pickle
import logging
import numpy as np
import pandas as pd
from typing import *
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

_all_problems = [
    SimpleTurbofanArch(),  # Realistic hierarchical problem

    HierarchicalGoldstein(),  # Hierarchical test problem by Pelamatti
    HierarchicalRosenbrock(),  # Hierarchical test problem by Pelamatti

    MOHierarchicalTestProblem(),  # Hierarchical test problem by Bussemaker (2021)

    MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
    CombHierBranin(),  # More realistic hierarchical test problem
    CombHierMORosenbrock(),  # More realistic multi-objective hierarchical test problem
]
_problems = [
    SimpleTurbofanArch(),  # Realistic hierarchical problem
    MDZDT1Small(),  # Non-hierarchical mixed-discrete test problem
    CombHierBranin(),  # More realistic hierarchical test problem
    CombHierMORosenbrock(),  # More realistic multi-objective hierarchical test problem
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
            x = RepairedExhaustiveSampling(n_cont=1).do(problem, 0).get('X')

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
        assert np.all(values == np.arange(len(values)))
        x_count[:len(counts), ix] = counts
    return x_count/x.shape[0]


def exp_01_02_sampling_similarity():
    """
    Investigates which non-exhaustive sampling algorithms best approximate the occurrence rates of exhaustive sampling.

    Hypothesis:
    For non-hierarchical problems it does not matter (probably LHS is better). For hierarchical problems the repaired
    samplers are closer to the exhaustively-sampled occurrence rates.

    Conclusions:
    - Correspondence to exhaustively-sampled rates is measured as max(max(rate-rate_exhaustive per x) over all x)
    - Each sampler was tested with 1000 samples
    - For non-hierarchical problems all samplers perform well: correspondence is < 3%
    - For hierarchical problems:
      - Non-repaired samplers always perform bad: 20% < correspondence < 33%
      - Repaired samplers perform well: correspondence < 4%
      - Repaired samplers do this by exhaustively-sampling all discrete design vectors and then randomly sampling these
      - A safeguard against time/memory usage is implemented; if triggered, they perform as non-repaired samplers
    """
    exp1_folder = set_results_folder(_exp_01_01_folder)
    folder = set_results_folder(_exp_01_02_folder)
    n_samples = 1000

    df_exhaustive: List[pd.DataFrame] = []
    for i, problem in enumerate(_problems):
        path = f'{exp1_folder}/df_{secure_filename(repr(problem))}.pkl'
        with open(path, 'rb') as fp:
            df_exhaustive.append(pickle.load(fp))

    samplers = [
        (False, FloatRandomSampling()),
        (False, MixedVariableSampling()),
        (False, LatinHypercubeSampling()),
        (True, RepairedRandomSampling()),
        (True, RepairedLatinHypercubeSampling()),
    ]
    for i, (is_repaired, sampler) in enumerate(samplers):
        with pd.ExcelWriter(f'{folder}/output_{i}_{sampler.__class__.__name__}.xlsx') as writer:
            for j, problem in enumerate(_problems):
                log.info(f'Sampling {problem!r} ({j+1}/{len(_problems)}) '
                         f'with sampler: {sampler!r} ({i+1}/{len(samplers)})')
                x = sampler.do(problem, n_samples).get('X')

                if isinstance(sampler, MixedVariableSampling):
                    x = np.array([[row[x_name] for x_name in problem.vars.keys()] for i, row in enumerate(x)])

                if not is_repaired:
                    x = ArchOptRepair().do(problem, x)

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

                x_rel = np.row_stack([x_rel, x_rel_diff, x_max_diff])
                cols = [f'x{ix}{" (c)" if problem.is_cont_mask[ix] else ""}' for ix in range(x_rel.shape[1])]

                x_rel = np.column_stack([x_rel, np.max(np.abs(x_rel), axis=1)])
                cols.append('max')

                df = pd.DataFrame(index=index, data=x_rel, columns=cols)
                df.to_excel(writer, sheet_name=repr(problem))


if __name__ == '__main__':
    # exp_01_01_dv_opt_occurrence(), exit()
    exp_01_02_sampling_similarity(), exit()
