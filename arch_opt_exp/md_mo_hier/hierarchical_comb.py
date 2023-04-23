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

This test suite contains a set of mixed-discrete, constrained, hierarchical, multi-objective problems.
"""
import itertools
import numpy as np
from typing import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.core.variable import Real, Integer, Choice

__all__ = ['CombinatorialHierarchicalMetaProblem', 'CombHierBranin', 'CombHierMO', 'CombHierDMO']


class CombinatorialHierarchicalMetaProblem(HierarchyProblemBase):
    """
    Meta problem that turns any (mixed-discrete, multi-objective) problem into a hierarchical optimization problem:
    - The problem is separated into n_parts per dimension:
      - Continuous variables are simply separated linearly
      - Discrete variables are separated such that there are at least 2 options in each part
    - Categorical selection variables are added to select subparts to apply divided variables to
      - Parts are selected ordinally (i.e. not per dimension, but from "one big list"); orders are randomized
      - For each subpart one or more original dimensions are deactivated
      - Selection variables are added for selecting which subpart to evaluate, but repeated separation into groups

    Note: the underlying problem should not be hierarchical!

    Settings:
    - n_parts: number of parts to separate each dimension; increases the nr of possible discrete design points
    - n_sel_dv: number of design variables to use for selecting which part to evaluate; higher values increase the
                imputation ratio, reduce the nr of options for each selection design variable
    - sep_power: controls how non-uniform the selection subdivisions are: higher than 1 increases imputation ratio and
                 difference between occurrence rates
    - target_n_opts_ratio: controls the nr of options of the last separation variable: higher reduces the imp ratio
    """

    def __init__(self, problem: ArchOptTestProblemBase, n_parts=2, n_sel_dv=4, sep_power=1.2, target_n_opts_ratio=5.,
                 repr_str=None):
        self._problem = problem
        self._n_parts = n_parts
        self._repr_str = repr_str

        # Separate the underlying design space in different parts
        parts = []
        is_cont_mask, xl, xu = problem.is_cont_mask, problem.xl, problem.xu
        div_bounds = 1/n_parts
        n_opts_max = np.zeros((problem.n_var,), dtype=int)
        for i_div in itertools.product(*[range(n_parts) for _ in range(problem.n_var)]):
            part = []
            for i_dv, i in enumerate(i_div):
                if is_cont_mask[i_dv]:
                    # Map continuous variable to a subrange
                    xl_i, xu_i = xl[i_dv], xu[i_dv]
                    bounds = tuple(np.array([i, i+1])*div_bounds*(xu_i-xl_i)+xl_i)
                    part.append((False, bounds))

                else:
                    # Map discrete variable to a subrange that exists of at least 2 options
                    n_opts = int(xu[i_dv]+1)
                    n_opt_per_div = max(2, np.ceil(n_opts / n_parts))
                    i_opts = np.arange(n_opt_per_div*i, n_opt_per_div*(i+1))

                    # Ensure that no options outside the bounds can be selected
                    i_opts = tuple(i_opts[i_opts < n_opts])
                    if len(i_opts) == 0:
                        # If too far outside the bounds, retain the last option only
                        i_opts = (n_opts-1,)

                    # Track the maximum nr of options
                    if len(i_opts) > n_opts_max[i_dv]:
                        n_opts_max[i_dv] = len(i_opts)

                    part.append((True, i_opts))
            parts.append(part)

        # Shuffle the parts
        n_sel_dv = max(2, n_sel_dv)
        rng = np.random.default_rng(problem.n_var * problem.n_obj * n_parts * n_sel_dv)
        self._parts = parts = [parts[i] for i in rng.permutation(np.arange(len(parts)))]

        # Define which mapped design variables are active for each part
        self._parts_is_active = parts_is_active = np.ones((len(parts), problem.n_var), dtype=bool)
        self._parts_is_discrete = parts_is_discrete = np.zeros((len(parts), problem.n_var), dtype=bool)
        self._parts_n_opts = parts_n_opts = np.ones((len(parts), problem.n_var), dtype=int)
        osc_period = max(5, int(len(parts)/5))
        idx_deactivate = np.where(is_cont_mask)[0]
        if len(idx_deactivate) == 0:
            idx_deactivate = np.arange(problem.n_var)
        n_inactive = np.floor(len(idx_deactivate)*(.5-.5*np.cos(osc_period*np.arange(len(parts))/np.pi))).astype(int)
        for i, part in enumerate(parts):
            # Discrete variables with one option only are inactive
            for i_dv, (is_discrete, settings) in enumerate(part):
                if is_discrete:
                    parts_is_discrete[i, i_dv] = True
                    parts_n_opts[i, i_dv] = len(settings)
                    if len(settings) < 2:
                        parts_is_active[i, i_dv] = False

            # Deactivate continuous variables based on an oscillating equation
            inactive_idx = idx_deactivate[len(idx_deactivate)-n_inactive[i]:]
            parts_is_active[i, inactive_idx] = False

        parts_n_opts[~parts_is_active] = 1

        # Define selection design variables
        # Repeatedly separate the number of parts to be selected
        init_range = .65  # Directly controls the range (= difference between min and max occurrence rates) of x0

        def _sep_group(n_values, n_sep):
            return np.round((np.linspace(0, 1, n_values)**sep_power)*(n_sep-.01)-.5).astype(int)

        n = len(parts)

        # We define one design variable that makes the initial separation
        init_sep_frac = .5+.5*init_range-.05+.1*rng.random()

        # Determine in how many groups we should separate at each step
        # Repeatedly subdividing the largest group determines how many values the last largest group has. We calculate
        # for several separation numbers what the fraction of the largest group is; then calculate how big the latest
        # largest group is. If this number is equal to the corresponding nr of separations, it means that each design
        # variable will have the same nr of options. We choose the nr of separations where the ratio is a bit higher
        # than 1 to introduce another source of non-uniformity in the problem formulation
        n_sep_possible = np.arange(2, max(2, np.floor(.25*n))+1, dtype=int)
        frac_largest_group = np.array([np.sum(_sep_group(100, n_sep) == 0)/100 for n_sep in n_sep_possible])
        n_rel_last_group = (init_sep_frac*frac_largest_group**(n_sel_dv-2))*n/n_sep_possible
        n_rel_lg_idx = np.where(n_rel_last_group > target_n_opts_ratio)[0]
        n_sep_per_dv = n_sep_possible[n_rel_lg_idx[-1]] if len(n_rel_lg_idx) > 0 else n_sep_possible[0]

        x_sel = np.zeros((n, n_sel_dv), dtype=int)
        is_active_sel = np.ones((n, n_sel_dv), dtype=bool)
        dv_groups = [np.arange(n, dtype=int)]
        for i in range(n_sel_dv):
            # Separate current selection groups
            next_dv_groups = []
            needed_separation = False
            for group_idx in dv_groups:
                if len(group_idx) == 1:
                    is_active_sel[group_idx[0], i:] = False
                    continue
                needed_separation = True

                # For the last group just add uniformly increasing values to avoid needing additional groups
                if i == n_sel_dv-1:
                    x_sel[group_idx, i] = np.arange(len(group_idx))
                    continue

                # For the first group separate based on the initial separation fraction
                if i == 0:
                    x_next_group = np.ones((len(group_idx),), dtype=int)
                    n_first_group = int(np.ceil(init_sep_frac*len(x_next_group)))
                    x_next_group[:n_first_group] = 0
                else:
                    # Distribute values unevenly (raising the power results in a more uneven distribution)
                    x_next_group = _sep_group(len(group_idx), n_sep_per_dv)

                x_sel[group_idx, i] = x_next_group

                # Determine next groups
                for x_next in np.unique(x_next_group):
                    next_dv_groups.append(group_idx[x_next_group == x_next])

            dv_groups = next_dv_groups
            if not needed_separation:
                x_sel = x_sel[:, :i]
                is_active_sel = is_active_sel[:, :i]
                break

        self._x_sel = x_sel
        self._is_active_sel = is_active_sel
        des_vars = []
        for i in range(x_sel.shape[1]):
            des_vars.append(Choice(options=list(sorted(np.unique(x_sel[:, i])))))

        # Add mapped design variables
        for i_dv, des_var in enumerate(problem.des_vars):
            if isinstance(des_var, Real):
                des_vars.append(Real(bounds=des_var.bounds))
            elif isinstance(des_var, Integer):
                des_vars.append(Integer(bounds=(0, n_opts_max[i_dv]-1)))
            elif isinstance(des_var, Choice):
                des_vars.append(Choice(options=list(range(n_opts_max[i_dv]))))
            else:
                raise RuntimeError(f'Design variable type not supported: {des_var!r}')

        super().__init__(des_vars, n_obj=problem.n_obj, n_ieq_constr=problem.n_ieq_constr,
                         n_eq_constr=problem.n_eq_constr)

        self.__correct_output = {}

    def _get_n_valid_discrete(self) -> int:
        # Sum the nr of combinations for the parts
        return int(np.sum(np.prod(self._parts_n_opts, axis=1)))

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # We already know the possible design vectors for selecting the different parts
        x_sel = self._x_sel
        x_discrete = np.zeros((x_sel.shape[0], self.n_var), dtype=int)
        n_dv_map = x_sel.shape[1]
        x_discrete[:, :n_dv_map] = x_sel.copy()

        is_act_discrete = np.ones(x_discrete.shape, dtype=bool)
        is_act_discrete[:, :n_dv_map] = self._is_active_sel.copy()

        # Expand for active discrete variables in the parts
        x_expanded = []
        is_act_expanded = []
        for i, part in enumerate(self._parts):
            # Check which discrete design variables are active, and get possible values
            is_active_part = self._parts_is_active[i, :]
            part_ix_discrete = np.where(self._parts_is_discrete[i, :])[0]
            discrete_opts = [list(range(n_opts)) for n_opts in self._parts_n_opts[i, part_ix_discrete]]

            x_sel_part = x_discrete[[i], :]
            is_act_sel_part = is_act_discrete[[i], :]
            is_act_sel_part[:, n_dv_map:] = is_active_part

            # if np.sum((self._parts_n_opts[i, part_ix_discrete] > 1) & ~is_active_part[part_ix_discrete]) > 0:
            #     raise RuntimeError(f'Inconsistent activeness vector and nr of options!')

            # Get cartesian product of values and expand
            if len(part_ix_discrete) > 0:
                part_x_discrete = np.array(list(itertools.product(*discrete_opts)))
                x_sel_part = np.repeat(x_sel_part, part_x_discrete.shape[0], axis=0)
                x_sel_part[:, n_dv_map+part_ix_discrete] = part_x_discrete
                is_act_sel_part = np.repeat(is_act_sel_part, part_x_discrete.shape[0], axis=0)

            x_expanded.append(x_sel_part)
            is_act_expanded.append(is_act_sel_part)

        x_discrete_all = np.row_stack(x_expanded)
        is_act_discrete_all = np.row_stack(is_act_expanded)
        return x_discrete_all, is_act_discrete_all

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        # Correct and impute
        self._correct_x_impute(x, is_active_out)
        i_part_selected = self.__correct_output['i_part_sel']

        parts = self._parts
        parts_is_active = self._parts_is_active
        n_dv_map = self._x_sel.shape[1]
        xl, xu = self._problem.xl, self._problem.xu

        # Map design variables to underlying problem
        x_underlying = x[:, n_dv_map:].copy()
        for i, i_part in enumerate(i_part_selected):
            is_active_i = parts_is_active[i_part, :]
            x_part = x_underlying[i, :]
            for i_dv, (is_discrete, settings) in enumerate(parts[i_part]):
                if is_discrete:
                    if is_active_i[i_dv]:
                        i_x_mapped = int(x_part[i_dv])
                        x_part[i_dv] = settings[i_x_mapped] if i_x_mapped < len(settings) else settings[-1]
                    else:
                        x_part[i_dv] = 0
                else:
                    bnd = settings
                    if is_active_i[i_dv]:
                        x_part[i_dv] = bnd[0]+(bnd[1]-bnd[0])*((x_part[i_dv]-xl[i_dv])/(xu[i_dv]-xl[i_dv]))
                    else:
                        x_part[i_dv] = .5*np.sum(bnd)

        # Evaluate underlying problem
        out = self._problem.evaluate(x_underlying, return_as_dictionary=True)
        if np.any(out['is_active'] == 0):
            raise RuntimeError('Underlying problem should not be hierarchical!')

        f_out[:, :] = out['F']
        if 'G' in out:
            g_out[:, :] = out['G']
        if 'H' in out:
            h_out[:, :] = out['H']

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        # Match to selection design vector
        x_sel = self._x_sel
        is_act_sel = self._is_active_sel
        n_dv_sel = x_sel.shape[1]
        i_part_selected = np.zeros((x.shape[0],), dtype=int)
        for i, xi in enumerate(x):

            # Recursively select design vectors matching ours
            idx_match = np.arange(x_sel.shape[0], dtype=int)
            for i_sel in range(n_dv_sel):
                idx_match_i = idx_match[x_sel[idx_match, i_sel] == xi[i_sel]]

                # If none found, we impute
                if len(idx_match_i) == 0:
                    xi[i_sel] = imp_dv_value = x_sel[idx_match[-1], i_sel]
                    idx_match_i = idx_match[x_sel[idx_match, i_sel] == imp_dv_value]

                # If one found, we have a match!
                if len(idx_match_i) == 1:
                    i_part = idx_match_i[0]
                    i_part_selected[i] = i_part
                    xi[:n_dv_sel] = x_sel[i_part, :]
                    is_active[i, :n_dv_sel] = is_act_sel[i_part, :]
                    break

                # Otherwise, we continue searching
                idx_match = idx_match_i
            else:
                raise RuntimeError(f'Could not match design vectors: {xi[:n_dv_sel]}')

        # Correct DVs of underlying problem and set activeness
        n_dv_map = x_sel.shape[1]
        part_is_active = self._parts_is_active
        part_is_discrete = self._parts_is_discrete
        part_n_opts = self._parts_n_opts
        for i, i_part in enumerate(i_part_selected):
            is_active[i, n_dv_map:] = part_is_active[i_part, :]

            # Correct upper bounds of discrete variables
            is_discrete = part_is_discrete[i_part, :]
            n_opts = part_n_opts[i_part, is_discrete]
            i_x_discrete = np.where(is_discrete)[0]+n_dv_map
            x_discrete_part = x[i, i_x_discrete]
            for i_opt, n_opt in enumerate(n_opts):
                if x_discrete_part[i_opt] >= n_opt:
                    x_discrete_part[i_opt] = n_opt-1

            x[i, i_x_discrete] = x_discrete_part

        self.__correct_output = {'i_part_sel': i_part_selected}

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return f'{self.__class__.__name__}()'


class CombHierBranin(CombinatorialHierarchicalMetaProblem):
    """Single-objective mixed-discrete hierarchical Branin test problem"""

    def __init__(self):
        super().__init__(MDBranin(), n_parts=4, n_sel_dv=4, sep_power=1.2, target_n_opts_ratio=5.)


class CombHierMO(CombinatorialHierarchicalMetaProblem):
    """Multi-objective mixed-discrete hierarchical test problem"""

    def __init__(self):
        problem = MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=3, n_vars_int=2)
        super().__init__(problem, n_parts=3, n_sel_dv=5, sep_power=1.1, target_n_opts_ratio=1.)


class CombHierDMO(CombinatorialHierarchicalMetaProblem):
    """Multi-objective discrete hierarchical test problem"""

    def __init__(self):
        problem = MixedDiscretizerProblemBase(OmniTest(n_var=6), n_opts=5)
        super().__init__(problem, n_parts=2, n_sel_dv=5, sep_power=1., target_n_opts_ratio=5.)


if __name__ == '__main__':
    # CombHierBranin().print_stats()
    # CombHierBranin().plot_pf()
    CombHierMO().print_stats()
    # CombHierMO().reset_pf_cache()
    # CombHierMO().plot_pf()
    CombHierDMO().print_stats()
    # CombHierDMO().reset_pf_cache()
    # CombHierDMO().plot_pf()

