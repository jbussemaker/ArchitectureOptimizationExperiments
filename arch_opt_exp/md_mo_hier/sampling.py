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
import itertools
import numpy as np
from typing import *
from scipy.spatial.distance import cdist
from pymoo.core.sampling import Sampling
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.design_space import *

__all__ = ['HierarchicalSamplingTestBase', 'NoGroupingHierarchicalSampling', 'NrActiveHierarchicalSampling',
           'ActiveVarHierarchicalSampling', 'RepairedSampler', 'HierarchicalCoveringSampling',
           'HierarchicalActSepSampling', 'HierarchicalSobolSampling', 'HierarchicalDirectSampling',
           'HierarchicalRandomSampling', 'ArchVarHierarchicalSampling', 'MRDHierarchicalSampling']


class RepairedSampler(Sampling):
    """Wraps another sampler and repairs generated samples"""

    def __init__(self, sampler: Sampling, repair=None):
        self._sampler = sampler
        super().__init__()
        self.repair = ArchOptRepair() if repair is None else repair

    def _do(self, problem, n_samples, **kwargs):
        x = self._sampler.do(problem, n_samples, **kwargs).get('X')
        x = self.repair.do(problem, x)
        x = x[~LargeDuplicateElimination.eliminate(x), :]
        return x


class HierarchicalSamplingTestBase(HierarchicalSampling):
    """Base class for testing random sampling: groups and weights discrete vectors"""

    def __init__(self, weight_by_nr_active=False, weight_by_group_size=False, sobol=True):
        self.weight_by_nr_active = weight_by_nr_active
        self.weight_by_group_size = weight_by_group_size
        self.n_iter = 10
        super().__init__(sobol=sobol)

    def _get_group_weights(self, groups: List[np.ndarray], is_act_all: np.ndarray) -> List[float]:

        if self.weight_by_nr_active:
            nr_active = np.sum(is_act_all, axis=1)
            avg_nr_active = [np.sum(nr_active[group])/len(group) for group in groups]
            return avg_nr_active

        if self.weight_by_group_size:
            exponent = .5
            return [(len(group))**exponent for group in groups]

        # Uniform sampling
        return [1.]*len(groups)

    def _sample_discrete_from_group(self, x_group: np.ndarray, is_act_group: np.ndarray, n_sel: int, choice_func,
                                    has_x_cont: bool) -> np.ndarray:
        n_in_group = x_group.shape[0]
        n_sel = n_sel
        i_x_selected = np.array([], dtype=int)
        while n_sel >= n_in_group:
            if n_sel == n_in_group or not has_x_cont:
                return np.concatenate([i_x_selected, np.arange(n_in_group)])

            i_x_selected = np.concatenate([i_x_selected, np.arange(n_in_group)])
            n_sel = n_sel-n_in_group

        i_x_tries = []
        metrics = []
        for _ in range(self.n_iter):
            i_x_try = choice_func(n_sel, n_in_group, replace=False)
            i_x_tries.append(i_x_try)

            x_try = x_group[i_x_try, :]
            dist = cdist(x_try, x_try, metric='cityblock')
            np.fill_diagonal(dist, np.nan)

            min_dist = np.nanmin(dist)
            median_dist = np.nanmean(dist)
            metrics.append((min_dist, median_dist))

        i_best = sorted(range(len(metrics)), key=metrics.__getitem__)[-1]
        return np.concatenate([i_x_selected, i_x_tries[i_best]])

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        """Separate design vectors into subproblem groups; should return a list of indices"""
        raise NotImplementedError

    def get_merged_x(self, problem: ArchOptProblemBase):
        is_cont_mask = problem.is_cont_mask
        x_all, is_act_all = self.get_hierarchical_cartesian_product(problem, self._repair)

        groups = self.group_design_vectors(x_all, is_act_all, is_cont_mask)
        weights = np.array(self._get_group_weights(groups, is_act_all))

        x_groups = np.zeros((x_all.shape[0],))
        x_weights = np.zeros((x_all.shape[0],))
        for i, i_grp in enumerate(groups):
            x_groups[i_grp] = i
            x_weights[i_grp] = weights[i]

        x_merged = x_all.copy()
        x_merged[~is_act_all] = -1
        return np.column_stack([x_groups, x_weights, x_merged])


class HierarchicalRandomSampling(HierarchicalSamplingTestBase):
    """Applies the fallback random sampling implemented in the hierarchical sampler"""

    @classmethod
    def get_hierarchical_cartesian_product(cls, *args, **kwargs):
        return None, None

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        raise RuntimeError('Should not be called')


class NoGroupingHierarchicalSampling(HierarchicalSamplingTestBase):
    """Applies no grouping: uniformly sample from all available discrete design vectors"""

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        return [np.arange(x_all.shape[0])]


class NrActiveHierarchicalSampling(HierarchicalSamplingTestBase):
    """Groups by nr of active design variables"""

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        n_active = np.sum(is_act_all, axis=1)
        n_active_unique, unique_indices = np.unique(n_active, return_inverse=True)
        return [np.where(unique_indices == i)[0] for i in range(len(n_active_unique))]


class ActiveVarHierarchicalSampling(HierarchicalSamplingTestBase):
    """Groups by active variables"""

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        is_active_unique, unique_indices = np.unique(is_act_all, axis=0, return_inverse=True)
        return [np.where(unique_indices == i)[0] for i in range(len(is_active_unique))]


class ArchVarHierarchicalSampling(HierarchicalSamplingTestBase):
    """Groups by problem-specific architecture variables"""

    def __init__(self, arch_var_idx, **kwargs):
        super().__init__(**kwargs)
        self.arch_var_idx = arch_var_idx

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        is_active_unique, unique_indices = np.unique(x_all[:, self.arch_var_idx], axis=0, return_inverse=True)
        return [np.where(unique_indices == i)[0] for i in range(len(is_active_unique))]


class MRDHierarchicalSampling(HierarchicalSamplingTestBase):
    """Groups by recursively inspecting max rate diversity"""

    def __init__(self, high_rd_split=.8, low_rd_split=.5, **kwargs):
        super().__init__(**kwargs)
        self.high_rd_split = high_rd_split
        self.low_rd_split = low_rd_split

    def group_design_vectors(self, x_all: np.ndarray, is_act_all: np.ndarray, is_cont_mask) -> List[np.ndarray]:
        is_discrete_mask = ~is_cont_mask
        high_rd_split = self.high_rd_split
        low_rd_split = self.low_rd_split
        i_low_rd_split = None

        def recursive_get_groups(group_i: np.ndarray) -> List[np.ndarray]:
            nonlocal i_low_rd_split

            x_grp = x_all[group_i, :]
            x_min = np.min(x_grp, axis=0).astype(int)
            is_act_grp = is_act_all[group_i, :]
            counts, diversity, active_diversity, i_opts = \
                ArchDesignSpace.calculate_discrete_rates_raw(x_grp - x_min, is_act_grp, is_discrete_mask)

            # Check low split rate
            xi_split = None
            if low_rd_split is not None:
                rd_split_rates, = np.where(active_diversity >= low_rd_split)
                if i_low_rd_split is None:  # If no low-split variable has been set
                    if len(rd_split_rates) == 0:
                        i_low_rd_split = -1  # Set to "no low-split var"
                    else:
                        i_low_rd_split = rd_split_rates[0]  # Choose first var
                        xi_split = rd_split_rates[0]

                elif i_low_rd_split != -1 and len(rd_split_rates) > 0 and rd_split_rates[0] == i_low_rd_split:
                    xi_split = rd_split_rates[0]

            # Check high split rate
            if xi_split is None:
                rd_split_rates, = np.where(active_diversity >= high_rd_split)
                if len(rd_split_rates) == 0:
                    return [group_i]
                xi_split = rd_split_rates[0]

            opt_rates = counts[1:, xi_split]
            i_opt_min = np.nanargmin(opt_rates) + x_min[xi_split]

            min_rate_group = x_grp[:, xi_split] == i_opt_min
            group_i_min = group_i[min_rate_group]
            group_i_other = group_i[~min_rate_group]

            return recursive_get_groups(group_i_min) + recursive_get_groups(group_i_other)

        return recursive_get_groups(np.arange(x_all.shape[0]))


class HierarchicalActSepSampling(HierarchicalSampling):

    def __init__(self):
        super().__init__(sobol=False)

    def _sample_discrete_x(self, n_samples: int, is_cont_mask, x_all: np.ndarray, is_act_all: np.ndarray, sobol=False):

        def _choice(n_choose, n_from, replace=True):
            return self._choice(n_choose, n_from, replace=replace, sobol=sobol)

        # Separate by nr of active discrete variables
        x_all_grouped, is_act_all_grouped, i_x_groups = self.split_by_discrete_n_active(x_all, is_act_all, is_cont_mask)

        # Uniformly choose from which group to sample
        i_groups = np.sort(_choice(n_samples, len(x_all_grouped)))
        x = []
        is_active = []
        has_x_cont = np.any(is_cont_mask)
        i_x_sampled = np.ones((x_all.shape[0],), dtype=bool)
        for i_group in range(len(x_all_grouped)):
            i_x_group = np.where(i_groups == i_group)[0]
            if len(i_x_group) == 0:
                continue

            # Randomly select values within group
            x_group = x_all_grouped[i_group]
            if len(i_x_group) < x_group.shape[0]:
                i_x = _choice(len(i_x_group), x_group.shape[0], replace=False)

            # If there are more samples requested than points available, only repeat points if there are continuous vars
            elif has_x_cont:
                i_x_add = _choice(len(i_x_group)-x_group.shape[0], x_group.shape[0])
                i_x = np.sort(np.concatenate([np.arange(x_group.shape[0]), i_x_add]))
            else:
                i_x = np.arange(x_group.shape[0])

            x.append(x_group[i_x, :])
            is_active.append(is_act_all_grouped[i_group][i_x, :])
            i_x_sampled[i_x_groups[i_group][i_x]] = True

        x = np.row_stack(x)
        is_active = np.row_stack(is_active)

        # Uniformly add discrete vectors if there are not enough (can happen if some groups are very small and there
        # are no continuous dimensions)
        if x.shape[0] < n_samples:
            n_add = n_samples-x.shape[0]
            x_available = x_all[~i_x_sampled, :]
            is_act_available = is_act_all[~i_x_sampled, :]

            if n_add < x_available.shape[0]:
                i_x = _choice(n_add, x_available.shape[0], replace=False)
            else:
                i_x = np.arange(x_available.shape[0])

            x = np.row_stack([x, x_available[i_x, :]])
            is_active = np.row_stack([is_active, is_act_available[i_x, :]])

        return x, is_active

    @staticmethod
    def split_by_discrete_n_active(x_discrete: np.ndarray, is_act_discrete: np.ndarray, is_cont_mask) \
            -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        # Calculate nr of active variables for each design vector
        is_discrete_mask = ~is_cont_mask
        n_active = np.sum(is_act_discrete[:, is_discrete_mask], axis=1)

        # Sort by nr active
        i_sorted = np.argsort(n_active)
        x_discrete = x_discrete[i_sorted, :]
        is_act_discrete = is_act_discrete[i_sorted, :]

        # Split by nr active
        # https://stackoverflow.com/a/43094244
        i_split = np.unique(n_active[i_sorted], return_index=True)[1][1:]
        x_all_grouped = np.split(x_discrete, i_split, axis=0)
        is_act_all_grouped = np.split(is_act_discrete, i_split, axis=0)
        i_x_groups = np.split(np.arange(x_discrete.shape[0]), i_split)

        return x_all_grouped, is_act_all_grouped, i_x_groups


class HierarchicalSobolSampling(HierarchicalSampling):

    def __init__(self):
        super().__init__(sobol=True)

    def _sample_discrete_x(self, n_samples: int, is_cont_mask, x_all: np.ndarray, is_act_all: np.ndarray, sobol=False):
        has_x_cont = np.any(is_cont_mask)

        x = x_all
        if n_samples < x.shape[0]:
            i_x = self._choice(n_samples, x.shape[0], replace=False, sobol=sobol)
        elif has_x_cont:
            # If there are more samples requested than points available, only repeat points if there are continuous vars
            i_x_add = self._choice(n_samples - x.shape[0], x.shape[0], sobol=sobol)
            i_x = np.sort(np.concatenate([np.arange(x.shape[0]), i_x_add]))
        else:
            i_x = np.arange(x.shape[0])

        x = x[i_x, :]
        is_active = is_act_all[i_x, :]
        return x, is_active


class HierarchicalDirectSampling(HierarchicalSobolSampling):
    """Directly sample from all available discrete design vectors"""

    def __init__(self):
        super().__init__()
        self.sobol = False


class HierarchicalCoveringSampling(ActiveVarHierarchicalSampling):
    """
    Sampler that generates covering arrays for sampling design vectors with as many pairwise combinations as
    possible. Introduction to covering arrays:
    http://hardlog.udl.cat/static/doc/ctlog/html/index.html
    """

    def _sample_discrete_x(self, n_samples: int, is_cont_mask, x_all: np.ndarray, is_act_all: np.ndarray, sobol=False):
        if n_samples > x_all.shape[0]:
            return super()._sample_discrete_x(n_samples, is_cont_mask, x_all, is_act_all, sobol=sobol)

        # Get optimally-covering samples
        i_covering = self._get_covering_samples(n_samples, is_cont_mask, x_all)
        if len(i_covering) > n_samples:
            i_covering = np.random.choice(i_covering, n_samples, replace=False)
        x_sampled, is_act_sampled = x_all[i_covering, :], is_act_all[i_covering, :]

        # Sample additional design vectors from non-sampled vectors
        if x_sampled.shape[0] < n_samples:
            n_extra = n_samples-x_sampled.shape[0]
            i_not_sampled = np.delete(np.arange(x_all.shape[0]), i_covering)
            x_extra, is_act_extra = super()._sample_discrete_x(
                n_extra, is_cont_mask, x_all[i_not_sampled, :], is_act_all[i_not_sampled, :], sobol=sobol)

            x_sampled = np.row_stack([x_sampled, x_extra])
            is_act_sampled = np.row_stack([is_act_sampled, is_act_extra])

        return x_sampled, is_act_sampled

    @staticmethod
    def _get_covering_samples(n_samples_target: int, is_cont_mask, x_all: np.ndarray) -> np.ndarray:

        ix_discr = np.where(~is_cont_mask)[0]
        x_all_discr = x_all[:, ix_discr].astype(np.int)
        if len(ix_discr) == 1:
            _, comb_idx = np.unique(x_all_discr, axis=0, return_index=True)
            return comb_idx

        comb_tuples = {}
        for i, j in itertools.combinations(range(x_all_discr.shape[1]), 2):
            unique_combs, comb_idx = np.unique(x_all_discr[:, [i, j]], axis=0, return_inverse=True)
            comb_tuples_ij = {tuple(comb): i_unique for i_unique, comb in enumerate(unique_combs)}
            comb_tuples[i, j] = (comb_tuples_ij, comb_idx)

        n_best = i_sampled_best = None
        for _ in range(50):
            i_sampled = np.random.choice(x_all_discr.shape[0], n_samples_target, replace=False)
            x_discr_sampled = x_all_discr[i_sampled, :]

            i_combs = np.zeros((i_sampled.shape[0], len(comb_tuples)), dtype=int)
            for i_comb, ((i, j), (comb_tuples_ij, _)) in enumerate(comb_tuples.items()):
                for i_sample, x_ij_sampled in enumerate(x_discr_sampled[:, [i, j]]):
                    i_combs[i_sample, i_comb] = comb_tuples_ij[tuple(x_ij_sampled)]

            n_combinations = sum([len(set(i_combs[:, i_comb])) for i_comb in range(i_combs.shape[1])])
            if n_best is None or n_combinations > n_best:
                n_best = n_combinations
                i_sampled_best = i_sampled

        return i_sampled_best


if __name__ == '__main__':
    from sb_arch_opt.problems.hierarchical import *
    HierarchicalCoveringSampling().do(HierCarside(), 20)
