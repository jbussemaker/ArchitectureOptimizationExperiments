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
import random
import numpy as np
from typing import *
from pymoo.core.sampling import Sampling
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *

__all__ = ['HierarchicalSamplingTestBase', 'NoGroupingHierarchicalSampling', 'NrActiveHierarchicalSampling',
           'ActiveVarHierarchicalSampling', 'RepairedSampler',
           'HierarchicalActSepSampling', 'HierarchicalSobolSampling', 'HierarchicalDirectSampling']


class RepairedSampler(Sampling):
    """Wraps another sampler and repairs generated samples"""

    def __init__(self, sampler: Sampling):
        self._sampler = sampler
        self.repair = ArchOptRepair()
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        x = self._sampler.do(problem, n_samples, **kwargs).get('X')
        x = ArchOptRepair().do(problem, x)
        x = x[~LargeDuplicateElimination.eliminate(x), :]
        return x


class HierarchicalSamplingTestBase(HierarchicalSampling):
    """Base class for testing random sampling: groups and weights discrete vectors"""

    def __init__(self, weight_by_nr_active=False, sobol=True):
        self.weight_by_nr_active = weight_by_nr_active
        super().__init__(sobol=sobol)

    def _get_group_weights(self, groups: List[np.ndarray], is_act_all: np.ndarray) -> List[float]:

        if self.weight_by_nr_active:
            nr_active = np.sum(is_act_all, axis=1)
            avg_nr_active = [np.sum(nr_active[group])/len(group) for group in groups]
            return avg_nr_active

        # Uniform sampling
        return [1.]*len(groups)

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
