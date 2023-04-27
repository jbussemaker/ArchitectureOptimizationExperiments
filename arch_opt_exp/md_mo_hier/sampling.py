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
from sb_arch_opt.problem import ArchOptRepair
from sb_arch_opt.sampling import *

__all__ = ['HierarchicalSamplingTestBase', 'NoGroupingHierarchicalSampling', 'NrActiveHierarchicalSampling',
           'ActiveVarHierarchicalSampling', 'RepairedSampler',
           'HierarchicalActSepRandomSampling', 'HierarchicalSobolSampling', 'HierarchicalDirectRandomSampling']


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


class HierarchicalSamplingTestBase(HierarchicalRandomSampling):
    """Base class for testing random sampling: groups and weights discrete vectors"""

    def __init__(self, weight_by_nr_active=False, sobol=True):
        self.weight_by_nr_active = weight_by_nr_active
        super().__init__(sobol=sobol)

    def _sample_discrete_x(self, n_samples: int, is_cont_mask, x_all: np.ndarray, is_act_all: np.ndarray, sobol=False):
        if x_all.shape[0] == 0:
            raise ValueError('Set of discrete vectors cannot be empty!')

        def _choice(n_choose, n_from, replace=True):
            return self._choice(n_choose, n_from, replace=replace, sobol=sobol)

        # Separate design vectors into groups
        groups = self.group_design_vectors(x_all, is_act_all, is_cont_mask)

        # Apply weights to the different groups
        weights = np.array(self._get_group_weights(groups, is_act_all))

        # Uniformly choose from which group to sample
        if len(groups) == 1:
            selected_groups = np.zeros((n_samples,), dtype=int)
        else:
            unit_weights = weights/np.sum(weights)
            selected_groups = np.zeros((n_samples,), dtype=int)
            selected_pos = np.linspace(0, 1, n_samples)
            for cum_weight in np.cumsum(unit_weights)[:-1]:
                selected_groups[selected_pos > cum_weight] += 1

        x = []
        is_active = []
        has_x_cont = np.any(is_cont_mask)
        i_x_sampled = np.ones((x_all.shape[0],), dtype=bool)
        for i_grp in range(len(groups)):
            i_x_tgt = np.where(selected_groups == i_grp)[0]
            if len(i_x_tgt) == 0:
                continue

            # Uniformly-randomly select values within group
            i_x_group = groups[i_grp]
            if len(i_x_tgt) < i_x_group.shape[0]:
                n_sel = len(i_x_tgt)
                n_avail = i_x_group.shape[0]
                n_sel_unit = (np.arange(n_sel)+np.random.random(n_sel)*.9999)/n_sel
                i_from_group = np.round(n_sel_unit*n_avail - .5).astype(int)

            # If there are more samples requested than points available, only repeat points if there are continuous vars
            elif has_x_cont:
                i_x_add = _choice(len(i_x_tgt)-i_x_group.shape[0], i_x_group.shape[0])
                i_from_group = np.sort(np.concatenate([np.arange(i_x_group.shape[0]), i_x_add]))
            else:
                i_from_group = np.arange(i_x_group.shape[0])

            x_all_choose = i_x_group[i_from_group]
            x.append(x_all[x_all_choose, :])
            is_active.append(is_act_all[x_all_choose, :])
            i_x_sampled[x_all_choose] = True

        x = np.row_stack(x)
        is_active = np.row_stack(is_active)

        # Uniformly add discrete vectors if there are not enough (can happen if some groups are very small and there
        # are no continuous dimensions)
        if x.shape[0] < n_samples:
            n_add = n_samples-x.shape[0]
            x_available = x_all[~i_x_sampled, :]
            is_act_available = is_act_all[~i_x_sampled, :]

            if n_add < x_available.shape[0]:
                i_from_group = _choice(n_add, x_available.shape[0], replace=False)
            else:
                i_from_group = np.arange(x_available.shape[0])

            x = np.row_stack([x, x_available[i_from_group, :]])
            is_active = np.row_stack([is_active, is_act_available[i_from_group, :]])

        return x, is_active

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


class HierarchicalActSepRandomSampling(HierarchicalRandomSampling):

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


class HierarchicalSobolSampling(HierarchicalRandomSampling):

    def __init__(self):
        super().__init__(sobol=True)


class HierarchicalDirectRandomSampling(HierarchicalRandomSampling):
    """Directly sample from all available discrete design vectors"""

    def __init__(self):
        super().__init__(sobol=False)
