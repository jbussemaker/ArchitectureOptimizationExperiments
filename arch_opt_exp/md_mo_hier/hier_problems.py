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
import timeit
import numpy as np
from cached_property import cached_property

from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *

from pymoo.indicators.hv import Hypervolume
from pymoo.problems.multi.zdt import ZDT1
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

from typing import *
from arch_opt_exp.md_mo_hier.naive import NaiveProblem, NaiveDesignSpace
from arch_opt_exp.md_mo_hier.correction import CorrectorBase
from arch_opt_exp.experiments.metrics_base import *

__all__ = ['SelectableTunableMetaProblem', 'SelectableTunableBranin', 'SelectableTunableZDT1',
           'TunableBranin', 'TunableZDT1', 'CorrectionTimeMetric']


class TunableBranin(TunableHierarchicalMetaProblem):

    def __init__(self, n_sub=8, diversity_range=.8, imp_ratio=10.):
        factory = lambda n: MDBranin()
        super().__init__(factory, imp_ratio=imp_ratio, n_subproblem=n_sub, diversity_range=diversity_range, n_opts=2)


class TunableZDT1(TunableHierarchicalMetaProblem):

    def __init__(self, n_sub=8):
        factory = lambda n: NoHierarchyWrappedProblem(ZDT1(n_var=n))
        super().__init__(factory, imp_ratio=10., n_subproblem=n_sub, diversity_range=.8, n_opts=2, cont_ratio=1)


class SelectableTunableMetaProblem(TunableHierarchicalMetaProblem):

    def __init__(self, *args, i_sub_opt: int = None, offset: float = 0., **kwargs):
        super().__init__(*args, **kwargs)
        if i_sub_opt is not None:
            self._mod_transform(i_sub_opt, offset)
        self._i_sub_opt = i_sub_opt
        self._offset = offset

        self.corrector_factory = None
        self._corrector = None
        self.last_corr_times = []

    def __getstate__(self):
        state = super().__getstate__()
        state['_corrector'] = None
        return state

    def _mod_transform(self, i_sub_opt, offset: float):
        ref_dirs = RieszEnergyReferenceDirectionFactory(n_dim=self.n_obj, n_points=10)()
        dist_to_origin = np.sqrt(np.sum(ref_dirs**2, axis=1))
        ref_dirs = (ref_dirs.T/dist_to_origin).T
        f_synthetic = self._pf_max-ref_dirs*(self._pf_max-self._pf_min)
        hv = Hypervolume(ideal=self._pf_min, nadir=self._pf_max, ref_point=self._pf_max)

        n_sub = self._transform.shape[0]
        f_hv = np.zeros((n_sub,))
        for i_sub in range(n_sub):
            i_sub_selected = np.ones((f_synthetic.shape[0],), dtype=int)*i_sub
            f_syn_transformed = super()._transform_out(f_synthetic, i_sub_selected)
            f_hv[i_sub] = hv(f_syn_transformed)

        i_hv_max = np.argmax(f_hv)

        i_rotate = np.arange(n_sub)
        i_rotate = i_rotate[np.roll(i_rotate, i_sub_opt-i_hv_max)]
        self._transform = transform = self._transform[i_rotate, :]

        transform[i_sub_opt, :self.n_obj] -= offset/.2

    @cached_property
    def _x_sub_map(self):
        return {tuple(xs): i for i, xs in enumerate(self._x_sub)}

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        super_corr = super()._correct_x

        def _is_valid(xi_):
            x_corr_ = np.array([xi_.copy()])
            is_active_corr_ = np.ones(x_corr_.shape, dtype=bool)
            super_corr(x_corr_, is_active_corr_)
            is_act_ = is_active_corr_[0, :]
            if np.all(x_corr_[0, is_act_] == xi_[is_act_]):
                return is_act_

        if self.corrector_factory is not None:
            if self._corrector is None:
                self._corrector = self.corrector_factory(self.design_space, _is_valid)
            corrector: CorrectorBase = self._corrector

            s = timeit.default_timer()
            corrector.correct_x(x, is_active)
            self.last_corr_times.append(timeit.default_timer()-s)

            x_sub_map = self._x_sub_map
            n_sub = self._x_sub.shape[1]
            self.design_space.impute_x(x, is_active)
            i_sub_selected = np.zeros((x.shape[0],), dtype=int)
            for i, xi in enumerate(x):
                i_sub_selected[i] = x_sub_map[tuple(xi[:n_sub])]
            self._correct_output = {'i_sub_sel': i_sub_selected}
            return

        s = timeit.default_timer()
        super()._correct_x(x, is_active)
        self.last_corr_times.append(timeit.default_timer()-s)

    def __repr__(self):
        return f'{self.__class__.__name__}(imp_ratio={self._imp_ratio}, n_sub={self._n_subproblem}, ' \
               f'div_range={self._diversity_range}, n_opts={self._n_opts}, cont_ratio={self._cont_ratio}, ' \
               f'i_sub_opt={self._i_sub_opt}, offset={self._offset})'


class CorrectionTimeMetric(Metric):
    """Record the time that correction takes"""

    @property
    def name(self):
        return 'corr_time'

    @property
    def value_names(self) -> List[str]:
        return ['mean', 'std']

    def _calculate_values(self, algorithm) -> List[float]:
        problem = algorithm.problem
        corr_times = []
        if isinstance(problem, NaiveProblem):
            design_space = problem.design_space
            assert isinstance(design_space, NaiveDesignSpace)
            corr_times = design_space.last_corr_times
        elif isinstance(problem, SelectableTunableMetaProblem):
            corr_times = problem.last_corr_times

        if len(corr_times) > 0:
            res = [float(np.mean(corr_times)), float(np.std(corr_times))]
            corr_times.clear()
            return res

        return [np.nan]*len(self.value_names)


class SelectableTunableBranin(SelectableTunableMetaProblem):

    def __init__(self, n_sub=128, i_sub_opt=0, diversity_range=.95, imp_ratio=10., n_opts=2, offset=0.):
        factory = lambda n: MDBranin()
        super().__init__(factory, imp_ratio=imp_ratio, n_subproblem=n_sub, diversity_range=diversity_range,
                         n_opts=n_opts, i_sub_opt=i_sub_opt, offset=offset)


class SelectableTunableZDT1(SelectableTunableMetaProblem):

    def __init__(self, n_sub=128, i_sub_opt=0, n_opts=2, offset=0.):
        factory = lambda n: NoHierarchyWrappedProblem(ZDT1(n_var=n))
        super().__init__(factory, imp_ratio=10., n_subproblem=n_sub, diversity_range=.95, n_opts=n_opts, cont_ratio=1,
                         i_sub_opt=i_sub_opt, offset=offset)


if __name__ == '__main__':
    SelectableTunableBranin(imp_ratio=1, diversity_range=0).print_stats()
    SelectableTunableBranin(diversity_range=0).print_stats()
    SelectableTunableBranin().print_stats()
    # SelectableTunableBranin().plot_transformation()
    SelectableTunableZDT1().print_stats()
    # SelectableTunableZDT1().plot_transformation()
