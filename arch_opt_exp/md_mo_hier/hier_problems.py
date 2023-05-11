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
import numpy as np

from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.problems_base import *

from pymoo.indicators.hv import Hypervolume
from pymoo.problems.multi.zdt import ZDT1
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

__all__ = ['SelectableTunableMetaProblem', 'SelectableTunableBranin', 'SelectableTunableZDT1']


class SelectableTunableMetaProblem(TunableHierarchicalMetaProblem):

    def __init__(self, *args, i_sub_opt: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mod_transform(i_sub_opt or 0)
        self._i_sub_opt = i_sub_opt

    def _mod_transform(self, i_sub_opt):
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

        swapped_transform = self._transform.copy()
        swapped_transform[i_sub_opt, :] = self._transform[i_hv_max, :]
        swapped_transform[i_hv_max, :] = self._transform[i_sub_opt, :]
        self._transform = swapped_transform

    def __repr__(self):
        return f'{self.__class__.__name__}(imp_ratio={self._imp_ratio}, n_sub={self._n_subproblem}, ' \
               f'div_range={self._diversity_range}, n_opts={self._n_opts}, cont_ratio={self._cont_ratio}, ' \
               f'i_sub_opt={self._i_sub_opt})'


class SelectableTunableBranin(SelectableTunableMetaProblem):

    def __init__(self, n_sub=128, i_sub_opt=0, diversity_range=.95, imp_ratio=10.):
        factory = lambda n: MDBranin()
        super().__init__(factory, imp_ratio=imp_ratio, n_subproblem=n_sub, diversity_range=diversity_range, n_opts=2,
                         i_sub_opt=i_sub_opt)


class SelectableTunableZDT1(SelectableTunableMetaProblem):

    def __init__(self, n_sub=128, i_sub_opt=0):
        factory = lambda n: NoHierarchyWrappedProblem(ZDT1(n_var=n))
        super().__init__(factory, imp_ratio=10., n_subproblem=n_sub, diversity_range=.95, n_opts=2, cont_ratio=1,
                         i_sub_opt=i_sub_opt)


if __name__ == '__main__':
    SelectableTunableBranin(imp_ratio=1, diversity_range=0).print_stats()
    SelectableTunableBranin(diversity_range=0).print_stats()
    SelectableTunableBranin().print_stats()
    # SelectableTunableBranin().plot_transformation()
    SelectableTunableZDT1().print_stats()
    # SelectableTunableZDT1().plot_transformation()
