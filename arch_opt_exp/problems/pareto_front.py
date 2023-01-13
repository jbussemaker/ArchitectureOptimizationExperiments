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
import os
import re
import pickle
import hashlib
import numpy as np
from typing import *
import concurrent.futures
from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi.zdt import ZDT1
from pymoo.core.evaluator import Evaluator
from pymoo.visualization.scatter import Scatter
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from arch_opt_exp.algorithms.sampling import *

__all__ = ['CachedParetoFrontMixin']


class CachedParetoFrontMixin(Problem):
    """Mixin to calculate the Pareto front once by simply running the problem a couple of times using NSGA2. Stores the
    results based on the repr of the main class, so make sure that one is set."""

    def reset_pf_cache(self):
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def get_repair(self):
        pass

    def _calc_pareto_front(self, *_, pop_size=200, n_gen=20, n_repeat=10, n_pts_keep=50, **__):
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

        n = 1
        xl, xu = self.bounds()
        for i, var in enumerate(self.vars):
            if isinstance(var, Real):
                n = None
                break
            n *= int(xu[i]-xl[i]+1)
        if n is not None and n < pop_size*n_gen*n_repeat:
            sampling = RepairedExhaustiveSampling(repair=self.get_repair())
            pop = sampling.do(self, n)
            Evaluator().eval(self, pop)

            pop = DefaultDuplicateElimination().do(pop)
            pf = pop.get('F')
            i_non_dom = NonDominatedSorting().do(pf, only_non_dominated_front=True)
            pf = pf[i_non_dom, :]

        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._run_minimize, pop_size, n_gen, i, n_repeat)
                           for i in range(n_repeat)]
                concurrent.futures.wait(futures)

                pf = None
                for i in range(n_repeat):
                    res = futures[i].result()
                    if pf is None:
                        pf = res.F
                    else:
                        pf_merged = np.row_stack([pf, res.F])
                        i_non_dom = NonDominatedSorting().do(pf_merged, only_non_dominated_front=True)
                        pf = pf_merged[i_non_dom, :]

        pf = np.unique(pf, axis=0)
        if n_pts_keep is not None and pf.shape[0] > n_pts_keep:
            for _ in range(pf.shape[0]-n_pts_keep):
                crowding_of_front = calc_crowding_distance(pf)
                i_max_crowding = np.argsort(crowding_of_front)[1:]
                pf = pf[i_max_crowding, :]

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(pf, fp)
        return pf

    def _run_minimize(self, pop_size, n_gen, i, n):
        print('Running PF discovery %d/%d (%d pop, %d gen)' % (i+1, n, pop_size, n_gen))
        return minimize(self, NSGA2(pop_size=pop_size), termination=('n_gen', n_gen))

    def plot_pf(self: Union[Problem, 'CachedParetoFrontMixin'], show_approx_f_range=False, **kwargs):
        scatter = Scatter()
        if show_approx_f_range:
            scatter.add(self.get_approx_f_range(), s=.1, color='white')

            pop = Initialization(LatinHypercubeSampling()).do(self, 100)
            Evaluator().eval(self, pop)
            scatter.add(pop.get('F'), s=5)

        scatter.add(self.pareto_front(**kwargs))
        scatter.show()

    def get_approx_f_range(self, n_sample=1000):
        pop = Initialization(LatinHypercubeSampling()).do(self, n_sample)
        Evaluator().eval(self, pop)
        f = pop.get('F')
        f_max = np.max(f, axis=0)
        f_min = np.min(f, axis=0)
        return np.array([f_min, f_max])

    def _pf_cache_path(self):
        class_str = repr(self)
        if class_str.startswith('<'):
            class_str = self.__class__.__name__
        class_str = re.sub('[^0-9a-z]', '_', class_str.lower().strip())

        if len(class_str) > 20:
            class_str = hashlib.md5(class_str.encode('utf-8')).hexdigest()[:20]

        return os.path.join(os.path.dirname(__file__), 'pf_cache', class_str+'.pkl')


class ZDT1Calc(CachedParetoFrontMixin, ZDT1):
    pass


if __name__ == '__main__':
    ZDT1Calc().plot_pf()
