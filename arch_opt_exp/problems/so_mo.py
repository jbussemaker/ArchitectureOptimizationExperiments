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

Copyright: (c) 2021, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""

import numpy as np
from pymoo.factory import get_problem
from pymoo.model.problem import Problem
from pymoo.problems.single.himmelblau import Himmelblau
from pymoo.problems.single.rosenbrock import Rosenbrock
from arch_opt_exp.problems.discretization import *

__all__ = ['MOHimmelblau', 'MOGoldstein', 'MORosenbrock', 'MORosenbrockInv', 'MIMOHimmelblau', 'MIMOGoldstein',
           'MIMORosenbrock']


class MOHimmelblau(Problem):

    def __init__(self):
        self._problem = p = Himmelblau()
        super(MOHimmelblau, self).__init__(n_var=p.n_var, n_obj=2, xl=p.xl, xu=p.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f_mo = np.zeros((x.shape[0], 2))
        f_mo[:, 0] = self._problem.evaluate(x)[:, 0]
        f_mo[:, 1] = self._problem.evaluate(x[:, ::-1])[:, 0]
        out['F'] = f_mo


class MIMOHimmelblau(MixedIntProblem):

    def __init__(self):
        super(MIMOHimmelblau, self).__init__(MOHimmelblau())


class MOGoldstein(Problem):

    def __init__(self):
        self._problem = p = get_problem('go-goldsteinprice')
        super(MOGoldstein, self).__init__(n_var=p.n_var, n_obj=2, xl=p.xl, xu=p.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f_mo = np.zeros((x.shape[0], 2))
        f_mo[:, 0] = self._problem.evaluate(x)[:, 0]
        f_mo[:, 1] = self._problem.evaluate(x+.15)[:, 0]
        out['F'] = f_mo


class MIMOGoldstein(MixedIntProblem):

    def __init__(self):
        super(MIMOGoldstein, self).__init__(MOGoldstein())


class MORosenbrock(Problem):

    def __init__(self):
        self._problem = p = Rosenbrock(n_var=2)
        super(MORosenbrock, self).__init__(n_var=p.n_var, n_obj=2, xl=p.xl, xu=p.xu)
        self._f2_x_mul = np.array([-1]+[1]*(self.n_var-1))

    def _evaluate(self, x, out, *args, **kwargs):
        f_mo = np.zeros((x.shape[0], self.n_obj))
        f_mo[:, 0] = self._problem.evaluate(x)[:, 0]
        # f_mo[:, 1] = self._problem.evaluate(x*self._f2_x_mul)[:, 0]
        f_mo[:, 1] = self._problem.evaluate(x)[:, 0]+x[:, 0]
        out['F'] = f_mo


class MIMORosenbrock(MixedIntProblem):

    def __init__(self):
        super(MIMORosenbrock, self).__init__(MORosenbrock())


class MORosenbrockInv(Problem):

    def __init__(self):
        self._problem = p = Rosenbrock(n_var=2)
        super(MORosenbrockInv, self).__init__(n_var=p.n_var, n_obj=2, xl=p.xl, xu=p.xu)
        self._f2_x_mul = np.array([-1]+[1]*(self.n_var-1))

    def _evaluate(self, x, out, *args, **kwargs):
        f_mo = np.zeros((x.shape[0], self.n_obj))
        f_mo[:, 0] = self._problem.evaluate(x)[:, 0]
        f_mo[:, 1] = (2500-self._problem.evaluate(x)[:, 0])**2*1e-4
        out['F'] = f_mo

if __name__ == '__main__':
    from pymoo.optimize import minimize
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.visualization.scatter import Scatter

    # prob = MOHimmelblau()
    prob = MOGoldstein()
    # prob = MORosenbrock()
    # prob = MORosenbrockInv()
    res = minimize(prob, NSGA2(pop_size=100), termination=('n_gen', 1000))
    Scatter().add(res.F).show()
    Scatter().add(res.X).show()
