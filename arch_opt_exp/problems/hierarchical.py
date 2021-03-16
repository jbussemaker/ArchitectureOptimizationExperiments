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

import enum
import numpy as np
from typing import *
from arch_opt_exp.problems.discrete import *
from arch_opt_exp.problems.discretization import *

__all__ = ['HierarchicalGoldsteinProblem', 'HierarchicalRosenbrockProblem', 'ZaeffererHierarchicalProblem',
           'ZaeffererProblemMode', 'MOHierarchicalGoldsteinProblem', 'MOHierarchicalRosenbrockProblem']


class HierarchicalGoldsteinProblem(MixedIntBaseProblem):
    """
    Variable-size design space Goldstein function from:
    Pelamatti 2020: "Bayesian Optimization of Variable-Size Design Space Problems", section 5.2 and Appendix B

    Properties:
    - 5 continuous variables
    - 4 integer variables
    - 2 categorical variables
    - Depending on the categorical variables, 8 different sub-problems are defined, ranging from 2 cont + 4 integer to
      5 cont + 2 integer variables
    - 1 objective, 1 constraint

    To validate, use so_run() and compare to Pelamatti 2020, Fig. 7
    """

    _mo = False

    def __init__(self):
        n_var = 11
        xl, xu = np.zeros((n_var,)), np.ones((n_var,))
        xu[:5] = 100.
        xu[5:9] = 2
        xu[9] = 3

        is_int_mask = np.array([0]*5+[1]*4+[0]*2, dtype=bool)
        is_cat_mask = np.array([0]*5+[0]*4+[1]*2, dtype=bool)

        n_obj = 2 if self._mo else 1
        super(HierarchicalGoldsteinProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=n_var, n_obj=n_obj, n_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        f_h_map = self._map_f_h()
        g_map = self._map_g()

        f = np.empty((x.shape[0], self.n_obj))
        g = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            x_i = x[i, :5]
            z_i = [int(z) for z in x[i, 5:9]]
            w_i = [int(w) for w in x[i, 9:]]

            f_idx = int(w_i[0]+w_i[1]*4)
            f[i, 0] = self.h(*f_h_map[f_idx](x_i, z_i))
            if self._mo:
                f2 = self.h(*f_h_map[f_idx](x_i+30, z_i))+(f_idx/7.)*5
                f[i, 1] = f2

            g_idx = int(w_i[0])
            g[i, 0] = self.g(*g_map[g_idx](x_i, z_i))

        out['is_active'], out['X'] = self.is_active(x)
        out['F'] = f
        out['G'] = g

    def _is_active(self, x: np.ndarray) -> np.ndarray:
        x = self.correct_x(x)
        w1 = x[:, 9].astype(np.int)
        w2 = x[:, 10].astype(np.int)

        is_active = np.zeros(x.shape, dtype=bool)
        is_active[:, [0, 1, 7, 8, 9, 10]] = True  # x1, x2, z3, z4, w1, w2

        is_active[:, 2] = np.bitwise_or(w1 == 1, w1 == 3)  # x3
        is_active[:, 3] = w1 >= 2  # x4
        is_active[:, 4] = w2 == 1  # x5

        is_active[:, 5] = np.bitwise_or(w1 == 0, w1 == 2) # z1
        is_active[:, 6] = w1 <= 1 # z2

        return is_active

    @staticmethod
    def h(x1, x2, x3, x4, x5, z3, z4, cos_term: bool) -> float:
        h = MixedIntGoldsteinProblem.h(x1, x2, x3, x4, z3, z4)
        if cos_term:
            h += 5.*np.cos(2.*np.pi*(x5/100.))-2.
        return h

    @staticmethod
    def _map_f_h() -> List[Callable[[np.ndarray, np.ndarray], tuple]]:

        # Appendix B, Table 6-11
        _x3 = [20, 50, 80]
        _x4 = [20, 50, 80]

        def _f1(x, z):
            return x[0], x[1], _x3[z[0]], _x4[z[1]], x[4], z[2], z[3], False

        def _f2(x, z):
            return x[0], x[1], x[2],      _x4[z[1]], x[4], z[2], z[3], False

        def _f3(x, z):
            return x[0], x[1], _x3[z[0]], x[3],      x[4], z[2], z[3], False

        def _f4(x, z):
            return x[0], x[1], x[2],      x[3],      x[4], z[2], z[3], False

        def _f5(x, z):
            return x[0], x[1], _x3[z[0]], _x4[z[1]], x[4], z[2], z[3], True

        def _f6(x, z):
            return x[0], x[1], x[2],      _x4[z[1]], x[4], z[2], z[3], True

        def _f7(x, z):
            return x[0], x[1], _x3[z[0]], x[3],      x[4], z[2], z[3], True

        def _f8(x, z):
            return x[0], x[1], x[2],      x[3],      x[4], z[2], z[3], True

        return [_f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8]

    @staticmethod
    def g(x1, x2, c1, c2):
        return -(x1-50.)**2. - (x2-50.)**2. + (20.+c1*c2)**2.

    @staticmethod
    def _map_g() -> List[Callable[[np.ndarray, np.ndarray], tuple]]:

        # Appendix B, Table 12-15
        _c1 = [3., 2., 1.]
        _c2 = [.5, -1., -2.]

        def _g1(x, z):
            return x[0], x[1], _c1[z[0]], _c2[z[1]]

        def _g2(x, z):
            return x[0], x[1], .5,        _c2[z[1]]

        def _g3(x, z):
            return x[0], x[1], _c1[z[0]], .7

        def _g4(x, z):
            return x[0], x[1], _c1[z[2]], _c2[z[3]]

        return [_g1, _g2, _g3, _g4]

    @classmethod
    def validate_ranges(cls, n_samples=5000, show=True):
        """Compare to Pelamatti 2020, Fig. 6"""
        import matplotlib.pyplot as plt
        from pymoo.model.initialization import Initialization
        from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

        problem = cls()
        init_sampling = Initialization(LatinHypercubeSampling(), repair=problem.get_repair())
        x = init_sampling.do(problem, n_samples).get('X')

        f, g = problem.evaluate(x)
        i_feasible = np.max(g, axis=1) <= 0.

        x_plt, y_plt = [], []
        for i in np.where(i_feasible)[0]:
            w_i = [int(w) for w in x[i, 9:]]
            f_idx = int(w_i[0]+w_i[1]*4)

            x_plt.append(f_idx)
            y_plt.append(f[i])

        plt.figure()
        plt.scatter(x_plt, y_plt, s=1)
        plt.xlabel('Sub-problem'), plt.ylabel('Feasible objective values')

        if show:
            plt.show()


class MOHierarchicalGoldsteinProblem(HierarchicalGoldsteinProblem):
    """
    Multi-objective adaptation of the hierarchical Goldstein problem. The Pareto front consists of a mix of SP6 and SP8,
    however it is difficult to get a consistent result with NSGA2.

    See Pelamatti 2020 Fig. 6 to compare. Colors in plot of run_test match colors of figure.
    """

    _mo = True

    @classmethod
    def run_test(cls):
        from pymoo.optimize import minimize
        from pymoo.algorithms.nsga2 import NSGA2
        from pymoo.visualization.scatter import Scatter

        res = minimize(cls(), NSGA2(pop_size=200), termination=('n_gen', 200))
        w_idx = res.X[:, 9] + res.X[:, 10] * 4
        Scatter().add(res.F, c=w_idx, cmap='tab10', vmin=0, vmax=10, color=None).show()


class HierarchicalRosenbrockProblem(MixedIntBaseProblem):
    """
    Variable-size design space Rosenbrock function from:
    Pelamatti 2020: "Bayesian Optimization of Variable-Size Design Space Problems", section 5.3 and Appendix C

    Properties:
    - 8 continuous variables
    - 3 integer variables
    - 2 categorical variables
    - Depending on the categorical variables, 4 different sub-problems are defined
    - 1 objective, 2 constraints

    To validate, use so_run() and compare to Pelamatti 2020, Fig. 14
    """

    _mo = False

    def __init__(self):
        n_var = 13
        xl, xu = np.zeros((n_var,)), np.ones((n_var,))
        xl[[0, 2, 4, 6]] = -1.
        xu[[0, 2, 4, 6]] = .5
        xu[[1, 3, 5, 7]] = 1.5
        xu[10] = 2

        is_int_mask = np.array([0]*8+[1]*3+[0]*2, dtype=bool)
        is_cat_mask = np.array([0]*8+[0]*3+[1]*2, dtype=bool)

        n_obj = 2 if self._mo else 1
        super(HierarchicalRosenbrockProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=n_var, n_obj=n_obj, n_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        a1 = [7, 7, 10, 10]
        a2 = [9, 6, 9, 6]
        add_z3 = [False, True, False, True]
        x_idx = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 3, 6, 7], [0, 1, 4, 5, 6, 7]]
        x_idx_g2 = [[0, 1, 2, 3], [0, 1, 2, 3, 6, 7]]

        f = np.empty((x.shape[0], self.n_obj))
        g = np.empty((x.shape[0], 2))
        for i in range(x.shape[0]):
            x_i = x[i, :8]
            z_i = [int(z) for z in x[i, 8:11]]

            w_i = [int(w) for w in x[i, 11:]]
            idx = int(w_i[0]*2+w_i[1])

            x_fg = x_i[x_idx[idx]]
            f[i, 0] = f1 = self.f(x_fg, z_i[0], z_i[1], z_i[2], a1[idx], a2[idx], add_z3[idx])
            if self._mo:
                f2 = f1+(x_fg[0]+1)**2*100
                if idx < 2:
                    f2 += 25
                f[i, 1] = f2

            g[i, 0] = self.g1(x_fg)
            g[i, 1] = self.g2(x_i[x_idx_g2[idx]]) if idx < 2 else 0.

        out['is_active'], out['X'] = self.is_active(x)
        out['F'] = f
        out['G'] = g

    def _is_active(self, x: np.ndarray) -> np.ndarray:
        x = self.correct_x(x)
        w1 = x[:, 11].astype(np.int)
        w2 = x[:, 12].astype(np.int)
        idx = w1*2+w2

        is_active = np.zeros(x.shape, dtype=bool)
        is_active[:, [0, 1, 8, 9, 11, 12]] = True  # x1, x2, z1, z2, w1, w2

        is_active[:, 2] = idx <= 2  # x3
        is_active[:, 3] = idx <= 2  # x4
        is_active[:, 4] = w2 == 1  # x5
        is_active[:, 5] = w2 == 1  # x6
        is_active[:, 6] = idx >= 1  # x7
        is_active[:, 7] = idx >= 1  # x8

        is_active[:, 10] = w2 == 1  # z3

        return is_active

    @staticmethod
    def f(x: np.ndarray, z1, z2, z3, a1, a2, add_z3: bool):
        s = 1. if z2 == 0 else -1.
        pre = 1. if z2 == 0 else .7

        xi, xi1 = x[:-1], x[1:]
        sum_term = np.sum(pre*a1*a2*(xi1-xi)**2 + ((a1+s*a2)/10.)*(1-xi)**2)
        f = 100.*z1 + sum_term
        if add_z3:
            f -= 35.*z3
        return f

    @staticmethod
    def g1(x: np.ndarray):
        xi, xi1 = x[:-1], x[1:]
        return np.sum(-(xi-1)**3 + xi1 - 2.6)

    @staticmethod
    def g2(x: np.ndarray):
        xi, xi1 = x[:-1], x[1:]
        return np.sum(-xi - xi1 + .4)

    @classmethod
    def validate_ranges(cls, n_samples=5000, show=True):
        """Compare to Pelamatti 2020, Fig. 13"""
        import matplotlib.pyplot as plt
        from pymoo.model.initialization import Initialization
        from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

        problem = cls()
        init_sampling = Initialization(LatinHypercubeSampling(), repair=problem.get_repair())
        x = init_sampling.do(problem, n_samples).get('X')

        f, g = problem.evaluate(x)
        i_feasible = np.max(g, axis=1) <= 0.

        x_plt, y_plt = [], []
        for i in np.where(i_feasible)[0]:
            w_i = [int(w) for w in x[i, 11:]]
            f_idx = int(w_i[0]*2+w_i[1])

            x_plt.append(f_idx)
            y_plt.append(f[i])

        plt.figure()
        plt.scatter(x_plt, y_plt, s=1)
        plt.xlabel('Sub-problem'), plt.ylabel('Feasible objective values')

        if show:
            plt.show()


class MOHierarchicalRosenbrockProblem(HierarchicalRosenbrockProblem):
    """
    Multi-objective adaptation of the hierarchical Rosenbrock problem.

    See Pelamatti 2020 Fig. 13 to compare. Colors in plot of run_test match colors of figure.
    """

    _mo = True

    @classmethod
    def run_test(cls):
        from pymoo.optimize import minimize
        from pymoo.algorithms.nsga2 import NSGA2
        from pymoo.visualization.scatter import Scatter

        res = minimize(cls(), NSGA2(pop_size=100), termination=('n_gen', 200))
        w_idx = res.X[:, 11]*2 + res.X[:, 12]
        Scatter().add(res.F, c=w_idx, cmap='tab10', vmin=0, vmax=10, color=None).show()


class ZaeffererProblemMode(enum.Enum):
    A_OPT_INACT_IMP_PROF_UNI = 'A'
    B_OPT_INACT_IMP_UNPR_UNI = 'B'
    C_OPT_ACT_IMP_PROF_BI = 'C'
    D_OPT_ACT_IMP_UNPR_BI = 'D'
    E_OPT_DIS_IMP_UNPR_BI = 'E'


class ZaeffererHierarchicalProblem(MixedIntBaseProblem):
    """
    Hierarchical test function from:
    Zaefferer 2018: "A First Analysis of Kernels for Kriging-Based Optimization in Hierarchical Search Spaces",
      section 5
    """

    _mode_map = {
        ZaeffererProblemMode.A_OPT_INACT_IMP_PROF_UNI: (.0, .6, .1),
        ZaeffererProblemMode.B_OPT_INACT_IMP_UNPR_UNI: (.1, .6, .1),
        ZaeffererProblemMode.C_OPT_ACT_IMP_PROF_BI: (.0, .4, .7),
        ZaeffererProblemMode.D_OPT_ACT_IMP_UNPR_BI: (.1, .4, .9),
        ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI: (.1, .4, .7),
    }

    def __init__(self, b=.1, c=.4, d=.7):
        self.b = b
        self.c = c
        self.d = d

        is_int_mask = np.zeros((2,), dtype=bool)
        is_cat_mask = np.zeros((2,), dtype=bool)

        xl, xu = np.zeros((2,)), np.ones((2,))
        super(ZaeffererHierarchicalProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=2, n_obj=1, xl=xl, xu=xu)

    @classmethod
    def from_mode(cls, problem_mode: ZaeffererProblemMode):
        b, c, d = cls._mode_map[problem_mode]
        return cls(b=b, c=c, d=d)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = (x[:, 0] - self.d)**2
        f2 = (x[:, 1] - .5)**2 + self.b
        f2[x[:, 0] <= self.c] = 0.

        out['is_active'], out['X'] = self.is_active(x)
        out['F'] = f1+f2

    def _is_active(self, x: np.ndarray) -> np.ndarray:
        is_active = np.ones(x.shape, dtype=bool)
        is_active[:, 1] = x[:, 0] > self.c  # x2 is active if x1 > c
        return is_active

    def plot(self, show=True):
        import matplotlib.pyplot as plt

        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        zz = self.evaluate(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        plt.figure(), plt.title('b = %.1f, c = %.1f, d = %.1f' % (self.b, self.c, self.d))
        plt.colorbar(plt.contourf(xx, yy, zz, 50, cmap='viridis')).set_label('$f$')
        plt.contour(xx, yy, zz, 5, colors='k')
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')

        if show:
            plt.show()


if __name__ == '__main__':
    # HierarchicalGoldsteinProblem.validate_ranges(show=False)
    # HierarchicalGoldsteinProblem().so_run()
    # MOHierarchicalGoldsteinProblem.run_test()

    # HierarchicalRosenbrockProblem.validate_ranges(show=False)
    # HierarchicalRosenbrockProblem().so_run(n_repeat=8, n_eval_max=2000, pop_size=30)
    # MOHierarchicalRosenbrockProblem.run_test()

    # ZaeffererHierarchicalProblem(b=.1, c=.4, d=.7).plot(show=False)
    # ZaeffererHierarchicalProblem(b=.0, c=.6, d=.1).plot()  # Zaefferer 2018, Fig. 1

    from arch_opt_exp.surrogates.sklearn_models.gp import SKLearnGPSurrogateModel
    sm = SKLearnGPSurrogateModel()
    problem = ZaeffererHierarchicalProblem(b=.1, c=.4, d=.7)
    # problem = ZaeffererHierarchicalProblem(b=.0, c=.6, d=.1)
    # problem.impute = False

    from arch_opt_exp.algorithms.surrogate.surrogate_infill import SurrogateBasedInfill
    SurrogateBasedInfill.plot_model_problem(sm, problem, n_pts=150)
