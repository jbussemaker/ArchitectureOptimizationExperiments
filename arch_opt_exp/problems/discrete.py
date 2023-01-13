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
import matplotlib.pyplot as plt
from pymoo.core.population import Population
from arch_opt_exp.problems.discretization import *

__all__ = ['MixedIntBraninProblem', 'AugmentedMixedIntBraninProblem', 'MixedIntGoldsteinProblem',
           'MunozZunigaToyProblem', 'Halstrup04']


class MixedIntBraninProblem(MixedIntBaseProblem):
    """
    Mixed-integer version of the Branin problem that introduces two discrete variables that transform the original
    Branin space in different ways.

    Implementation based on:
    Pelamatti 2020: "Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and
    Discrete Variables", section 4.1
    """

    def __init__(self):
        xl, xu = np.zeros((4,)), np.ones((4,))
        is_int_mask = np.array([False]*4, dtype=bool)
        is_cat_mask = np.array([False, False, True, True], dtype=bool)
        super(MixedIntBraninProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=4, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        f = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            h = self._h(x[i, 0], x[i, 1])

            z1, z2 = x[i, 2], x[i, 3]
            if z1 == 0:
                f[i, 0] = h if z2 == 0 else (.4*h + 1.1)
            else:
                f[i, 0] = (-.75*h + 5.2) if z2 == 0 else (-.5*h - 2.1)

        out['F'] = f

    @staticmethod
    def _h(x1, x2):
        t1 = (15*x2 - (5/(4*np.pi**2))*(15*x1-5)**2 + (5/np.pi)*(15*x1-5) - 6)**2
        t2 = 10*(1-1/(8*np.pi))*np.cos(15*x1-5) + 10
        return ((t1+t2)-54.8104)/51.9496

    def plot(self, z1=0, z2=0, show=True):
        xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        x = np.column_stack([xx.ravel(), yy.ravel(), np.ones((xx.size,))*z1, np.ones((xx.size,))*z2])

        out = Population.new(X=x)
        out = self.evaluate(x, out)
        ff = out.reshape(xx.shape)

        plt.figure(), plt.title('Discrete Branin: $z_1$ = %d, $z_2$ = %d' % (z1, z2))
        plt.colorbar(plt.contourf(xx, yy, ff, 50, cmap='viridis'))
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')
        plt.xlim([0, 1]), plt.ylim([0, 1])

        if show:
            plt.show()


class AugmentedMixedIntBraninProblem(MixedIntBraninProblem):
    """
    Mixed-integer version of the Branin function with more continuous input dimensions.

    Implementation based on:
    Pelamatti 2020: "Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and
    Discrete Variables", section 4.2
    """

    def __init__(self):
        xl, xu = np.zeros((12,)), np.ones((12,))
        is_int_mask = np.array([False]*12, dtype=bool)
        is_cat_mask = np.array([False]*10+[True, True], dtype=bool)
        super(MixedIntBraninProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=12, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        f = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            h = sum([self._h(x[i, j], x[i, j+1]) for j in range(0, 10, 2)])

            z1, z2 = x[i, 2], x[i, 3]
            if z1 == 0:
                f[i, 0] = h if z2 == 0 else (.4*h + 1.1)
            else:
                f[i, 0] = (-.75*h + 5.2) if z2 == 0 else (-.5*h - 2.1)

        out['F'] = f


class MixedIntGoldsteinProblem(MixedIntBaseProblem):
    """
    Mixed-integer version of the Branin problem that introduces two discrete variables that transform the original
    Branin space in different ways.

    Implementation based on:
    Pelamatti 2020: "Overview and Comparison of Gaussian Process-Based Surrogate Models for Mixed Continuous and
    Discrete Variables", section 4.1
    """

    def __init__(self):
        xl, xu = np.zeros((4,)), np.array([100., 100., 2, 2])
        is_int_mask = np.array([False, False, True, True], dtype=bool)
        is_cat_mask = np.array([False]*4, dtype=bool)
        super(MixedIntGoldsteinProblem, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=4, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        _x3 = [20, 50, 80]
        _x4 = [20, 50, 80]

        f = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            x3, x4 = _x3[int(x[i, 2])], _x4[int(x[i, 3])]
            f[i, 0] = self.h(x[i, 0], x[i, 1], x3, x4)

        out['F'] = f

    @staticmethod
    def h(x1, x2, x3, x4, z3=4, z4=3):
        return sum([
            53.3108,
            .184901 * x1,
            -5.02914 * x1**3 * 1e-6,
            7.72522 * x1**z3 * 1e-8,
            0.0870775 * x2,
            -0.106959 * x3,
            7.98772 * x3**z4 * 1e-6,
            0.00242482 * x4,
            1.32851 * x4**3 * 1e-6,
            -0.00146393 * x1 * x2,
            -0.00301588 * x1 * x3,
            -0.00272291 * x1 * x4,
            0.0017004 * x2 * x3,
            0.0038428 * x2 * x4,
            -0.000198969 * x3 * x4,
            1.86025 * x1 * x2 * x3 * 1e-5,
            -1.88719 * x1 * x2 * x4 * 1e-6,
            2.50923 * x1 * x3 * x4 * 1e-5,
            -5.62199 * x2 * x3 * x4 * 1e-5,
        ])

    def plot(self, z1=0, z2=0, show=True):
        xx, yy = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
        x = np.column_stack([xx.ravel(), yy.ravel(), np.ones((xx.size,))*z1, np.ones((xx.size,))*z2])

        out = Population.new(X=x)
        out = self.evaluate(x, out)
        ff = out.reshape(xx.shape)

        plt.figure(), plt.title('Discrete Goldstein: $z_1$ = %d, $z_2$ = %d' % (z1, z2))
        plt.colorbar(plt.contourf(xx, yy, ff, 50, cmap='viridis'))
        plt.xlabel('$x_1$'), plt.ylabel('$x_2$')
        plt.xlim([0, 100]), plt.ylim([0, 100])

        if show:
            plt.show()


class MunozZunigaToyProblem(MixedIntBaseProblem):
    """
    Toy problem from:
    Munoz Zuniga 2020: "Global optimization for mixed categorical-continuous variables based on Gaussian process models
    with a randomized categorical space exploration step", 10.1080/03155986.2020.1730677

    Minimum: -2.329605
    """

    def __init__(self):
        xl, xu = np.zeros((2,)), np.array([1., 9])
        is_int_mask = np.array([False, True], dtype=bool)
        is_cat_mask = np.zeros((2,), dtype=bool)
        super(MunozZunigaToyProblem, self).__init__(is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=2, n_obj=1,
                                                    xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        f = [
            lambda x_: np.cos(3.6 * np.pi * (x_ - 2)) + x_ - 1,
            lambda x_: 2 * np.cos(1.1 * np.pi * np.exp(x_)) - .5 * x_ + 2,
            lambda x_: np.cos(2 * np.pi * x_) + .5 * x_,
            lambda x_: x_ * (np.cos(3.4 * np.pi * (x_ - 1)) - .5 * (x_ - 1)),
            lambda x_: -.5 * x_ ** 2,
            lambda x_: 2 * np.cos(.25 * np.pi * np.exp(-x_ ** 4)) ** 2 - .5 * x_ + 1,
            lambda x_: x_ * np.cos(3.4 * np.pi * x_) - .5 * x_ + 1,
            lambda x_: x_ * (-np.cos(7 * .5 * np.pi * x_) - .5 * x_) + 2,
            lambda x_: -.5 * x_ ** 5 + 1,
            lambda x_: -np.cos(5 * .5 * np.pi * x_) ** 2 * np.sqrt(x_) - .5 * np.log(x_ + .5) - 1.3,
        ]

        y = np.empty((x.shape[0], 1))
        for i in range(10):
            i_x = x[:, 1] == i
            if len(np.where(i_x)[0] > 0):
                y[i_x, 0] = f[i](x[i_x, 0])

        out['F'] = y

    def plot(self, show=True):
        x = np.linspace(0, 1, 100)
        z = np.array(list(range(10)))
        xx, zz = np.meshgrid(x, z)
        xx = xx.ravel()
        zz = zz.ravel()

        out = Population.new(X=np.column_stack([xx, zz]))
        out = self.evaluate(out.get('X'), out)
        f = out[:]

        plt.figure(), plt.title('Munoz-Zuniga Toy Problem')
        for i in z:
            i_x = zz == i
            plt.plot(x, f[i_x], linewidth=1, label='$z = %d$' % (i+1,))
        plt.xlim([0, 1]), plt.xlabel('$x$'), plt.ylabel('$f$'), plt.legend()

        if show:
            plt.show()


class Halstrup04(MixedIntBaseProblem):
    """
    Fourth mixed-integer test problem from:
    Halstrup 2016, "Black-Box Optimization of Mixed Discrete-Continuous Optimization Problems"

    Minimum: 1.7025 (https://mixed-optimization-benchmark.github.io/cases/hal04/)
    Original report contains an error
    """

    f_aux_mod = [
        [  # a
            [(1., .0), (1., .2)],  # d
            [(.9, .0), (1., .25)],  # e
        ],
        [  # b
            [(1., .5), (.8, .0)],  # d
            [(.5, .0), (1., .8)],  # e
        ],
        [  # c
            [(1., .9), (.5, .0)],  # d
            [(1., 1.), (1., 1.25)],  # e
        ],
    ]

    def __init__(self):
        xl, xu = np.zeros((8,)), np.array([1., 1., 1., 1., 1., 2, 1, 1])
        is_int_mask = np.zeros((8,), dtype=bool)
        is_cat_mask = np.array([False]*5+[True]*3, dtype=bool)
        super(Halstrup04, self).__init__(
            is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, n_var=8, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x = self.correct_x(x)

        x_ = x[:, :5]
        z_ = x[:, 5:].astype(np.int)

        d = 8
        x2_term = 2**(np.arange(5)/(d-1))
        f_aux = np.sum((5*x_+(1-x_))**2*x2_term, axis=1)-2.75

        f = np.empty((x.shape[0], 1))
        for i in range(z_.shape[0]):
            f_aux_mul, f_aux_add = self.f_aux_mod[z_[i, 0]][z_[i, 1]][z_[i, 2]]
            f[i, 0] = f_aux[i]*f_aux_mul + f_aux_add

        out['F'] = f

    # def validate(self):
    #     x = np.array([[.5, .5, .5, .5, .5, 0, 0, 0]])
    #     out = Population.new(X=x)
    #     out = self.evaluate(out.get('X'), out)
    #     print(out[0])
    #     assert out[0] == -2.75


if __name__ == '__main__':
    # MixedIntBraninProblem().plot(z1=0, z2=0)
    # MixedIntGoldsteinProblem().plot(z1=0, z2=0)
    # MunozZunigaToyProblem().plot()
    Halstrup04().so_run()
