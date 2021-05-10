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
from pymoo.model.problem import Problem
from arch_opt_exp.surrogates.model import *
from pymoo.model.evaluator import Evaluator
from matplotlib.colors import DivergingNorm
from pymoo.model.population import Population
from arch_opt_exp.problems.discretization import *
from pymoo.model.initialization import Initialization
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

from arch_opt_exp.algorithms.surrogate.mo.enhanced_poi import *
from arch_opt_exp.algorithms.surrogate.surrogate_infill import *

from arch_opt_exp.surrogates.smt_models.smt_krg import *
from arch_opt_exp.surrogates.sklearn_models.gp import *
from arch_opt_exp.surrogates.sklearn_models.mixed_int_dist import *
from arch_opt_exp.surrogates.sklearn_models.hierarchical_dist import *
from arch_opt_exp.surrogates.sklearn_models.hierarchical_decomp_kernel import *


def investigate_sm_infill(problem: Problem):
    smt_kwargs = {'theta0': 1.}
    sklearn_kwargs = {'alpha': 1e-6, 'int_as_discrete': True}

    def _investigate(sm_, name_):
        # investigate(problem, sm_, ExpectedImprovementInfill(), sm_name=name_, show=False)
        investigate(problem, sm_, MinimumPOIInfill(), sm_name=name_, show=False)

    _investigate(SMTKrigingSurrogateModel(**smt_kwargs), 'SMT Kriging')
    _investigate(SKLearnGPSurrogateModel(kernel=HammingDistance().kernel(), **sklearn_kwargs), 'SKL GP MI Ham')
    _investigate(SKLearnGPSurrogateModel(
        kernel=IndefiniteConditionalDistance().kernel(), **sklearn_kwargs), 'SKL GP MI+H Ico')
    _investigate(SKLearnGPSurrogateModel(
        kernel=SPWDecompositionKernel(CompoundSymmetryKernel().kernel()), **sklearn_kwargs), 'SKL GP MI+H SPW+CS')

    plt.show()


def investigate(problem: Problem, sm: SurrogateModel, infill: SurrogateInfill, sm_name=None, show=True):
    n_sm, n_infill = 100, 500
    pop = get_pop(problem, n_sm)
    sbi = SurrogateBasedInfill(sm, infill, pop_size=n_sm)
    infill.initialize(problem, sbi.surrogate_model)
    pf_real = pf = problem.pareto_front()

    Evaluator().eval(problem, pop)
    sbi.problem = problem
    sbi.total_pop = pop
    sbi._build_model(pop)

    infill_problem = sbi._get_infill_problem()
    pop_infill = get_pop(infill_problem, n_infill)
    x = pop_infill.get('X')

    f, _ = infill.predict(x)
    f_var, _ = infill.predict_variance(x)
    f_infill, g_infill = infill.evaluate(x)
    n_f = f.shape[1]

    pop_real = Population.new(X=sbi._denormalize(x))
    Evaluator().eval(problem, pop_real)
    f_real = pop_real.get('F')
    g_real = pop_real.get('G')
    n_g = 0 if g_real[0] is None else g_real.shape[1]

    f_init, _, _ = sbi._normalize_y(pop.get('F'), y_min=sbi.y_train_min[:n_f], y_max=sbi.y_train_max[:n_f])
    if pf is not None:
        pf, _, _ = sbi._normalize_y(pf, y_min=sbi.y_train_min[:n_f], y_max=sbi.y_train_max[:n_f])

    if n_f == 1:
        _, ax = plt.subplots(2+n_g, 2)
        ax[0, 0].title.set_text(
            '%s - %s - %s' % (problem.name(), sm_name or sm.__class__.__name__, infill.__class__.__name__))
        ax[0, 0].scatter(f_init[:, 0]*0, f_init[:, 0], 8, 'r', label='Samples')
        ax[0, 0].scatter(f[:, 0]*0+1, f[:, 0], 8, 'b', label='Infill')
        if pf is not None:
            ax[0, 0].scatter([0, 1], [pf[0, 0], pf[0, 0]], 'g', label='Optimum')
        ax[0, 0].legend()

        ax[0, 1].scatter(f[:, 0], np.sqrt(f_var[:, 0]), 8, c='k')
        ax[0, 1].set_xlabel('f'), ax[0, 1].set_ylabel('Std dev')

        ax[1, 0].scatter(f[:, 0], f_infill[:, 0], 8, c='k')
        ax[1, 0].set_xlabel('f'), ax[1, 0].set_ylabel('Infill f')

        ax[1, 1].scatter(f_real[:, 0], f_infill[:, 0], 8, c='k')
        ax[1, 1].set_xlabel('f real'), ax[1, 1].set_ylabel('Infill f')

        for i in range(n_g):
            c = ax[2+i, 0].scatter(f[:, 0], f_infill[:, 0], 8, c=g_infill[:, i], cmap='RdYlBu_r', norm=DivergingNorm(0))
            plt.colorbar(c, ax=ax[2+i, 0]).set_label('Infill $g_%d$' % i)
            ax[2+i, 0].set_xlabel('f'), ax[2+i, 0].set_ylabel('Infill f')

            c = ax[2+i, 1].scatter(
                f_real[:, 0], f_infill[:, 0], 8, c=g_real[:, i], cmap='RdYlBu_r', norm=DivergingNorm(0))
            plt.colorbar(c, ax=ax[2+i, 1]).set_label('Real $g_%d$' % i)
            ax[2+i, 1].set_xlabel('f real'), ax[2+i, 1].set_ylabel('Infill f')

    else:
        n_f_infill = f_infill.shape[1]
        _, ax = plt.subplots((2 if n_f_infill < 2 else 3)+n_g, 3)
        ax[0, 1].title.set_text(
            '%s - %s - %s' % (problem.name(), sm_name or sm.__class__.__name__, infill.__class__.__name__))
        ax[0, 0].scatter(f_init[:, 0], f_init[:, 1], 8, 'r', label='Samples')
        ax[0, 0].scatter(f[:, 0], f[:, 1], 8, 'b', label='Infill')
        if pf is not None:
            ax[0, 0].scatter(pf[:, 0], pf[:, 1], 8, 'g', label='PF')
        ax[0, 0].legend()

        for i in range(n_f):
            c = ax[0, 1+i].scatter(f[:, 0], f[:, 1], 8, c=np.sqrt(f_var[:, i]), cmap='viridis')
            plt.colorbar(c, ax=ax[0, 1+i]).set_label('Std dev $f_%d$' % i)

        for i in range(n_f_infill):
            c = ax[1, i].scatter(f[:, 0], f[:, 1], 8, c=f_infill[:, i], cmap='viridis')
            plt.colorbar(c, ax=ax[1, i]).set_label('Infill $f_%d$' % i)
            if pf is not None:
                ax[1, i].scatter(pf[:, 0], pf[:, 1], 8, 'g', label='PF')

        ii = [(1, 2)] if n_f_infill == 1 else [(2, 0), (2, 1)]
        for i in range(n_f_infill):
            c = ax[ii[i]].scatter(f_real[:, 0], f_real[:, 1], 8, c=f_infill[:, i], cmap='viridis')
            plt.colorbar(c, ax=ax[ii[i]]).set_label('Infill $f_%d$ (f real)' % i)
            if pf is not None:
                ax[ii[i]].scatter(pf_real[:, 0], pf_real[:, 1], 8, 'g', label='PF')

        i_rg = 2 if n_f_infill < 2 else 3
        for i in range(n_g):
            c = ax[i_rg+i, 0].scatter(f[:, 0], f[:, 1], 8, c=g_infill[:, i], cmap='RdYlBu_r', norm=DivergingNorm(0))
            plt.colorbar(c, ax=ax[i_rg+i, 0]).set_label('Infill $g_%d$' % i)

            c = ax[i_rg+i, 1].scatter(f[:, 0], f[:, 1], 8, c=g_real[:, i], cmap='RdYlBu_r', norm=DivergingNorm(0))
            plt.colorbar(c, ax=ax[i_rg+i, 1]).set_label('Real $g_%d$' % i)

            ax[i_rg+i, 2].scatter(g_infill[:, i], g_real[:, i], 8, c='k')
            ax[i_rg+i, 2].set_xlabel('Infill $g_%d$' % i), ax[i_rg+i, 2].set_ylabel('Real $g_%d$' % i)

    if show:
        plt.show()


def get_pop(problem: Problem, n=100):
    init_sampling = Initialization(LatinHypercubeSampling(), repair=MixedIntProblemHelper.get_repair(problem))
    return init_sampling.do(problem, n)


def investigate_metrics(problem: Problem, pop, title, show=False):
    from pymoo.algorithms.nsga2 import NSGA2
    from arch_opt_exp.metrics.performance import DeltaHVMetric, IGDMetric

    pf = problem.pareto_front()
    hv_metric = DeltaHVMetric(pf)
    igd_metric = IGDMetric(pf)

    algo = NSGA2()
    algo.pop = pop
    f_pop = pop.get('F')
    pf_pop = hv_metric.get_pareto_front(f_pop)
    algo.opt = Population.new(F=pf_pop)

    hv_metric.calculate_step(algo)
    _, ax = plt.subplots(2, 2)
    ax[0, 0].title.set_text(title+'\n$\\Delta$HV = %.4f' % hv_metric.values['delta_hv'][-1])
    ax[0, 0].scatter(f_pop[:, 0], f_pop[:, 1], 5, label='Pop F')
    ax[0, 0].scatter(pf_pop[:, 0], pf_pop[:, 1], 8, label='Pop PF')
    ax[0, 0].scatter(pf[:, 0], pf[:, 1], 8, label='Real PF')
    ax[0, 0].scatter([hv_metric._hv.ideal_point[0]], [hv_metric._hv.ideal_point[1]], 8, label='Ideal')
    ax[0, 0].scatter([hv_metric._hv.nadir_point[0]], [hv_metric._hv.nadir_point[1]], 8, label='Nadir')
    ax[0, 0].legend(), ax[0, 0].set_xlabel('$f_1$'), ax[0, 0].set_ylabel('$f_2$')

    igd_metric.calculate_step(algo)
    ax[0, 1].title.set_text('IGD = %.4f' % igd_metric.values['indicator'][-1])
    ax[0, 1].scatter(pf_pop[:, 0], pf_pop[:, 1], 8, label='Pop PF')
    ax[0, 1].scatter(pf[:, 0], pf[:, 1], 8, label='Real PF')
    ax[0, 1].legend(), ax[0, 1].set_xlabel('$f_1$'), ax[0, 1].set_ylabel('$f_2$')

    if show:
        plt.show()


if __name__ == '__main__':
    from pymoo.factory import get_problem
    from arch_opt_exp.problems.so_mo import *
    from arch_opt_exp.problems.discrete import *
    from arch_opt_exp.problems.hierarchical import *

    investigate_sm_infill(MOHierarchicalTestProblem())  # MI+H, MO (++), 2g
    # investigate_sm_infill(MOHierarchicalRosenbrockProblem())  # MI+H, MO, 2g
    # investigate_sm_infill(MOHierarchicalGoldsteinProblem())  # MI+H, MO, 1g
    # investigate_sm_infill(MIMOHimmelblau())  # MI, MO
    # investigate_sm_infill(MOGoldstein())  # C, MO

    # investigate_sm_infill(HierarchicalRosenbrockProblem())  # MI+H (C+), 2g
    # investigate_sm_infill(AugmentedMixedIntBraninProblem())  # MI (C+)
    # investigate_sm_infill(MixedIntGoldsteinProblem())  # MI
    # investigate_sm_infill(get_problem('go-goldsteinprice'))  # C
