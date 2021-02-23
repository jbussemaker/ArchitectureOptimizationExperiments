"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2020, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""

import numpy as np
import matplotlib.pyplot as plt
from arch_opt_exp.algorithms.surrogate.mo.mo_modulate import *
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *

__all__ = ['ExpectedMaximinFitnessInfill', 'ModExpectedMaximinFitnessInfill']


class ExpectedMaximinFitnessInfill(ProbabilityOfFeasibilityInfill):
    """
    The Expected Maximin Fitness Infill (EMFI) criterion. The EMF function quantities the improvement over an existing
    Pareto front for a given predicted mean and variance. The maximin function represents the maximum over all Pareto
    points of the minimum of the distances to each objective.

    Implementation based on:
    Svenson, J., "Multiobjective Optimization of Expensive-to-Evaluate Deterministic Computer Simulation Models", 2016,
        10.1016/j.csda.2015.08.011
    """

    def __init__(self, n_monte_carlo: int = 1000, **kwargs):
        super(ExpectedMaximinFitnessInfill, self).__init__(**kwargs)

        self.n_mc = n_monte_carlo
        self.f_pareto = None

    def set_training_values(self, x_train: np.ndarray, y_train: np.ndarray):
        super(ExpectedMaximinFitnessInfill, self).set_training_values(x_train, y_train)

        self.f_pareto = self.get_pareto_front(y_train[:, :self.problem.n_obj])

    def get_n_infill_objectives(self) -> int:
        return 1

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        emf = np.empty((f_predict.shape[0], 1))
        for i in range(emf.shape[0]):
            emf[i, 0] = self._emf_monte_carlo(f_predict[i, :], f_var_predict[i, :], self.f_pareto, n=self.n_mc)
        return 1.-emf

    @classmethod
    def _emf_monte_carlo(cls, f_mean, var, f_pareto, n=10000) -> float:  # Eq. 34
        f_rand = np.random.normal(loc=f_mean, scale=np.sqrt(var), size=(n, len(f_mean)))
        return np.mean(cls._emf(f_rand, f_pareto))

    @staticmethod
    def _emf(f: np.ndarray, f_pareto: np.ndarray) -> float:  # Eq. 21
        # f(n_pt, n_obj), f_pareto(n_pareto, n_obj)

        # Pareto point-wise difference: (n_pareto, n_obj, n_pt)
        f_diff = np.array([f.T]) - np.atleast_3d(f_pareto)

        # Minima (over objectives) of the improvements for each point: (n_pareto, n_pt)
        f_diff_min = np.min(f_diff, axis=1)

        # Maximum of improvements (over points): (n_pt,)
        f_diff_min_max = -np.max(f_diff_min, axis=0)

        # Truncate
        f_diff_min_max[f_diff_min_max < 0.] = 0.
        return f_diff_min_max

    @classmethod
    def plot_emfi(cls, n_pareto=3, n=10000, var=None, n_rand=None, contour=False, show=True):  # Fig. 1
        f_pareto = np.zeros((n_pareto, 2))
        f_pareto[:, 0] = (1.-np.cos(.5*np.pi*np.linspace(0, 1, n_pareto+2)[1:-1]))**.8
        f_pareto[:, 1] = (1.-np.cos(.5*np.pi*(1-np.linspace(0, 1, n_pareto+2)[1:-1])))**.8

        if var is not None and np.isscalar(var):
            var = [var, var]
        kw_rand = {}
        if n_rand is not None:
            kw_rand['n'] = n_rand

        dom_res_set = []
        res_sets = {i_obj: {i_pt: [] for i_pt in range(n_pareto)} for i_obj in range(2)}

        xx1 = xx2 = None
        if contour:
            n_dim = int(np.sqrt(n))
            n = n_dim**2
            xx1, xx2 = np.meshgrid(np.linspace(0, 1, n_dim), np.linspace(0, 1, n_dim))
            f = np.column_stack([xx1.ravel(), xx2.ravel()])
        else:
            f = np.random.random((n, 2))

        mm_fit = np.zeros((n,))
        for i in range(f.shape[0]):
            f_pt = f[i, :]
            if var is None:
                mm = cls._emf(np.array([f_pt]), f_pareto)
            else:
                mm = cls._emf_monte_carlo(f_pt, var, f_pareto, **kw_rand)
            mm_fit[i] = mm

            if var is None:
                if mm == 0.:
                    dom_res_set.append(f_pt)
                else:
                    i_pt, i_obj = np.argwhere(np.abs((f_pareto-f_pt) - mm) < 1e-10)[0]
                    res_sets[i_obj][i_pt].append(f_pt)

        plt.figure(), plt.title('Minimax')
        if contour:
            yy_mm_fit = mm_fit.reshape(xx1.shape)
            c = plt.contourf(xx1, xx2, yy_mm_fit, 50)
        else:
            c = plt.scatter(f[:, 0], f[:, 1], s=5, c=mm_fit)

        plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=10, c='k')
        plt.ylim(0, 1), plt.xlim(0, 1)
        plt.colorbar(c)

        if var is None:
            plt.figure()
            plt.title('Minimax regions')
            if len(dom_res_set) > 0:
                dom_res_set = np.array(dom_res_set)
                plt.scatter(dom_res_set[:, 0], dom_res_set[:, 1], s=5, label='$R_D$')
            for i_obj in res_sets:
                for i_pt in res_sets[i_obj]:
                    pts = res_sets[i_obj][i_pt]
                    if len(pts) > 0:
                        pts = np.array(pts)
                        plt.scatter(pts[:, 0], pts[:, 1], s=5, label='$R_{%d,%d}$' % (i_obj+1, i_pt+1))
            plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=10, c='k')
            plt.legend(), plt.ylim(0, 1), plt.xlim(0, 1)

        if show:
            plt.show()


class ModExpectedMaximinFitnessInfill(ModulatedMOInfill):
    """
    Modulate the scalar EMFI criterion to a multi-objective criterion to increase spread along the currently existing
    Pareto front.
    """

    def __init__(self, **kwargs):
        underlying = ExpectedMaximinFitnessInfill(**kwargs)
        super(ModExpectedMaximinFitnessInfill, self).__init__(underlying)


if __name__ == '__main__':
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from smt.surrogate_models.kpls import KPLS
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
    from pymoo.factory import get_problem, get_reference_directions

    # ExpectedMaximinFitnessInfill.plot_emfi(n_pareto=5, n=1000, var=.1**2, n_rand=1000, contour=True, show=False)
    # ExpectedMaximinFitnessInfill.plot_emfi(n_pareto=5, n=10000, var=None, show=True), exit()

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = KPLS(n_comp=5, theta0=[1e-2] * 5)
        sbo_emfi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ExpectedMaximinFitnessInfill(n_monte_carlo=1000),
            termination=25, verbose=True,
        )
        sbo_mo_emfi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ModExpectedMaximinFitnessInfill(n_monte_carlo=1000),
            termination=100, verbose=True,
        )
        sbo_y = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimateInfill(),
            termination=100, verbose=True,
        )

        validate_loo_cv = True
        n_eval, n_eval_sbo, n_repeat = 10000, 250, 4
        algorithms = [
            # (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo_y.algorithm(infill_size=25, init_size=50), sbo_y.name, n_eval_sbo),
            # (sbo_emfi.algorithm(infill_size=10, init_size=50), sbo_emfi.name, 100),
            (sbo_mo_emfi.algorithm(infill_size=25, init_size=50), sbo_mo_emfi.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = get_problem('dtlz2', n_var=11, n_obj=2)
        pf = problem.pareto_front(get_reference_directions('das-dennis', problem.n_obj, n_partitions=12))
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(pf),
            IGDMetric(pf),
            InfillMetric(),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, ['min_range'], ['cr'], ['mdr']]

        if validate_loo_cv:
            from arch_opt_exp.algorithms.surrogate.validation import SurrogateQualityMetric
            metrics += [SurrogateQualityMetric(include_loo_cv=True, n_loo_cv=10)]
            plot_names += [['loo_cv']]

        # # Plot infill selection
        # algo_eval = algorithms[3]
        # res_infill: SurrogateBasedInfill = Experimenter(problem, algo_eval[0], n_eval_max=algo_eval[2])\
        #     .run_effectiveness().algorithm.infill
        # res_infill.plot_infill_selection(show=False)
        # for ii in range(problem.n_obj):
        #     res_infill.plot_model(i_y=ii, show=False)
        # for ii in range(problem.n_constr):
        #     res_infill.plot_model(i_y=problem.n_obj+ii, show=False)
        # plt.show(), exit()

        # Run algorithms
        results = [ExperimenterResult.aggregate_results(
            Experimenter(problem, algorithm, n_eval_max=n_eval_algo, metrics=metrics)
                .run_effectiveness_parallel(n_repeat=n_repeat)) for algorithm, _, n_eval_algo in algorithms]

        # Plot metrics
        for ii, metric in enumerate(metrics):
            ExperimenterResult.plot_compare_metrics(
                results, metric.name, titles=[name for _, name, _ in algorithms],
                plot_value_names=plot_names[ii], plot_evaluations=True, show=False)
        plt.show()
