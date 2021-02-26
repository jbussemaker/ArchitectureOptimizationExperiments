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
from pymoo.util.normalization import normalize
from pymoo.performance_indicator.hv import Hypervolume
from arch_opt_exp.algorithms.surrogate.mo.mo_modulate import *
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *

__all__ = ['ExpectedHypervolumeImprovementInfill', 'ModExpectedHypervolumeImprovementInfill']


class ExpectedHypervolumeImprovementInfill(ProbabilityOfFeasibilityInfill):
    """
    The Expected Hypervolume Improvement (EHVI) criterion measures how much the hypervolume can be expected to increase
    in size given a function estimate and its variance.

    Note that the EHVI is a computationally expensive infill criterion.

    Implementation based on:
    Emmerich, M., "Single- and Multiobjective Evolutionary Optimization Assisted by Gaussian Random Field Metamodels",
        2006, 10.1109/TEVC.2005.859463
    """

    def __init__(self, n_monte_carlo: int = 1000, **kwargs):
        super(ExpectedHypervolumeImprovementInfill, self).__init__(**kwargs)

        self.n_mc = n_monte_carlo
        self.f_pareto = None
        self.nadir_point = None
        self.ideal_point = None
        self.f_pareto_norm = None
        self.hv = None

    def set_samples(self, x_train: np.ndarray, y_train: np.ndarray):
        super(ExpectedHypervolumeImprovementInfill, self).set_samples(x_train, y_train)

        self.f_pareto = f_pareto = self.get_pareto_front(y_train[:, :self.problem.n_obj])

        self.nadir_point, self.ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        self.f_pareto_norm = f_pareto_norm = normalize(f_pareto, x_max=self.nadir_point, x_min=self.ideal_point)
        self.hv = self._hv(f_pareto_norm)

    def get_n_infill_objectives(self) -> int:
        return 1

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self._evaluate_f_static(f_predict, f_var_predict, self.f_pareto, self.nadir_point, self.ideal_point,
                                       self.hv, self.f_pareto_norm, self.n_mc)

    @classmethod
    def _evaluate_f_kwargs(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, n_mc=1000) -> dict:
        nadir_point, ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        f_pareto_norm = normalize(f_pareto, x_max=nadir_point, x_min=ideal_point)
        hv = cls._hv(f_pareto_norm)

        return {
            'nadir_point': nadir_point,
            'ideal_point': ideal_point,
            'hv': hv,
            'f_pareto_norm': f_pareto_norm,
            'n_mc': n_mc,
        }

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, nadir_point=None,
                           ideal_point=None, hv=None, f_pareto_norm=None, n_mc=1000) -> np.ndarray:
        f_predict_norm, f_predict_var_norm = cls._normalize_f_var(
            f, f_var, nadir_point, ideal_point)

        ehvi = np.empty((f.shape[0], 1))
        for i in range(f_predict_norm.shape[0]):
            ehvi[i, 0] = cls._ehvi(f_pareto_norm, f_predict_norm[i, :], f_predict_var_norm[i, :], hv, n=n_mc)
        return 1.-ehvi

    @classmethod
    def _ehvi(cls, f_pareto_norm: np.ndarray, f_norm: np.ndarray, var_norm: np.ndarray, hv: float, n=10000) -> float:
        f_rand = np.random.normal(loc=f_norm, scale=np.sqrt(var_norm), size=(n, len(f_norm)))

        hvi_sampled = np.empty((n,))
        for i in range(n):
            hvi_sampled[i] = cls._hvi(f_pareto_norm, f_rand[i, :], hv)
        return np.mean(hvi_sampled)

    @classmethod
    def _hvi(cls, f_pareto_norm: np.ndarray, f: np.ndarray, hv: float) -> float:
        hv_mod = cls._hv(np.concatenate([f_pareto_norm, np.array([f])]))
        return hv_mod-hv

    @staticmethod
    def _hv(f_pareto_norm: np.ndarray) -> float:
        hv_obj = Hypervolume(ref_point=np.ones(f_pareto_norm.shape[1]))
        return hv_obj.calc(f_pareto_norm)

    @staticmethod
    def _normalize_f_var(f: np.ndarray, f_var: np.ndarray, nadir_point, ideal_point):
        f_norm = normalize(f, x_max=nadir_point, x_min=ideal_point)

        norm = nadir_point-ideal_point
        f_var_norm = f_var/(norm**2)

        return f_norm, f_var_norm


class ModExpectedHypervolumeImprovementInfill(ModulatedMOInfill):
    """
    Modulate the scalar EHVI criterion to a multi-objective criterion to increase spread along the currently existing
    Pareto front.

    Note that the EHVI is a computationally expensive infill criterion.
    """

    def __init__(self, **kwargs):
        underlying = ExpectedHypervolumeImprovementInfill(**kwargs)
        super(ModExpectedHypervolumeImprovementInfill, self).__init__(underlying)


if __name__ == '__main__':
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
    from pymoo.factory import get_problem, get_reference_directions
    from arch_opt_exp.surrogates.smt.smt_krg import SMTKPLSSurrogateModel

    # ExpectedHypervolumeImprovementInfill.plot(var=.05**2, n_pareto=5, n=20, n_mc=200), exit()

    # ExpectedHypervolumeImprovementInfill.benchmark_evaluation_time(n_pareto=5, n_f=100, n_mc=100)
    # ExpectedHypervolumeImprovementInfill.benchmark_evaluation_time(n_pareto=10, n_f=100, n_mc=100)
    # exit()

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = SMTKPLSSurrogateModel(n_comp=5, theta0=1e-2)
        sbo_ehvi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ExpectedHypervolumeImprovementInfill(n_monte_carlo=200),
            termination=10, verbose=True,
        )
        sbo_mo_ehvi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ModExpectedHypervolumeImprovementInfill(n_monte_carlo=200),
            termination=10, verbose=True,
        )
        sbo_y = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimateInfill(),
            termination=100, verbose=True,
        )

        n_eval, n_eval_sbo, n_repeat = 10000, 500, 8
        algorithms = [
            (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo_y.algorithm(infill_size=25, init_size=50), sbo_y.name, n_eval_sbo),

            (sbo_ehvi.algorithm(infill_size=10, init_size=50), sbo_ehvi.name, 100),
            (sbo_mo_ehvi.algorithm(infill_size=25, init_size=50), sbo_mo_ehvi.name, n_eval_sbo),
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
