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
from scipy.stats import norm
from pymoo.util.normalization import normalize
from arch_opt_exp.algorithms.surrogate.p_of_feasibility import *

__all__ = ['ProbabilityOfImprovementInfill', 'ExpectedImprovementInfill', 'LowerConfidenceBoundInfill',
           'EstimateVarianceInfill']


class ProbabilityOfImprovementInfill(ProbabilityOfFeasibilityInfill):
    """
    Probability of Improvement represents the probability that some point will be better than the current best estimate
    with some offset:

    PoI(x) = Phi((T - y(x))/sqrt(s(x)))
    where
    - Phi is the cumulative distribution function of the normal distribution
    - T is the improvement target (current best estimate minus some offset)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate

    PoI was developed for single-objective optimization, and because of the use of the minimum current objective value,
    it tends towards suggesting improvement points only at the edges of the Pareto front. It has been modified to
    evaluate the PoI with respect to the closest Pareto front point instead.

    Implementation based on:
    Hawe, G.I., "An Enhanced Probability of Improvement Utility Function for Locating Pareto Optimal Solutions", 2007
    """

    def __init__(self, f_min_offset: float = 0., **kwargs):
        super(ProbabilityOfImprovementInfill, self).__init__(**kwargs)

        self.f_min_offset = f_min_offset

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self._evaluate_f_poi(f_predict, f_var_predict, self.y_train[:, :f_predict.shape[1]], self.f_min_offset)

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        return cls._evaluate_f_poi(f, f_var, f_pareto, **kwargs)

    @classmethod
    def _evaluate_f_poi(cls, f: np.ndarray, f_var: np.ndarray, f_current: np.ndarray, f_min_offset=0.) -> np.ndarray:
        # Normalize current and predicted objectives
        f_pareto = cls.get_pareto_front(f_current)
        nadir_point, ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        f_pareto_norm = normalize(f_pareto, x_max=nadir_point, x_min=ideal_point)
        f_norm, f_var_norm = cls._normalize_f_var(f, f_var, nadir_point, ideal_point)

        # Get PoI for each point using closest point in the Pareto front
        f_poi = np.empty(f.shape)
        for i in range(f.shape[0]):
            i_par_closest = np.argmin(np.sum((f_pareto_norm-f_norm[i, :])**2, axis=1))
            f_par_targets = f_pareto_norm[i_par_closest, :]-f_min_offset
            poi = cls._poi(f_par_targets, f_norm[i, :], f_var_norm[i, :])
            f_poi[i, :] = 1.-poi

        return f_poi

    @staticmethod
    def _normalize_f_var(f: np.ndarray, f_var: np.ndarray, nadir_point, ideal_point):
        f_norm = normalize(f, x_max=nadir_point, x_min=ideal_point)
        f_var_norm = f_var/((nadir_point-ideal_point)**2)
        return f_norm, f_var_norm

    @staticmethod
    def _poi(f_targets: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        return norm.cdf((f_targets-f) / np.sqrt(f_var))


class ExpectedImprovementInfill(ProbabilityOfImprovementInfill):
    """
    The Expected Improvement (EI) naturally balances exploitation and exploration by representing the expected amount
    of improvement at some point taking into accounts its probability of improvement.

    EI(x) = (f_min-y(x)) * Phi((f_min - y(x))/s(x)) + s(x) * phi((f_min - y(x)) / s(x))
    where
    - f_min is the current best point (real)
    - y(x) the surrogate model estimate
    - s(x) the surrogate model variance estimate
    - Phi is the cumulative distribution function of the normal distribution
    - phi is the probability density function of the normal distribution

    EI was developed for single-objective optimization, and because of the use of the minimum current objective value,
    it tends towards suggesting improvement points only at the edges of the Pareto front. It has been modified to
    evaluate the EI with respect to the closest Pareto front point instead.

    Implementation based on:
    Jones, D.R., "Efficient Global Optimization of Expensive Black-Box Functions", 1998, 10.1023/A:1008306431147
    """

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        return self._evaluate_f_ei(f_predict, f_var_predict, self.y_train[:, :f_predict.shape[1]])

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        return cls._evaluate_f_ei(f, f_var, f_pareto)

    @classmethod
    def _evaluate_f_ei(cls, f: np.ndarray, f_var: np.ndarray, f_current: np.ndarray) -> np.ndarray:
        # Normalize current and predicted objectives
        f_pareto = cls.get_pareto_front(f_current)
        nadir_point, ideal_point = np.max(f_pareto, axis=0), np.min(f_pareto, axis=0)
        f_pareto_norm = normalize(f_pareto, x_max=nadir_point, x_min=ideal_point)
        f_norm, f_var_norm = cls._normalize_f_var(f, f_var, nadir_point, ideal_point)

        # Get PoI for each point using closest point in the Pareto front
        f_ei = np.empty(f.shape)
        for i in range(f.shape[0]):
            i_par_closest = np.argmin(np.sum((f_pareto_norm-f_norm[i, :])**2, axis=1))
            f_par_min = f_pareto_norm[i_par_closest, :]
            ei = cls._ei(f_par_min, f_norm[i, :], f_var_norm[i, :])
            ei[ei < 0.] = 0.
            f_ei[i, :] = 1.-ei

        return f_ei

    @staticmethod
    def _ei(f_min: np.ndarray, f: np.ndarray, f_var: np.ndarray) -> np.ndarray:
        dy = f_min-f
        ei = dy*norm.cdf(dy/np.sqrt(f_var)) + f_var*norm.pdf(dy/np.sqrt(f_var))
        return ei


class LowerConfidenceBoundInfill(ProbabilityOfFeasibilityInfill):
    """
    The Lower Confidence Bound (LCB) represents the lowest expected value to be found at some point given its standard
    deviation.

    LCB(x) = y(x) - alpha * sqrt(s(x))
    where
    - y(x) the surrogate model estimate
    - alpha is a scaling parameter (typical value is 2) --> lower means more exploitation, higher more exploration
    - s(x) the surrogate model variance estimate
    
    Implementation based on:
    Cox, D., "A Statistical Method for Global Optimization", 1992, 10.1109/icsmc.1992.271617
    """

    def __init__(self, alpha: float = 2., **kwargs):
        super(LowerConfidenceBoundInfill, self).__init__(**kwargs)

        self.alpha = alpha

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        lcb = f_predict-self.alpha*np.sqrt(f_var_predict)
        return lcb

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, alpha=2.) -> np.ndarray:
        return f-alpha*np.sqrt(f_var)


class EstimateVarianceInfill(ProbabilityOfFeasibilityInfill):
    """
    Add the function estimate and the variances directly as objectives for the infill problem, so that the trade-off
    between exploration and exploitation is automatically satisfied.

    This is similar to the Multiobjective Infill Criterion (MIC) of:
    Tian, J., "Multiobjective Infill Criterion Driven Gaussian Process-Assisted Particle Swarm Optimization of
        High-Dimensional Expensive Problems", 2019, 10.1109/TEVC.2018.2869247
    """

    def __init__(self, **kwargs):
        super(EstimateVarianceInfill, self).__init__(**kwargs)

        self.std_max = None

    def get_n_infill_objectives(self) -> int:
        return self.problem.n_obj*2

    def _evaluate_f(self, f_predict: np.ndarray, f_var_predict: np.ndarray) -> np.ndarray:
        f_std_predict = np.sqrt(f_var_predict)
        if self.std_max is None:
            self.std_max = np.max(f_std_predict, axis=0)
            self.std_max[self.std_max == 0] = 1.

        return self._evaluate_f_var_infill(f_predict, f_var_predict, std_max=self.std_max)

    @classmethod
    def _evaluate_f_static(cls, f: np.ndarray, f_var: np.ndarray, f_pareto: np.ndarray, **kwargs) -> np.ndarray:
        return cls._evaluate_f_var_infill(f, f_var)

    @classmethod
    def _evaluate_f_var_infill(cls, f: np.ndarray, f_var: np.ndarray, std_max=None) -> np.ndarray:
        n_f = f.shape[1]
        f_out = np.empty((f.shape[0], n_f*2))

        # Function estimates as first set of objectives
        f_out[:, :n_f] = f[:, :]

        # Variances as second set of objectives
        f_std_predict = np.sqrt(f_var)
        if std_max is None:
            std_max = np.max(f_std_predict, axis=0)
            std_max[std_max == 0] = 1.

        f_out[:, n_f:] = 1.-f_std_predict/std_max

        return f_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from pymoo.algorithms.nsga2 import NSGA2
    from smt.surrogate_models.kpls import KPLS
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from arch_opt_exp.algorithms.surrogate.func_estimate import *
    from arch_opt_exp.algorithms.surrogate.surrogate_infill import *
    from pymoo.factory import get_problem, get_reference_directions

    # ProbabilityOfImprovementInfill.plot(var=.05**2, n_pareto=5, show=False)
    # ProbabilityOfImprovementInfill.plot(var=.05**2, n_pareto=5, concave=True, show=False)
    # # ExpectedImprovementInfill.plot(var=.05**2, n_pareto=5, show=False)
    # # ExpectedImprovementInfill.plot(var=.05**2, n_pareto=5, concave=True, show=False)
    # # LowerConfidenceBoundInfill.plot(var=.05**2, n_pareto=5, alpha=2., show=False)
    # # EstimateVarianceInfill.plot(var=.05**2, n_pareto=5, show=False)
    # plt.show()
    # exit()

    # ProbabilityOfImprovementInfill.benchmark_evaluation_time(n_pareto=5, n_f=1000)
    # ProbabilityOfImprovementInfill.benchmark_evaluation_time(n_pareto=10, n_f=1000)
    # ExpectedImprovementInfill.benchmark_evaluation_time(n_pareto=5, n_f=1000)
    # ExpectedImprovementInfill.benchmark_evaluation_time(n_pareto=10, n_f=1000)
    # # LowerConfidenceBoundInfill.benchmark_evaluation_time(n_pareto=5, n_f=1000, alpha=2.)
    # # LowerConfidenceBoundInfill.benchmark_evaluation_time(n_pareto=10, n_f=1000, alpha=2.)
    # # EstimateVarianceInfill.benchmark_evaluation_time(n_pareto=5, n_f=1000)
    # # EstimateVarianceInfill.benchmark_evaluation_time(n_pareto=10, n_f=1000)
    # exit()

    with Experimenter.temp_results():
        # Define algorithms to run
        surrogate_model = KPLS(n_comp=5, theta0=[1e-2]*5)
        sbo_poi = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ProbabilityOfImprovementInfill(f_min_offset=0.),
            termination=100, verbose=True,
        )
        sbo_ei = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=ExpectedImprovementInfill(),
            termination=100, verbose=True,
        )
        sbo_lcb = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=LowerConfidenceBoundInfill(alpha=2.),
            termination=100, verbose=True,
        )
        sbo_est_var = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=EstimateVarianceInfill(),
            termination=100, verbose=True,
        )
        sbo_y = SurrogateBasedInfill(
            surrogate_model=surrogate_model,
            infill=FunctionEstimateInfill(),
            termination=100, verbose=True,
        )

        n_eval, n_eval_sbo, n_repeat = 10000, 500, 8
        algorithms = [
            (NSGA2(pop_size=100), 'NSGA2', n_eval),
            (sbo_y.algorithm(infill_size=50, init_size=100), sbo_y.name, n_eval_sbo),
            (sbo_lcb.algorithm(infill_size=50, init_size=100), sbo_lcb.name, n_eval_sbo),
            (sbo_est_var.algorithm(infill_size=50, init_size=100), sbo_est_var.name, n_eval_sbo),
            (sbo_poi.algorithm(infill_size=50, init_size=100), sbo_poi.name, n_eval_sbo),
            (sbo_ei.algorithm(infill_size=50, init_size=100), sbo_ei.name, n_eval_sbo),
        ]

        # Define problem and metrics
        problem = get_problem('dtlz2', n_var=11, n_obj=3)
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
        # algo_eval = algorithms[2]
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
