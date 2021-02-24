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

import contextlib
import numpy as np
from typing import *
from enum import Enum

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.util.dominator import Dominator

__all__ = ['MetricType', 'DominanceRelation', 'MultipleDominanceRelations', 'inject_dominator', 'NSGA2MultiDR']


class MetricType(Enum):
    F = 1  # Objective
    G = 2  # Constraint
    CV = 3  # Constraint violation (max constraint violation for all constraints)


DominanceRelation = Tuple[Tuple[MetricType, int], ...]


class MultipleDominanceRelations(Dominator):
    """
    Dominator that uses the Multiple Dominance Relations (MDR) principle. MDR allows to apply different dominance
    relations to different metrics, both taking objectives and constraints into account. A hierarchy of constraints and
    objectives is determined, whereby lower-hierarchy metrics are searched for within the Pareto-optimal set of metrics
    that are higher up in the hierarchy. It is shown that the best designs are found for the following order: design
    constraints, primary objectives, lower priority objectives. This approach is especially interesting for
    multi-objective design problems where there exists a hierarchy in design goals, for example coming from different
    goals of stakeholders.

    References:
    Cook, L.W., "Design Optimization Using Multiple Dominance Relations", 2020, 10.1002/nme.6316
    Phillips, S., "Assessing the Significance of Nesting Order in Optimization Using Multiple Dominance Relations",
        2020, 10.2514/6.2020-3134
    """

    def __init__(self, hierarchy: List[DominanceRelation]):
        self.hierarchy = hierarchy

        self._dom = Dominator.get_relation

    def get_relation(self, a, b, cva=None, cvb=None, ga=None, gb=None) -> int:
        if ga is not None and len(ga) == 0:
            ga = None
        if gb is not None and len(gb) == 0:
            gb = None

        def _get_ab(dom_rel: DominanceRelation) -> Tuple[np.ndarray, np.ndarray]:
            a_ = np.zeros((len(dom_rel),))
            b_ = np.zeros((len(dom_rel),))

            for i, (met_type, idx) in enumerate(dom_rel):
                if met_type == MetricType.F:
                    a_[i] = a[idx]
                    b_[i] = b[idx]

                elif met_type == MetricType.G:
                    if ga is not None:
                        a_[i] = np.max([ga[idx], 0.])
                    if gb is not None:
                        b_[i] = np.max([gb[idx], 0.])

                elif met_type == MetricType.CV:
                    if cva is not None:
                        a_[i] = cva
                    if cvb is not None:
                        b_[i] = cvb

            return a_, b_

        # Loop over dominance relations
        for dom_relation in self.hierarchy:
            rel = self._dom(*_get_ab(dom_relation))

            # If this relation is not indifferent, we return as we don't need to check any further relations
            if rel != 0:
                return rel

        # None of the relations returned a specific result
        return 0

    def calc_domination_matrix_loop(self, F, G):
        n = F.shape[0]
        CV = np.sum(G * (G > 0).astype(np.float), axis=1)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = self.get_relation(F[i, :], F[j, :], CV[i], CV[j], G[i, :], G[j, :])
                M[j, i] = -M[i, j]

        return M

    def calc_domination_matrix(self, F, _F=None, epsilon=0.0):
        return self.calc_domination_matrix_loop(F, np.zeros((F.shape[0], 0)))


@contextlib.contextmanager
def inject_dominator(dominator: Union[Dominator, Type[Dominator]]):
    """
    Context manager for injecting a specific Dominator subclass (or instance thereof) to be used globally in pymoo.
    Unfortunately this is a necessity as pymoo does not expose a nicely OOP interface for using the Dominator in other
    places of the code.
    """

    funcs = ['get_relation', 'calc_domination_matrix_loop', 'calc_domination_matrix']

    orig_funcs = [getattr(Dominator, func) for func in funcs]
    for func in funcs:
        setattr(Dominator, func, getattr(dominator, func))

    yield

    for i, func in enumerate(funcs):
        setattr(Dominator, func, orig_funcs[i])


class NSGA2MultiDR(NSGA2):
    """NSGA2 with Multiple Dominance Relations instead of the regular dominance relation."""

    def __init__(self, multi_dr: MultipleDominanceRelations, **kwargs):
        super(NSGA2MultiDR, self).__init__(**kwargs)

        self.multi_dr = multi_dr

    def _initialize(self):
        with inject_dominator(self.multi_dr):
            super(NSGA2MultiDR, self)._initialize()

    def _next(self):
        with inject_dominator(self.multi_dr):
            super(NSGA2MultiDR, self)._next()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from arch_opt_exp.experimenter import *
    from pymoo.problems.many.dtlz import DTLZ2
    from arch_opt_exp.metrics.filters import *
    from arch_opt_exp.metrics.convergence import *
    from arch_opt_exp.metrics.performance import *
    from pymoo.factory import get_reference_directions

    with Experimenter.temp_results():
        # Define algorithms to run
        multi_dr_ = MultipleDominanceRelations([
            ((MetricType.F, 0), (MetricType.F, 1)),
            ((MetricType.F, 2), (MetricType.F, 3)),
        ])

        algorithms = [
            NSGA2MultiDR(multi_dr_, pop_size=100),
            NSGA2(pop_size=100),
        ]
        n_eval, n_repeat = 10000, 8

        # Define problem and metrics
        problem = DTLZ2(n_var=13, n_obj=4)
        ref_dirs = get_reference_directions('das-dennis', 4, n_partitions=12)
        pf = problem.pareto_front(ref_dirs)
        metrics = [
            # Metrics for evaluating the algorithm performance
            DeltaHVMetric(pf),
            IGDMetric(pf),

            # Metrics for detecting convergence
            ExpMovingAverageFilter(ConsolidationRatioMetric(), n=5),
            ExpMovingAverageFilter(MutualDominationRateMetric(), n=5),
        ]
        plot_names = [['delta_hv'], None, ['cr'], ['mdr']]

        # Run algorithms
        results = [ExperimenterResult.aggregate_results(
            Experimenter(problem, algorithm, n_eval_max=n_eval, metrics=metrics)
                .run_effectiveness_parallel(n_repeat=n_repeat)) for algorithm in algorithms]

        # Plot metrics
        for ii, metric in enumerate(metrics):
            ExperimenterResult.plot_compare_metrics(
                results, metric.name, titles=[algo.__class__.__name__ for algo in algorithms],
                plot_value_names=plot_names[ii], plot_evaluations=True, show=False)
        plt.show()
