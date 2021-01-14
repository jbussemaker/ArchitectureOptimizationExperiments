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

from arch_opt_exp.metrics_base import *
from arch_opt_exp.metrics.filters import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.metrics.convergence import *

__all__ = ['SpreadTermination', 'HVTermination', 'GDTermination', 'IGDTermination', 'MCDTermination', 'SPITermination',
           'FHITermination', 'CRTermination', 'MDRTermination', 'MGBMTermination']


class SpreadTermination(MetricDiffTermination):

    def __init__(self, limit=1e-4, smooth_n=5):
        metric = ExpMovingAverageFilter(SpreadMetric(), n=smooth_n)
        super(SpreadTermination, self).__init__(metric, limit=limit)


class HVTermination(MetricDiffTermination):

    def __init__(self, limit=1e-4, smooth_n=2):
        metric = MovingAverageFilter(HVMetric(), n=smooth_n)
        super(HVTermination, self).__init__(metric, limit=limit)


class GDTermination(MetricDiffTermination):

    def __init__(self, limit=1e-4, smooth_n=2):
        metric = MovingAverageFilter(GDConvergenceMetric(), n=smooth_n)
        super(GDTermination, self).__init__(metric, limit=limit)


class IGDTermination(MetricDiffTermination):

    def __init__(self, limit=1e-4, smooth_n=2):
        metric = MovingAverageFilter(IGDConvergenceMetric(), n=smooth_n)
        super(IGDTermination, self).__init__(metric, limit=limit)


class MCDTermination(MetricDiffTermination):

    def __init__(self, limit=5e-4, smooth_n=2):
        metric = ExpMovingAverageFilter(CrowdingDistanceMetric(), n=smooth_n, filtered_values=['max'])
        super(MCDTermination, self).__init__(metric, value_name='max', limit=limit)


class SPITermination(MetricTermination):

    def __init__(self, n=40, limit=.02, smooth_n=2):
        metric = MovingAverageFilter(SteadyPerformanceIndicator(n_last_steps=n), n=smooth_n)
        super(SPITermination, self).__init__(metric, lower_limit=limit)


class FHITermination(MetricDiffTermination):

    def __init__(self, limit=1e-4, smooth_n=2):
        metric = ExpMovingAverageFilter(FitnessHomogeneityIndicator(), n=smooth_n)
        super(FHITermination, self).__init__(metric, limit=limit)


class CRTermination(MetricTermination):

    def __init__(self, limit=.8, n_delta=1, smooth_n=2):
        metric = ExpMovingAverageFilter(ConsolidationRatioMetric(n_delta=n_delta), n=smooth_n)
        super(CRTermination, self).__init__(metric, upper_limit=limit)


class MDRTermination(MetricTermination):

    def __init__(self, limit=.1, smooth_n=2):
        metric = ExpMovingAverageFilter(MutualDominationRateMetric(), n=smooth_n)
        super(MDRTermination, self).__init__(metric, lower_limit=limit)


class MGBMTermination(MetricTermination):
    """
    Martí-García-Berlanga-Molina stopping criterion: measures convergence by applying a Kalman filter to the MDR metric.

    Implementation based on:
    Martí, L., "A Stopping Criterion for Multi-Objective Evolutionary Algorithms", 2016, 10.1016/j.ins.2016.07.025
    """

    def __init__(self, limit=.1, r=.1, q=.1):
        metric = KalmanFilter(MutualDominationRateMetric(), r=r, q=q)
        super(MGBMTermination, self).__init__(metric, lower_limit=limit)
