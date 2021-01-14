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
from typing import *
from arch_opt_exp.metrics_base import *

__all__ = ['MovingAverageFilter', 'ExpMovingAverageFilter', 'KalmanFilter']


class MovingAverageFilter(FilteredMetric):

    def __init__(self, underlying_metric: Metric, n: int, **kwargs):
        super(MovingAverageFilter, self).__init__(underlying_metric, **kwargs)

        self.n = n

    @property
    def filter_name(self) -> str:
        return 'moving_average'

    def _filter_values(self, value_name: str, values: List[float], previous_filtered_values: List[float]) -> float:
        if len(values) < self.n:
            return np.nan

        return np.mean(values[-self.n:])


class ExpMovingAverageFilter(MovingAverageFilter):
    """Exponential moving average, react quicker in fast-moving scenarios. Implementation based on pandas implementation
    (pandas.core.window.EWM, pandas._libs.windows.pyx)."""

    @property
    def filter_name(self) -> str:
        return 'exp_moving_average'

    def _filter_values(self, value_name: str, values: List[float], previous_filtered_values: List[float]) -> float:
        values = [value for value in values if value is not None and not np.isnan(value)]
        if len(values) == 0:
            return np.nan

        alpha = 1. / (1. + self.n)
        weight_factor = 1. - alpha
        next_weight = 1.

        ema = values[0]
        weight = 1.
        for i in range(1, len(values)):
            value = values[i]
            weight *= weight_factor
            if ema != value:
                ema = ((weight * ema) + (next_weight * value)) / (weight + next_weight)
            weight += next_weight

        return ema


class KalmanFilter(FilteredMetric):
    """
    One dimensional Kalman filter, based on the implementation in:
    Martí, L., "A Stopping Criterion for Multi-Objective Evolutionary Algorithms", 2016, 10.1016/j.ins.2016.07.025

    The paper reports the use of Q=0 (no prediction error), but this leads to a predicted signal which does not follow
    the measured signal closely. A value of Q in the range of R results in a better prediction.

    Parameters:
        - r: Noise ratio (estimate for variance of noise), between .05 and .15 normally
        - q: Prediction error (lower value (but not zero) brings the prediction signal closer to underlying)
    """

    def __init__(self, underlying_metric: Metric, r=.1, q=.1, **kwargs):
        super(KalmanFilter, self).__init__(underlying_metric, **kwargs)

        self.r = r
        self.q = q

        self.kalman_filters = {}

    @property
    def filter_name(self) -> str:
        return 'kalman'

    def _filter_values(self, value_name: str, values: List[float], previous_filtered_values: List[float]) -> float:
        """Get the next filtered value for a list of values of length n (previous filtered values are length n-1)."""

        if value_name not in self.kalman_filters:
            self.kalman_filters[value_name] = kf = _KalmanFilter(r=self.r, q=self.q)
            for value in values[:-1]:
                kf.process(value)

        kf: _KalmanFilter = self.kalman_filters[value_name]
        kf.process(values[-1])
        return kf.x_estimate


class _KalmanFilter:
    """
    One-dimensional Kalman filter. Based on the implementation discussed in:
    Martí, L., "A Stopping Criterion for Multi-Objective Evolutionary Algorithms", 2016, 10.1016/j.ins.2016.07.025
    """

    def __init__(self, r=.1, a=1., b=0., q=.1, h=1., mu=0., x_init=None, pt_init=None):
        self.a = a  # State matrix
        self.b = b  # Control matrix
        self.q = q  # Prediction error
        self.h = h  # Measurement matrix
        self.mu = mu  # Control signal

        # Measurement noise: controls sensitivity (param between .05 and .15 approx)
        self.r = r

        # Initial estimates
        self.x_init = x_init
        self.pt_init = pt_init or r

        self.samples = []
        self.x_pri = []
        self.x_post = []
        self.kt = []
        self.pt_pri = []
        self.pt_post = []

    def process(self, x_measured):
        """
        Process an observation of the variable to be tracked. Method is based on section 4.2.1 of the paper:
        - Calculate the a priori estimation and covariance
        - Calculate Kalman gain
        - Calculate the a posteriori estimation and covariance

        :return:
        """

        # Get previous results
        x_prev = self.x_post[-1] if len(self.x_post) > 0 else (self.x_init or x_measured)
        pt_prev = self.pt_post[-1] if len(self.pt_post) > 0 else self.pt_init

        # Calculate a-priori values
        x_pri = self._a_priori(x_prev)
        pt_pri = self._a_priori_cov(pt_prev)

        # Calculate Kalman gain
        kt = self._kalman_gain(pt_pri)

        # Calculate a-posteriori values
        x_post = self._a_posteriori(x_pri, kt, x_measured)
        pt_post = self._a_posteriori_cov(kt, pt_pri)

        # Store results
        self.samples.append(x_measured)
        self.x_pri.append(x_pri)
        self.x_post.append(x_post)
        self.kt.append(kt)
        self.pt_pri.append(pt_pri)
        self.pt_post.append(pt_post)

        # Return state estimate
        return x_post

    @property
    def x_estimate(self):
        if len(self.x_post) == 0:
            return None
        return self.x_post[-1]

    @property
    def cov_estimate(self):
        if len(self.pt_post) == 0:
            return None
        return self.pt_post[-1]

    def _a_priori(self, x_prev):  # xt-, Eq. 12
        return self.a*x_prev + self.b*self.mu

    def _a_priori_cov(self, pt_prev):  # Pt-, Eq. 13
        return pt_prev*self.a**2 + self.q

    def _kalman_gain(self, pt_pri):  # Kt, Eq. 14 / Eq. 21
        return (pt_pri*self.h) / (pt_pri*self.h**2 + self.r)

    def _a_posteriori(self, x_pri, kt, x_measured):  # Xt, Eq. 15
        return x_pri + kt*(x_measured - self.h*x_pri)

    def _a_posteriori_cov(self, kt, pt_pri):  # Pt, Eq. 16
        return (1-kt*self.h)*pt_pri
