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
from typing import *
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from sb_arch_opt.problem import ArchOptProblemBase

from pymoo.core.algorithm import Algorithm
from pymoo.core.indicator import Indicator
from pymoo.core.population import Population
from pymoo.core.termination import Termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = ['Metric', 'IndicatorMetric', 'MetricTermination', 'MetricDiffTermination', 'FilteredMetric']


class Metric:
    """More general metric class, that can output multiple values to track. Should be serializable."""

    def __init__(self):
        self.values = {name: [] for name in self.value_names}
        self.values_std = None  # Used in ExperimenterResults
        self.values_agg = None

    def calculate_step(self, algorithm: Algorithm):
        values = self._calculate_values(algorithm)
        names = self.value_names
        if len(values) != len(names):
            raise ValueError('Values should have the same length as the number of values')

        for i, name in enumerate(names):
            self.values[name].append(values[i])

    def results(self) -> Dict[str, np.ndarray]:
        return {key: np.array(value) for key, value in self.values.items()}

    def results_std(self) -> Dict[str, np.ndarray]:
        if self.values_std is None:
            return {}
        return {key: np.array(value) for key, value in self.values_std.items()}

    def results_agg(self, agg_key) -> Dict[str, np.ndarray]:
        if self.values_agg is None:
            return {}
        return {key: np.array(value[agg_key]) for key, value in self.values_agg.items()}

    def plot(self, show=True, **kwargs):
        self.plot_multiple([self], show=show, **kwargs)

    @staticmethod
    def plot_multiple(metrics: List['Metric'], titles: List[str] = None, colors: List[str] = None,
                      styles: List[str] = None, plot_value_names: List[str] = None, n_eval: List[List[float]] = None,
                      plot_callback=None, save_filename=None, save_svg=False, show=True):
        """Function for plotting multiple metrics of the same kind, but coming from different optimization runs."""

        type_ = type(metrics[0])
        if not all([isinstance(m, type_) for m in metrics]):
            raise ValueError('Metrics should be of same type!')

        if colors is not None and len(colors) != len(metrics):
            raise ValueError('Provide same amount of colors as metrics!')
        if styles is not None and len(styles) != len(metrics):
            raise ValueError('Provide same amount of styles as metrics!')

        if titles is not None and len(titles) != len(metrics):
            raise ValueError('Provide same amount of titles as metrics!')

        if plot_value_names is None:
            plot_value_names = metrics[0].value_names

        if styles is None:
            styles = ['-']*10 + ['-.']*10 + [':']*10 + ['--']*10
            styles = styles*10

        for value_name in plot_value_names:
            fig = plt.figure(figsize=(16, 12))

            x_max = None
            line_handles = []
            for i, metric in enumerate(metrics):
                if metric.values_agg is not None:
                    y = np.atleast_1d(metric.values_agg[value_name]['median'])
                else:
                    y = np.atleast_1d(metric.values[value_name])

                if n_eval is not None:
                    x = np.atleast_1d(n_eval[i])
                    if len(x) != len(y):
                        raise ValueError('List with evaluations should be same length as number of steps!')
                else:
                    x = list(range(len(y)))

                y_q25 = np.array(metric.values_agg[value_name]['q25']) if metric.values_agg is not None else None
                y_q75 = np.array(metric.values_agg[value_name]['q75']) if metric.values_agg is not None else None

                kwargs = {'linewidth': 1}
                if len(metrics) == 1:
                    kwargs['color'] = 'k'
                elif colors is not None:
                    kwargs['color'] = colors[i]
                line_style = styles[i]
                if titles is not None:
                    kwargs['label'] = titles[i]

                l, = plt.plot(x, y, line_style, **kwargs)
                line_handles.append(l)
                color = l.get_color()
                kwargs['color'] = color

                if y_q25 is not None:
                    plt.fill_between(x, y_q25, y_q75, alpha=.05, color=kwargs['color'], linewidth=0)

                metric.plot_fig_callback(x, value_name, color=None if len(metrics) == 1 else color)

                if x_max is None or x[-1] > x_max:
                    x_max = x[-1]

            plt.title('Metric: %s.%s' % (metrics[0].name, value_name))
            plt.xlim([0, x_max])
            plt.xlabel('Iterations' if n_eval is None else 'Function evaluations')
            plt.ylabel(value_name)

            if plot_callback is not None:
                plot_callback(fig, metrics, metrics[0].name, value_name, line_handles, titles)
            elif titles is not None:
                plt.legend()

            if save_filename is not None:
                save_value_filename = '%s_%s' % (save_filename, secure_filename(value_name))
                plt.savefig(save_value_filename+'.png')
                if save_svg:
                    plt.savefig(save_value_filename+'.svg')

        if show:
            plt.show()

    @classmethod
    def calc_doe(cls, problem, n_samples=10000, sampling=None, output=True):
        from pymoo.core.evaluator import Evaluator
        if sampling is None:
            from pymoo.operators.sampling.rnd import FloatRandomSampling
            sampling = FloatRandomSampling()

        pop = sampling.do(problem, n_samples=n_samples)
        Evaluator().eval(problem, pop)

        dummy_algo = Algorithm()
        dummy_algo.pop = pop
        dummy_algo._set_optimum()

        metric = cls()
        metric.calculate_step(dummy_algo)

        if output:
            print('%s results for %r' % (cls.__name__, problem))
            for value_name in metric.value_names:
                print('%s: %r' % (value_name, metric.values[value_name][-1]))
        return metric.results()

    def plot_fig_callback(self, x, value_name: str, color=None):
        pass

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def value_names(self) -> List[str]:
        raise NotImplementedError

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError

    @classmethod
    def _get_pop_x(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        """Design vectors of the population: (n_pop, n_x)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get('X')

    @classmethod
    def _get_pop_f(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        """Objective values of the population: (n_pop, n_f)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get('F')

    @classmethod
    def _get_pop_g(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        """Constraint values of the population: (n_pop, n_g)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get('G')

    @classmethod
    def _get_pop_cv(cls, algorithm: Algorithm, feasible_only=False, valid_only=False) -> np.ndarray:
        """Constraint violation values of the population: (n_pop, n_g)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only, valid_only=valid_only).get('CV')

    @classmethod
    def _get_pop(cls, algorithm: Algorithm, feasible_only=False, valid_only=False):
        pop = algorithm.pop
        if valid_only or feasible_only:
            pop = cls.get_valid_pop(pop)
        if feasible_only:
            i_feasible = np.where(pop.get('feasible'))[0]
            pop = pop[i_feasible]
        return pop

    @staticmethod
    def get_valid_pop(population: Population) -> Population:
        is_failed = ArchOptProblemBase.get_failed_points(population)
        return population[~is_failed]

    @classmethod
    def _get_opt_x(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Design vectors of the optimum population (non-dominated current Pareto front): (n_opt, n_x)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('X').astype(float)

    @classmethod
    def _get_opt_f(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Objective values of the optimum population: (n_opt, n_f)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('F').astype(float)

    @classmethod
    def _get_opt_g(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint values of the optimum population: (n_opt, n_g)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('G').astype(float)

    @classmethod
    def _get_opt_cv(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint violation values of the optimum population: (n_opt, n_g)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('CV').astype(float)

    @staticmethod
    def _get_opt(algorithm: Algorithm, feasible_only=False):
        opt = algorithm.opt
        if feasible_only:
            i_feasible = np.where(opt.get('feasible'))[0]
            return opt[i_feasible]
        return opt

    @staticmethod
    def get_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        i_non_dom = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return np.copy(f[i_non_dom, :])


class IndicatorMetric(Metric):
    """Metric based on a performance indicator. Performance indicators only depend on the objective values."""

    def __init__(self, indicator: Indicator):
        super(IndicatorMetric, self).__init__()
        self.indicator = indicator

    @property
    def name(self) -> str:
        return self.indicator.__class__.__name__

    @property
    def value_names(self) -> List[str]:
        return ['indicator']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        f_opt = self._get_opt_f(algorithm)
        if f_opt.shape[0] == 0 or f_opt.shape[1] == 0:
            return [np.nan]
        return [self.indicator.do(f_opt)]


class MetricTermination(Termination):
    """Termination based on a metric."""

    def __init__(self, metric: Metric, value_name: str = None, lower_limit: float = None, upper_limit: float = None,
                 n_eval_check: int = None):
        if lower_limit is None and upper_limit is None:
            raise ValueError('Provide at least either a lower or an upper limit!')
        self.metric = metric
        self.n_eval = []
        self.value_name = value_name or metric.value_names[0]
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.n_eval_check = n_eval_check
        self._scale = None

        super(MetricTermination, self).__init__()

    @property
    def metric_name(self) -> str:
        return self.metric.name

    def _update(self, algorithm):

        values = self._calc_step(algorithm)
        value = self._get_compare_value(values)
        if np.isnan(value):
            return 0.

        diff = None
        if self.lower_limit is not None:
            diff = value - self.lower_limit

        if self.upper_limit is not None:
            diff = self.upper_limit - value

        if diff is not None:
            if self._scale is None:
                self._scale = diff if diff > 0 else 1.
            progress = max(0, 1 - diff/self._scale)
            return progress

        return 0.

    def _get_compare_value(self, values) -> float:
        return values[-1]

    def _calc_step(self, algorithm: Algorithm):
        do_calc = True
        if len(self.n_eval) > 0 and self.n_eval_check is not None:
            n_eval_next = self.n_eval[-1]+self.n_eval_check
            n_eval_algo = algorithm.evaluator.n_eval
            if n_eval_algo < n_eval_next:
                do_calc = False

        if do_calc:
            self.metric.calculate_step(algorithm)
            self.n_eval.append(algorithm.evaluator.n_eval)
        return self._get_check_values()

    def _get_check_values(self):
        return np.array(self.metric.values[self.value_name])

    def plot(self, save_filename=None, show=True):
        plt.figure()
        plt.title('Metric termination: %s.%s' % (self.metric_name, self.value_name))

        y = self.metric.values[self.value_name]
        x = list(range(len(y)))

        plt.plot(x, y, '-k', linewidth=1)
        if self.lower_limit is not None:
            plt.plot(x, np.ones((len(x),))*self.lower_limit, '--k', linewidth=1)
        if self.upper_limit is not None:
            plt.plot(x, np.ones((len(x),))*self.upper_limit, '--k', linewidth=1)

        plt.xlim([0, x[-1]])
        plt.xlabel('Iteration')
        plt.ylabel(self.value_name)

        if save_filename is not None:
            plt.savefig(save_filename+'.png')
            # plt.savefig(save_filename+'.svg')
        if show:
            plt.show()


class MetricDiffTermination(MetricTermination):
    """Termination based on the rate of change of a metric."""

    def __init__(self, metric: Metric, value_name: str = None, limit: float = None, **kwargs):
        super(MetricDiffTermination, self).__init__(metric, value_name=value_name, lower_limit=limit, **kwargs)

        self.diff_values = []

    def _get_compare_value(self, values) -> float:
        values = np.array(values)
        real_values = values[~np.isnan(values)]

        if len(real_values) < 2:
            return np.nan
        diff = abs(real_values[-1]-real_values[-2])
        self.diff_values.append(diff)
        return diff

    def plot(self, save_filename=None, show=True):
        _ll = self.lower_limit
        self.lower_limit = None
        super(MetricDiffTermination, self).plot(save_filename=save_filename, show=False)
        self.lower_limit = _ll

        plt.figure()
        plt.title('Metric termination (diff): %s.%s' % (self.metric_name, self.value_name))

        y = self.diff_values
        x = list(range(len(y)))

        plt.semilogy(x, y, '-k', linewidth=1)
        plt.semilogy(x, np.ones((len(x),))*self.lower_limit, '--k', linewidth=1)

        plt.xlim([0, x[-1]])
        plt.xlabel('Iteration')
        plt.ylabel(self.value_name+' diff')

        if save_filename is not None:
            plt.savefig(save_filename+'_diff.png')
            # plt.savefig(save_filename+'_diff.svg')
        if show:
            plt.show()


class FilteredMetric(Metric):
    """Base class for a metric that filters another metrics output."""

    def __init__(self, underlying_metric: Metric, filtered_values: List[str] = None):
        self.metric = underlying_metric
        self.filtered_values = set(filtered_values or underlying_metric.value_names)

        super(FilteredMetric, self).__init__()

    @property
    def name(self) -> str:
        return '%s(%s)' % (self.filter_name, self.metric.name)

    @property
    def value_names(self) -> List[str]:
        return self.metric.value_names

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        metric = self.metric
        metric.calculate_step(algorithm)

        values = []
        filtered_values = self.filtered_values
        for value_name in metric.value_names:
            if value_name in filtered_values:
                values.append(self._filter_values(value_name, metric.values[value_name], self.values[value_name]))
            else:
                values.append(metric.values[value_name][-1])

        return values

    def plot_fig_callback(self, x, value_name: str, color=None):
        if self.values_std is not None:  # Aggregated results
            return

        kwargs = {}
        if color is None:
            style = '--b'
        else:
            style = '-.'
            kwargs['color'] = color

        underlying_values = self.metric.values[value_name]
        plt.plot(x, underlying_values, style, linewidth=1, **kwargs)

    @property
    def filter_name(self) -> str:
        raise NotImplementedError

    def _filter_values(self, value_name: str, values: List[float], previous_filtered_values: List[float]) -> float:
        """Get the next filtered value for a list of values of length n (previous filtered values are length n-1)."""
        raise NotImplementedError
