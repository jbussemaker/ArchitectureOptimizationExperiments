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

from typing import *
from arch_opt_exp.experiments import runner
from arch_opt_exp.metrics.termination import *
from arch_opt_exp.metrics.performance import *
from arch_opt_exp.metrics_base import MetricTermination

from arch_opt_exp.experiments.effectiveness.effectiveness import run_effectiveness_analytical


def run_efficiency_analytical(do_run=True):
    exp = run_effectiveness_analytical(return_exp=True)
    mt = get_metric_terminations()
    plot_metric_values = {
        'delta_hv': ['delta_hv'],
        'IGD': None,
        'spread': ['delta'],
    }
    run(exp, mt, plot_metric_values=plot_metric_values, do_run=do_run)


def get_metric_terminations():
    n_eval_check = 10
    return [
        MetricTermination(NrEvaluationsMetric(), value_name='n_eval', upper_limit=400, n_eval_check=n_eval_check),
        MDRTermination(limit=.1, smooth_n=2, n_eval_check=n_eval_check),
        MGBMTermination(limit=.1, r=.1, q=.1, n_eval_check=n_eval_check),
        FHITermination(limit=1e-4, smooth_n=2, n_eval_check=n_eval_check),
        GDTermination(limit=1e-3, smooth_n=2, n_eval_check=n_eval_check),
        IGDTermination(limit=1e-3, smooth_n=2, n_eval_check=n_eval_check),
        SpreadTermination(limit=1e-4, smooth_n=5, n_eval_check=n_eval_check),
        HVTermination(limit=1e-3, smooth_n=2, n_eval_check=n_eval_check),
        CRTermination(limit=.8, n_delta=1, smooth_n=2, n_eval_check=n_eval_check),
        MCDTermination(limit=5e-4, smooth_n=2, n_eval_check=n_eval_check),
        SPITermination(n=4, limit=.02, smooth_n=2, n_eval_check=n_eval_check),
    ]


def run(exp, metric_terminations: List[MetricTermination], plot_metric_values=None, do_run=True):
    if do_run:
        runner.run_efficiency_multi(exp, metric_terminations)
    runner.plot_efficiency_results(exp, metric_terminations, plot_metric_values, save=True, show=False)


if __name__ == '__main__':
    run_efficiency_analytical(
        # do_run=False,
    )
