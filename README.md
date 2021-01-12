# Architecture Optimization Experiments

Code for experimenting with algorithms for system architecture optimization.

A turbofan engine architecting benchmark problem is used to investigate what optimization algorithms might be useful
for optimizing system architectures. System architecture optimization problems are hard to solve because they are
generally **black-box, hierarchical, mixed-integer, multi-objective** optimization problems. Black-box, because analysis
is done with simulation code, and therefore no analytical description of the design space is available. Hierarchical,
because design variables might activate/deactivate other design variables (i.e. there exists a hierarchy between design
variables, and the design space size itself is variable). Mixed-integer, because architectural decisions can both be
discrete (yes/no include this element) and continuous (e.g. sizing parameters). Multi-objective, because in general
there might be multiple conflicting design goals to be satisfied, coming from conflicting system stakeholder needs.

This repository aims to create a reproducible set of experiments to help explain the salient features of architecture
optimization, and investigate effectiveness and efficiency of optimization algorithms. Effectiveness represents how
well an algorithm is able to find a Pareto front (e.g. how close it is to some pre-known Pareto front, and how well
spread it is along this front). Efficiency represents the trade-off between effectiveness and number of function
evaluations (i.e. system simulations) needed to get to a certain result quality.

## Experimenter

The `Experimenter` class handles running of experiments against a combination of a problem and an algorithm. It can run
two types of experiments:
1. *Effectiveness*, for determining how effective an algorithm is at solving the problem, by running the algorithm with
   a fixed function evaluation budget.
2. *Efficiency*, for investigating the trade-off between effectiveness and evaluation cost, by determining when an
   effectiveness run would have terminated with a given termination metric.

```python
from arch_opt_exp.metrics import *
from arch_opt_exp.experimenter import *

from pymoo.factory import get_problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.performance_indicator.igd import IGD

# Instantiate the problem, algorithm and experimenter
problem = get_problem('zdt1')
algorithm = NSGA2(pop_size=100)  # Don't worry about the termination at this point

igd_metric = IndicatorMetric(IGD(problem.pareto_front()))

experimenter = Experimenter(
    problem, algorithm,
    n_eval_max=20000,
    metrics=[igd_metric],
    results_folder='results',
)

# Run the effectiveness experiment multiple times to correct for random effects
n = 10
effectiveness_results = experimenter.run_effectiveness_parallel(n_repeat=n)
assert len(effectiveness_results) == n

# Alternative (offline) loading of results
effectiveness_result = experimenter.get_effectiveness_result(repeat_idx=0)
metric_values = effectiveness_result.metrics[igd_metric.name].results()['indicator']

# Run the efficiency experiment with a given termination metric
metric_termination = MetricTermination(igd_metric, lower_limit=.5)  # Define convergence limit

efficiency_results = experimenter.run_efficiency_repeated(metric_termination)
assert len(effectiveness_result) == n

efficiency_result = experimenter.get_efficiency_result(metric_termination, repeat_idx=0)
assert efficiency_result.metric_converged
assert len(efficiency_result.history) < len(effectiveness_result.history)
assert len(efficiency_result.termination.metric.results()['indicator']) == len(efficiency_result.history)
```
