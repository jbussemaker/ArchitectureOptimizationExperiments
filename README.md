[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4964341.svg)](https://doi.org/10.5281/zenodo.4964341)

# Architecture Optimization Experiments

Code for experimenting with algorithms for system architecture optimization. System architectures describe how system
components are allocated to perform functions; choosing what does what are important decisions taken early in the design
process with large impact on system performance.

For an overview of and practical implementation of architecture optimization, refer to
[SBArchOpt](https://github.com/jbussemaker/SBArchOpt).

This repository aims to create a reproducible set of experiments to help explain the salient features of architecture
optimization, and investigate effectiveness and efficiency of optimization algorithms. Effectiveness represents how
well an algorithm is able to find a Pareto front (e.g. how close it is to some pre-known Pareto front, and how well
spread it is along this front). Efficiency represents the trade-off between effectiveness and number of function
evaluations (i.e. system simulations) needed to get to a certain result quality.

Optimization is done using the [SBArchOpt](https://github.com/jbussemaker/SBArchOpt), which under the hood uses
[pymoo](https://pymoo.org/) (multi-objective optimization in python) framework. This framework includes many
multi-objective evolutionary algorithms..

## Citing

For the accompanying paper, please refer to:
[Effectiveness of Surrogate-Based Optimization Algorithms for System Architecture Optimization](https://arc.aiaa.org/doi/10.2514/6.2021-3095)

Please cite this paper if you are using this code in your work.

## Installing

```
conda create --name opt python=3.9
conda activate opt
conda install numpy
pip install -r requirements.txt
```

## Analytical Test Problems

There are two main architecture benchmark problems. Both are based on the Goldstein problem, and include mixed-discrete
and hierarchical design variables, and have two objectives. They are implemented in SBArchOpt.

### Test Problem

![Pareto front](resources/pf_an_prob.svg)

Properties:
- 2 objectives
- 27 design variables
  - 16 continuous
  - 6 integer
  - 5 categorical
- ~42% of design variables are active in the initial DOE

```python
from sb_arch_opt.problems.hierarchical import MOHierarchicalTestProblem

problem = MOHierarchicalTestProblem()
problem.print_stats()

# Run an optimization using NSGA2 to visualize the Pareto front
problem.plot_pf()
```

### Test Problem with Hidden Constraints

![Pareto front](resources/pf_an_prob_hc.svg)

Properties:
- Same as above
- ~60% of evaluations failed in the initial DOE

```python
from sb_arch_opt.problems.hidden_constraints import HCMOHierarchicalTestProblem

problem = HCMOHierarchicalTestProblem()
problem.print_stats()

# Run an optimization using NSGA2 to visualize the Pareto front
problem.plot_pf()
```
