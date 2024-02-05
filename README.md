# Architecture Optimization Experiments

Code for experimenting with algorithms for system architecture optimization. System architectures describe how system
components are allocated to perform functions; choosing what does what are important decisions taken early in the design
process with large impact on system performance.

For an overview of and practical implementation of architecture optimization, refer to
[SBArchOpt](https://github.com/jbussemaker/SBArchOpt).

This repository aims to create a reproducible set of experiments to help explain the salient features of architecture
optimization, and investigate effectiveness and efficiency of optimization algorithms.

Optimization is done using the [SBArchOpt](https://github.com/jbussemaker/SBArchOpt), which under the hood uses
[pymoo](https://pymoo.org/) (multi-objective optimization in python) framework. This framework includes many
multi-objective evolutionary algorithms..

## Reproducibility

This version of the repository was used to generate results for two papers.
Here, we list which results were generated with which experiment function.
Experiment files are found in the `arch_opt_exp.experiments` module.

Note: `sbo=x` refers to `sbo=False` for the NSGA-II results and `sbo=True` for the BO results.

### Paper 1: Bussemaker, J.H., et al., "System Architecture Optimization Strategies: Dealing with Expensive Hierarchical Problems", 2024.

1. Hierarchical sampling (Tables 10 and 11): `exp_01_sampling.exp_01_05_correction(sampling=True, sbo=x)`
2. Hierarchical correction (Tables 13 and 14): `exp_01_sampling.exp_01_05_correction(sampling=False, sbo=x)`
3. Hierarchical integration strategies (Tables 17 and 18): `exp_02_hierarchy.exp_02_02_hier_strategies(sbo=x)`
4. Jet engine application (Figures 7 and 9, Table 19): `exp_03_hidden_constraints.exp_03_07_engine_arch()`

### Paper 2: Bussemaker, J.H., et al., "Surrogate-Based Optimization of System Architectures Subject to Hidden Constraints", AIAA Aviation Forum 2024.

1. Example optimization steps (Figure 3): `exp_03_hidden_constraints.exp_03_04_simple_optimization()`
2. Strategy performances (Tables 4 and 5): `exp_03_hidden_constraints.exp_03_05_optimization()`
3. Detailed strategy configurations (Table 6 and Figure 4): `exp_03_hidden_constraints.exp_03_04a_doe_size_min_pov()`
4. Jet engine application (Figure 5): `exp_03_hidden_constraints.exp_03_07_engine_arch()`

## Installing

```
conda create --name opt python=3.9 numpy
conda activate opt
pip install -r requirements.txt
```
