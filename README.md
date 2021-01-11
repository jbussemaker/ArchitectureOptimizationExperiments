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
