from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from arch_opt_exp.algorithms.sampling import RepairedLatinHypercubeSampling

__all__ = ['get_ga_algo']


def get_ga_algo(pop_size=100, repair: Repair = None, **kwargs):
    """NSGA2 (a multi-objective genetic algorithm)"""
    return NSGA2(pop_size=pop_size, sampling=RepairedLatinHypercubeSampling(repair), **kwargs)
