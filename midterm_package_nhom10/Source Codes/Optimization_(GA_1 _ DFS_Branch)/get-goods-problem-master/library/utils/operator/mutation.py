from ...EA import *

import numpy as np

class AbstractMutation:
    def __init__(self, *arg, **kwargs):
        self.pm = None
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        pass

class SwapMutation(AbstractMutation):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
    def __call__(self, ind: Individual, return_newInd=True, *arg, **kwargs) -> Individual:
        o = np.copy(ind.genes[1:])
        dim = len(o)
        i1 = np.random.randint(dim - 1)
        i2 = np.random.randint(i1 + 1, dim)
        o = np.copy(o)
        temp = o[i1]
        o[i1] = o[i2]
        o[i2] = temp

        offspring = Individual(dim + 1)
        offspring.genes[1:] = o
        return offspring


