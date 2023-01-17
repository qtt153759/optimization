from typing import Tuple
from ...EA import *

import numpy as np

class AbstractCrossover:
    def __init__(self):
        pass
    def __call__(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        pass


class Two_Cut_Crossover(AbstractCrossover):
    def __init__(self):
        super().__init__()
    def __call__(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        p1 = parent1.genes[1:]
        p2 = parent2.genes[1:]
        dim = len(p1)
        i1 = np.random.randint(dim - 1)
        i2 = np.random.randint(i1 + 1, dim)

        o1 = np.zeros(dim, dtype=int)
        o2 = np.zeros(dim, dtype=int)

        o1[i1 : i2] = p1[i1 : i2]
        # print(o1[i1 : i2])
        i = 0
        for gene in p2:
            if i == len(p2):
                break
            if i == i1:
                i = i2
            if gene not in o1:
                o1[i] = gene
                i += 1
        o2[i1 : i2] = p2[i1 : i2]
        i = 0
        for gene in p1:
            if i == len(p1):
                break
            if i == i1:
                i = i2
            if gene not in o2:
                o2[i] = gene
                i += 1
        offspring1 = Individual(dim + 1)
        offspring2 = Individual(dim + 1)
        offspring1.genes[1:] = o1
        offspring2.genes[1:] = o2
        return offspring1, offspring2



class OX_Crossover(AbstractCrossover):
    def __init__(self):
        super().__init__()
    def __call__(self, parent1: Individual, parent2: Individual, i1=0, i2=0) -> Tuple[Individual, Individual]:
        p1 = parent1.genes[1:]
        p2 = parent2.genes[1:]
        dim = len(p1)
        i1 = np.random.randint(dim - 2)
        i2 = np.random.randint(i1 + 1, dim-1)
        o1 = np.zeros(dim, dtype=int)
        o2 = np.zeros(dim, dtype=int)
        
        o1[i1 : i2] = p1[i1 : i2]
        i = i2
        for g2 in p2:
            if g2 not in o1:
                if i == dim:
                    i = 0
                o1[i] = g2
                i += 1
        
        o2[i1 : i2] = p2[i1 : i2]
        i = i2
        for g1 in p1:
            if g1 not in o2:
                if i == dim:
                    i = 0
                o2[i] = g1
                i += 1
       
        offspring1 = Individual(dim + 1)
        offspring2 = Individual(dim + 1)
        offspring1.genes[1:] = o1
        offspring2.genes[1:] = o2
        return offspring1, offspring2