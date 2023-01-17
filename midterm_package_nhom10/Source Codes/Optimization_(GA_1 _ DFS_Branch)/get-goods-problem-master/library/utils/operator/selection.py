from math import ceil

from ...EA import *


class AbstractSelection():
    def __init__(self, *args, **kwds) -> None:
        pass
    def __call__(self, population:Population, nb_inds_tasks:list, *args, **kwds):
        pass

class ElitismSelection(AbstractSelection):
    def __init__(self, random_percent = 0, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        assert 0<= random_percent and random_percent <= 1
        self.random_percent = random_percent
        
        
    def __call__(self, population:Population, nb_inds: int, *args, **kwds) -> Population:
        id_survival = np.argwhere(population.ranking < nb_inds).reshape(-1)
        new_ls_indiv = []
        for id in id_survival:
            new_ls_indiv.append(population[id])
        population.list_indiv.clear()
        population.list_indiv = new_ls_indiv
        population.num_indiv = len(population)
        return population
    
