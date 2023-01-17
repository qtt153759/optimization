from ..object.EA import *
import numpy as np


class ElitismSelection:
    def __init__(self, select_percent = 0.5) -> None:
        assert 0<= select_percent and select_percent <= 1
        self.select_percent = select_percent

    def __call__(self, population: Population) -> list[int]:
        fitness = population.fitness
        #print(f'Number of inds: {len(fitness)} - fitness:{fitness}')
        num_selected = int(len(fitness)*self.select_percent)
        return np.argsort(np.array(fitness))[num_selected:]