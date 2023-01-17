import numpy as np
from ..tasks.task import AbstractTask

class Individual:
    def __init__(self, genes: np.ndarray, task: AbstractTask):
        self.genes = genes
        self.dim = len(genes)
        self.fitness = task(self.genes)
        self.total_candidate_per_slot = [0 for i in range(self.dim)]

    # def cal_num_seat(self):
    #     for idx, gen in enumerate(self.genes):
    #         self.total_candidate_per_slot[gen] += self.task.

class Population:
    def __init__(self, num_inds: int, task: AbstractTask) -> None:
        self.num_inds = num_inds
        self.task = task
        self.dim = task.num_course
        self.population = [Individual(task.encode(), self.task) for i in range(num_inds)]
        self.fitness: list
        self.eval()
    def __getRandomIndividual__(self, size: int):
        output = []
        ids = np.random.randint(low=0,high=self.num_inds,size=size)
        for i in ids:
            output.append(self.population[i])
        return output

    def __addIndividual__(self, ind: Individual):
        self.population.append(ind)
        self.fitness.append(ind.fitness)
        self.num_inds += 1

    def eval(self):
        self.fitness = [ind.fitness for ind in self.population]

    def __len__(self):
        return self.num_inds

    def __getGenes__(self):
        output = []
        for ind in self.population:
            output.append(ind.genes)
        return output



