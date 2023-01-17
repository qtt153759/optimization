import numpy as np
from .utils.load_data import Load

class Individual:
    def __init__(self, dim:int):
        self.genes = np.zeros(dim, dtype=int)
        self.genes[1:] = np.random.permutation(dim-1)+1
        self.fcost = np.inf
        self.solution = None
    def __getitem__(self, id):
        return self.genes[id]
    def __len__(self):
        return len(self.genes)
    def eval(self, data: Load):
        cost = 0
        curr_collected = np.zeros_like(data.order)
        for i in range(len(self)-1):
            if np.all(curr_collected >= data.order):
                self.solution = np.copy(self.genes[:i+1])
                self.solution = np.append(self.solution,0)
                self.fcost = cost + data.mat_dis[self[i]][0]
                return self.fcost
            # print(self)
            cost += data.mat_dis[self[i]][self[i+1]]
            curr_collected += data.mat_info[:, self[i+1]-1]
        if np.all(curr_collected >= data.order):
            self.solution = np.copy(self.genes)
            self.fcost = cost + data.mat_dis[self[len(self)-1]][0]
            self.solution = np.append(self.solution,0)
        return self.fcost
    def __repr__(self):
        return f"\t+ Genes: {self.genes}\
                \n\t+ Factorial cost: {self.fcost}\
                \n\t+ Path: {self.solution}\n"

class Population:
    def __init__(self, num_individual: int, dim: int, data: Load, eval_initial = False, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.data = data
        self.num_indiv = num_individual
        self.list_indiv = [Individual(dim) for i in range(num_individual)]
        self.cost_pop = np.full(num_individual, np.inf)
        if eval_initial:
            self.eval()

    def __getitem__(self, id):
        return self.list_indiv[id]
    def __len__(self):
        return len(self.list_indiv)
    def __repr__(self):
        return f"{self.list_indiv}"
    def eval(self):
        data = self.data
        for i in range(self.num_indiv):
            if self[i].fcost == np.inf:
                self[i].eval(data)
            self.cost_pop[i] = self[i].fcost
            self.ranking = np.argsort(np.argsort(self.cost_pop))
    def get_best(self) -> Individual:
        return self[np.argmin(self.cost_pop)]
    def get_parents(self, num_indiv:int):
        list_indiv = []
        for i in range(num_indiv):
            list_indiv.append(self[np.random.randint(len(self))])
        return list_indiv
    def elitism_selection(self, num_indiv):
        new_pop = []
        pos_elite = np.where(self.ranking < num_indiv)[0]
        for i in pos_elite:
            new_pop.append(self[i])
        self.list_indiv.clear()
        self.list_indiv = new_pop
    def __addIndividual__(self, indiv: Individual):
        self.list_indiv.append(indiv)
    def __getRandomInds__(self, size: int = None):
        if size == None:
            return self.list_indiv[np.random.randint(len(self))]
        return [self.list_indiv[np.random.randint(len(self))] for i in range(size)]
    def __add__(self, other):
        new_pop = Population(
                num_individual=0,
                dim=self.data.num_bins + 1,
                data=self.data
            )
        new_pop.list_indiv = self.list_indiv + other.list_indiv
        return new_pop
    def update_rank(self):
        self.num_indiv = len(self.list_indiv)
        self.cost_pop = np.full(self.num_indiv, np.inf)
        self.eval()