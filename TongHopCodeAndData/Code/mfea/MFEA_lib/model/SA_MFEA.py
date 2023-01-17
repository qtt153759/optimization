import numpy as np

from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
import matplotlib.pyplot as plt

class Memory:
    def __init__(self, H=5, sigma=0.1):
        self.H = H
        self.index = 0
        self.sigma = sigma
        self.M = np.zeros((H), dtype=float) + 0.5
    

    def random_Gauss(self):
        mean = np.random.choice(self.M)

        rmp_sampled = 0
        while rmp_sampled <= 0:
            rmp_sampled = mean + self.sigma * np.sqrt(-2.0 * np.log(np.random.uniform())) * np.sin(2.0 * np.pi * np.random.uniform())

        if rmp_sampled > 1:
            return 1
        return rmp_sampled
    
    def update_M(self, value):
        self.M[self.index] = value
        self.index = (self.index + 1) % self.H


class model(AbstractModel.model): 
    def compile(self, 
        IndClass: Type[Individual],
        tasks: list[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)

    def Update_History_Memory(self, history_memories, S, sigma):
        for i in range((len(self.tasks))):
            j = i + 1
            while j < len(self.tasks):
                if len(S[i][j]) != 0:
                    history_memories[i][j].update_M(
                        np.sum(np.array(sigma[i][j]) * np.array(S[i][j]) ** 2)
                        / np.sum(np.array(sigma[i][j]) * (np.array(S[i][j])) + 1e-10)
                    )
                j += 1

        return history_memories
    
    def Linear_population_size_reduction(self, evaluations, current_size_pop, max_eval_each_tasks, max_size, min_size):
        for task in range(len(self.tasks)):
            new_size = (min_size[task] - max_size[task]) * evaluations[task] / max_eval_each_tasks[task] + max_size[task] 

            new_size= int(new_size) 
            if new_size < current_size_pop[task]: 
                current_size_pop[task] = new_size 
        
        return current_size_pop 

    def render_rmp(self): 
        fig = plt.figure(figsize = (50, 12), dpi= 500)
        fig.suptitle("LSA RMP Val\n", size = 15)
        fig.set_facecolor("white")

        for i in range(len(self.tasks)):
            for j in range (len(self.tasks)): 
                plt.subplot(2, int(len(self.tasks)/2), i + 1)
                a = i; 
                b = j
                if i > j: 
                    tmp = a 
                    a = b
                    b = tmp
                plt.plot(np.arange(len(self.history_rmp[a][b])), np.array(self.history_rmp[a][b]), label= 'task: ' +str( j + 1))
                plt.legend()



            plt.title('task ' + str( i+1))
            plt.xlabel("Epoch")
            plt.ylabel("M_rmp")
            plt.ylim(bottom = -0.1, top = 1.1)
        

    def fit(self, max_inds_each_task: list, min_inds_each_task: list, max_eval_each_task: list, H = 30,LSA = False,  bound = [0, 1], evaluate_initial_skillFactor = False,
        *args, **kwargs): 
        super().fit(*args, **kwargs)

        current_inds_each_task = np.copy(max_inds_each_task) 
        eval_each_task = np.zeros_like(max_eval_each_task)

        population = Population(
            self.IndClass,
            nb_inds_tasks= current_inds_each_task, 
            dim = self.dim_uss, 
            list_tasks= self.tasks, 
            evaluate_initial_skillFactor= evaluate_initial_skillFactor
        )

        self.history_rmp = [[[] for i in range(len(self.tasks))] for i in range(len(self.tasks))]

        memory_H = [[Memory(H) for i in range(len(self.tasks))] for j in range(len(self.tasks))]

        while np.sum(eval_each_task) < np.sum(max_eval_each_task):

            S = np.empty((len(self.tasks), len(self.tasks), 0)).tolist() 
            sigma = np.empty((len(self.tasks), len(self.tasks), 0)).tolist()

            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim = self.dim_uss, 
                list_tasks= self.tasks,
            )
            list_generate_rmp = np.empty((len(self.tasks), len(self.tasks), 0)).tolist()
            # create new offspring population 
            while len(offsprings) < len(population): 
                pa, pb = population.__getRandomInds__(size =2) 

                if pa.skill_factor > pb.skill_factor: 
                    pa, pb = pb, pa
                
                

                # crossover 
                if pa.skill_factor == pb.skill_factor: 
                    oa, ob = self.crossover(pa, pb)
                    oa.skill_factor, ob.skill_factor= np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True) 

                else: 
                    # create rmp 
                    rmp = memory_H[pa.skill_factor][pb.skill_factor].random_Gauss()
                    list_generate_rmp[pa.skill_factor][pb.skill_factor].append(rmp) 
                    r = np.random.uniform() 
                    if r < rmp: 
                        oa, ob = self.crossover(pa, pb)
                        oa.skill_factor, ob.skill_factor= np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True) 
                    else: 
                        pa1 = population[pa.skill_factor].__getRandomItems__()
                        while pa1 is pa:
                            pa1 = population[pa.skill_factor].__getRandomItems__()
                        oa, _ = self.crossover(pa, pa1) 
                        oa.skill_factor = pa.skill_factor 

                        pb1 = population[pb.skill_factor].__getRandomItems__()
                        while pb1 is pb:
                            pb1 = population[pb.skill_factor].__getRandomItems__()
                        ob, _ = self.crossover(pb, pb1) 
                        ob.skill_factor = pb.skill_factor 

                    
                
                # append and eval 
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob) 

                # cal delta
                if pa.skill_factor != pb.skill_factor:

                    delta = 0 

                    if oa.skill_factor == pa.skill_factor : 
                        if pa.fcost > 0: 
                            delta = np.max([delta, (pa.fcost - oa.fcost) / (pa.fcost)])
                    else: 
                        if pb.fcost > 0: 
                            delta = np.max([delta, (pb.fcost - oa.fcost) / (pb.fcost)]) 
                    
                    if ob.skill_factor == pa.skill_factor:
                        if pa.fcost > 0: 
                            delta = np.max([delta, (pa.fcost - ob.fcost) / (pa.fcost)])    
                    else: 
                        if pb.fcost > 0: 
                            delta = np.max([delta, (pb.fcost - ob.fcost) / (pb.fcost)]) 
                    

                    # update S and sigma 
                    if delta > 0: 
                        S[pa.skill_factor][pb.skill_factor].append(rmp)
                        sigma[pa.skill_factor][pb.skill_factor].append(delta) 
                
                eval_each_task[oa.skill_factor] += 1 
                eval_each_task[ob.skill_factor] += 1 
            

            # update memory H 
            memory_H = self.Update_History_Memory(memory_H, S, sigma) 

            # linear size 
            if LSA is True: 
                current_inds_each_task = self.Linear_population_size_reduction(eval_each_task, current_inds_each_task, max_eval_each_task, max_inds_each_task, min_inds_each_task) 
            # merge 
            population = population + offsprings 
            population.update_rank()

            # selection 
            self.selection(population, current_inds_each_task)

            # save history 
            
            
            
            # print 
    
            # if (int(eval_each_task[0] / 100) + 1 - len(self.history_cost)) % (num_epochs_printed) == 0:
            #     if log_oneline == True:
            #         sys.stdout.write('\r')
            #     sys.stdout.write('Epoch [{}/{}], [%-20s] %3d%% ,func_val: {}'
            #         .format((int(eval_each_task[0]/100)) + 1, max_eval_each_task[0],self.history_cost[-1])
            #         % ('=' * ((np.sum(eval_each_task)) // (np.sum(max_eval_each_task) // 20)) + '>' , (np.sum(eval_each_task) + 1) * 100 // np.sum(max_eval_each_task))
            #         )
            #     if log_oneline == False:
            #         print("\n")
            #     sys.stderr.flush()
            #     sys.stdout.flush()
            if int(eval_each_task[0] / 100) > len(self.history_cost):
                for i in range(len(self.tasks)):
                    j = i + 1
                    while j < len(self.tasks):
                        if len(list_generate_rmp[i][j]) > 0:
                            self.history_rmp[i][j].append(
                                np.sum(list_generate_rmp[i][j])
                                / len(list_generate_rmp[i][j])
                            )
                        j += 1  

                self.history_cost.append([ind.fcost for ind in population.get_solves()])        
                self.render_process(np.sum(eval_each_task)/ np.sum(max_eval_each_task),["cost"], [self.history_cost[-1]])
    
        print("End")

        # solve 
        self.last_pop = population 

        return self.last_pop.get_solves() 




