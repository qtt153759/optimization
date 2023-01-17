from audioop import cross
from library.EA import *
from library.utils.operator.selection import ElitismSelection
from library.utils.operator.crossover import OX_Crossover,Two_Cut_Crossover
from library.utils.operator.mutation import SwapMutation
from library.model.ga import model
from library.utils.load_data import Load

from IPython.display import display
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--pop_size", default=1000, type=int)
    parser.add_argument("--prob_m", default=0.2, type=float)
    parser.add_argument("--prob_c", default=0.7, type=float)
    parser.add_argument("--num_run", default=1, type=int )
    parser.add_argument("--max_render", default=5, type=int)
    parser.add_argument("--ox_crossover", action='store_true')
    parser.add_argument("--random_seed", action='store_true')




    param = parser.parse_args()
    print("\nINFO - DATA:")
    data = Load()
    data(param.dataset)
    print(data)
    print(f"\nGENETIC ALGORITHM: \n\t+ Population: {param.pop_size} individuals\n\t+ Number generations: {param.num_epoch} \
        \n\t+ Probability crossover: {param.prob_c}\n\t+ Probability mutation: {param.prob_m}")

    crossover = Two_Cut_Crossover()
    if param.ox_crossover:
        crossover = OX_Crossover()
    print(f"\t+ Crossover: {type(crossover).__name__}")
    sum_cost = 0
    sum_time = 0
    print("\nLOADING...")
    for num_run in range(param.num_run):
        time_begin = time.time()
        if param.random_seed:
            seed=None
        else:
            seed = num_run
        ga_model = model(seed=seed)

        ga_model.compile(
            data_loc=param.dataset,
            crossover=crossover,
            mutation=SwapMutation(),
            selection=ElitismSelection()
        )
        solution = ga_model.fit(
        num_generations=param.num_epoch,
        num_individuals=param.pop_size,
        prob_crossover=param.prob_c,
        prob_mutation=param.prob_m,
        max_render=param.max_render
        )
        print(f"\nComplete: {num_run+1}/{param.num_run}!\n")
        # print(solution)

        sum_time += round(time.time() - time_begin, 2)
        sum_cost += solution.fcost
        if (num_run == 0):
            history_cost = ga_model.res
            best_solution = solution
            worst_solution = solution
        else:
            if (solution.fcost < best_solution.fcost):
                history_cost = ga_model.res
                best_solution = solution
            elif (solution.fcost > worst_solution.fcost):
                worst_solution = solution


    print("-"*100)
    print(f"\nResult GA - {param.num_epoch} epoch - {param.pop_size} individuals, after {param.num_run} times run:")
    print(f"  - Best solution:\n {best_solution}")
    print(f"  - Worst solution:\n {worst_solution}")
    print(f"  - Average of cost: {sum_cost/param.num_run}")
    seconds = sum_time/param.num_run

    minutes = seconds // 60 
    seconds = seconds - minutes * 60 
    display("  - Average of time: %02dm %2.02fs "%(minutes, seconds))

    plt.plot(history_cost)
    plt.title(f"Convergence process - {type(crossover).__name__}")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()