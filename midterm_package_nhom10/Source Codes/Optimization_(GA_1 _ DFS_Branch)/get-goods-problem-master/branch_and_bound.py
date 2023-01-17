from lib2to3.pgen2.token import MINUS
import numpy as np
import time
import sys
import os

from IPython.display import clear_output
from library.utils.load_data import Load
from argparse import ArgumentParser

def render_process(list_desc, list_value, time_begin):
    seconds = time.time() - time_begin
    minutes = seconds // 60
    seconds = seconds - minutes * 60
    print_line = str("")
    print_line = print_line + "Time: %02dm %2.02fs"%(minutes, seconds) + "\t--> "

    for i in range(len(list_desc)):
        desc = str("")
        desc = desc + f"{list_value[i]}" + "  "
        line = '{}: {}  '.format(list_desc[i], desc)
        print_line = print_line + line 
  
    sys.stdout.flush() 
    sys.stdout.write("\r" + print_line)
    sys.stdout.flush()    


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str)
    param = parser.parse_args()
    
    data = Load()
    data(param.dataset)

    N = data.num_kinds
    M = data.num_bins
    Q = data.mat_info
    d = data.mat_dis
    q = data.order
    X = np.full(M+1, -1)
    cost = 0
    min_cost = np.inf
    solution = np.copy(X)
    update_time = 1
    history_cost = []
    time_begin = time.time()
    print("\n\nINFO - DATA:")
    print(data)
    print("\n")
    print("\nBRANCH & BOUND ALGORITHM:")

    def Try(k):
        global Q, d, q, N, M, X, cost, min_cost, solution, update_time
        for v in range(1, M+1):
            if v not in X[:k]:
                # update the state
                X[k] = v
                cost += d[X[k-1], X[k]]
                q -= Q[:, v-1]

                if (cost + np.min(d) <= min_cost):
                    if np.all(q <= 0) or k == M:
                        if cost + d[X[k],0] < min_cost:
                            min_cost = cost + d[X[k],0]
                            solution = np.copy(X[:k+1])
                            # print(f"Update {update_time}: - Cost: {min_cost}")
                            # print(f"\t   - Path: {np.append(solution,0)}")
                            
                            render_process(["Cost", "Path"], [min_cost, np.append(solution, 0)], time_begin)
                            update_time+=1
                            history_cost.append(min_cost)
                    else:
                        Try(k+1)
                # recover the state
                cost -= d[X[k-1], X[k]]
                q += Q[:, v-1]


    # Run
    X[0] = 0
    time_begin = time.time()
    Try(1)


    
    print("\n\nFINISHED:")
    print(f"\t+ Time taken: {round(time.time() - time_begin, 2)}")
    print(f"\t+ Path: {np.append(solution,0)}")
    print(f"\t+ Cost: {min_cost}")
    

