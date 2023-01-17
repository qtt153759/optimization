import time
from ortools.linear_solver import pywraplp
C = 1e6
# file_name = "./data/data_20_15.txt"
# fn = "data_20_15"
# file_name = "./data/" + fn + ".txt"
# output = "./result/" + fn + ".txt"
fn = "case_3"
file_name = "./6_26_new_data/" + fn + ".txt"
output = "./result/" + fn + ".txt"
# read data


def read_data(file_name):
    with open(file_name, 'r') as f:
        # add N,M
        [N, M] = [int(x) for x in f.readline().split()]

        # add Q
        Q = []
        for i in range(0, N):
            Q_ = [int(x) for x in f.readline().split()]
            Q_.insert(0, 0)
            Q.append(Q_)

        # add d
        d = []
        for i in range(0, M+1):
            d_ = [int(x) for x in f.readline().split()]
            d.append(d_)
        q = [int(x) for x in f.readline().split()]
    return N, M, Q, d, q


def create_variables(M):
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = {}
    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                x[i, j] = solver.IntVar(0, 1, 'x('+str(i)+','+str(j)+')')

    y = [solver.IntVar(0, 1, 'y('+str(i)+')') for i in range(0, M+2)]
    r = [solver.IntVar(0, M+1, 'r('+str(i)+')') for i in range(0, M+2)]
    return x, y, r, solver


def create_constraint_1(solver, M, x, y):
    #y[0] == 1
    ct = solver.Constraint(1, 1)
    ct.SetCoefficient(y[0], 1)
    #y[M+1] == 1
    ct = solver.Constraint(1, 1)
    ct.SetCoefficient(y[M+1], 1)

    ct = solver.Constraint(1, 1)
    for i in range(1, M+1):
        ct.SetCoefficient(x[i, M+1], 1)

    ct = solver.Constraint(1, 1)
    for i in range(1, M+1):
        ct.SetCoefficient(x[0, i], 1)
    return


def create_constraint_2(solver, M, x, y):
    for i in range(1, M+1):
        ct = solver.Constraint(0, 0)
        for j in range(0, M+2):
            if i != j:
                ct.SetCoefficient(x[i, j], -1)
                ct.SetCoefficient(x[j, i], 1)
    return


def create_constraint_3(solver, M, x, y):
    for i in range(0, M+1):
        ct = solver.Constraint(0, 0)
        ct.SetCoefficient(y[i],-1)
        for j in range(0, M+2):
            if i != j:
                ct.SetCoefficient(x[i, j], 1)
    return

def create_constraint_4(solver, M, x, y,r):
    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                solver.Add(x[i,j]*M + r[i] - r[j] <= M-1)
    return


def create_constraint_5(solver, M, N, y, Q, q, total):
    for i in range(0, N):
        ct = solver.Constraint(q[i], total[i])
        for j in range(1, M+2):
            ct.SetCoefficient(y[j % (M+1)], Q[i][j % (M+1)])
    return


def creat_objective(solver, M, x, d):
    objective = solver.Objective()
    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                objective.SetCoefficient(x[i, j], d[i % (M+1)][j % (M+1)])

    objective.SetMinimization()
    return


def Trace(M, rs):
    trace_ = int(0)
    trace = [0]
    S = 0
    with open(output, 'w') as wf:
        wf.write(f'{0} ')
        while True:
            for i in range(0, M+2):
                if i != trace_ and rs[trace_][i] > 0:
                    S += d[trace_][i%(M+1)]
                    trace_ = i
                    wf.write(f'{trace_} ')
                    break 
            if trace_ == M+1:
                break
            trace.append(trace_)
        wf.write(f'{0}\n')
        wf.write(f'{S}\n')
        trace.append(0)
        print('S_min = ', S)
    return trace


def Solve(M, N, Q, q, d, total):
    x, y,r, solver = create_variables(M)
    create_constraint_1(solver, M, x, y)
    create_constraint_2(solver, M, x, y)
    create_constraint_3(solver, M, x, y)
    create_constraint_4(solver, M, x, y,r)
    create_constraint_5(solver, M, N, y, Q, q, total)
    creat_objective(solver,M,x,d)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # print('S_min = ', int(solver.Objective().Value()))
        rs = [[0 for i in range(0, M+2)] for i in range(0, M+2)]
        for i in range(0, M+2):
            for j in range(0, M+2):
                if i != j:
                    rs[i][j] = x[i, j].solution_value()
        #         print(int(rs[i][j]),end="      ")
        #     print("")      
        # for i in range(0, M+2):
        #     print(int(y[i].solution_value()),"      ", int(r[i].solution_value()))
        print(Trace(M, rs))
    else:
        print("...")
    return


if __name__ == "__main__":
    print("start")
    N, M, Q, d, q = read_data(file_name)
    total = [0 for i in range(0, N)]
    for i in range(0, N):
        for j in range(0, M+1):
            total[i] = total[i] + Q[i][j]
    time1 = time.time()
    Solve(M, N, Q, q, d, total)
    time2 = time.time()
    t = time2-time1
    with open(output, 'a') as wf:
        wf.write(f'{t}')
    print("Time: ", time2 - time1)
    print("end")
