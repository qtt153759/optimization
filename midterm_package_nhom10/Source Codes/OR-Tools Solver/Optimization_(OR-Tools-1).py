import time
from ortools.linear_solver import pywraplp
C = 1e6
# file_name = "./data/data_20_15.txt"
fn = "case_4"
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


def create_variables(M, maxd):
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = {}
    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                x[i, j] = solver.IntVar(0, 1, 'x('+str(i)+','+str(j)+')')

    y = [solver.IntVar(0, 1, 'y('+str(i)+')') for i in range(0, M+2)]
    s = [solver.IntVar(0, maxd, 's('+str(i)+')') for i in range(0, M+2)]
    return x, y, s, solver


def create_constraint_1(solver, M, x, y, s):
    #y[0] == 1
    ct = solver.Constraint(1, 1)
    ct.SetCoefficient(y[0], 1)
    #y[M+1] == 1
    ct = solver.Constraint(1, 1)
    ct.SetCoefficient(y[M+1], 1)
    #z[0] == 0
    ct = solver.Constraint(0, 0)
    ct.SetCoefficient(s[0], 1)

    ct = solver.Constraint(1, 1)
    for i in range(1, M+1):
        ct.SetCoefficient(x[i, M+1], 1)

    ct = solver.Constraint(1, 1)
    for i in range(1, M+1):
        ct.SetCoefficient(x[0, i], 1)
    return


def create_constraint_2(solver, M, x, y):
    for i in range(1, M+1):
        for j in range(1, M+1):
            if i != j:
                solver.Add(y[i]+y[j] + C*(1-x[i, j]) >= 2)
                solver.Add(y[i]+y[j] + C*(x[i, j]-1) <= 2)
    return


def create_constraint_3(solver, M, x, y):
    for i in range(1, M+1):
        ct = solver.Constraint(1-C, C)
        ct.SetCoefficient(y[i], -C)
        for j in range(1, M+2):
            if i != j:
                ct.SetCoefficient(x[i, j], 1)

        ct = solver.Constraint(-C, C+1)
        ct.SetCoefficient(y[i], C)
        for j in range(1, M+2):
            if i != j:
                ct.SetCoefficient(x[i, j], 1)
    return

def create_constraint_4(solver, M, x, y):
    for i in range(1, M+1):
        ct = solver.Constraint(1-C, C)
        ct.SetCoefficient(y[i], -C)
        for j in range(0, M+1):
            if i != j:
                ct.SetCoefficient(x[j, i], 1)

        ct = solver.Constraint(-C, C+1)
        ct.SetCoefficient(y[i], C)
        for j in range(0, M+1):
            if i != j:
                ct.SetCoefficient(x[j, i], 1)
    return

def create_constraint_5(solver, M, x, s, d):
    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                solver.Add(s[j] + C*(1-x[i, j]) >=
                           s[i]+  d[i % (M+1)][j % (M+1)])
                solver.Add(s[j] + C*(x[i, j]-1) <=
                           s[i] + d[i % (M+1)][j % (M+1)])
    return


def create_constraint_6(solver, M, N, y, Q, q, total):
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


def Solve(M, N, Q, q, d, total, maxd):
    x, y, s, solver = create_variables(M, maxd)
    create_constraint_1(solver, M, x, y, s)
    create_constraint_2(solver, M, x, y)
    create_constraint_3(solver, M, x, y)
    create_constraint_4(solver, M, x, y)
    create_constraint_5(solver, M, x, s, d)
    create_constraint_6(solver, M, N, y, Q, q, total)
    creat_objective(solver,M,x,d)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # print('S_min = ', int(solver.Objective().Value()))
        rs = [[0 for i in range(0, M+2)] for i in range(0, M+2)]
        for i in range(0, M+2):
            for j in range(0, M+2):
                if i != j:
                    rs[i][j] = x[i, j].solution_value()
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
    maxd = 0
    for i in range(0, M+1):
        for j in range(i+1, M+1):
            maxd = maxd + d[i][j]
    time1 = time.time()
    Solve(M, N, Q, q, d, total, maxd)
    time2 = time.time()
    t = time2-time1
    with open(output, 'a') as wf:
        wf.write(f'{t}')
    print("Time: ", time2 - time1)
    print("end")
