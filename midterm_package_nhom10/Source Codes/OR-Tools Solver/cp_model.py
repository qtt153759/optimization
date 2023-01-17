import time
from ortools.sat.python import cp_model
C = 1e6
# file_name = "./data/data_20_15.txt"
# fn = "data_20_15"
# file_name = "./data/" + fn + ".txt"
# output = "./result/" + fn + ".txt"
fn = "case_1"
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
    model = cp_model.CpModel()
    x = {}
    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                x[i, j] = model.NewIntVar(0, 1, 'x('+str(i)+','+str(j)+')')

    y = [model.NewIntVar(0, 1, 'y('+str(i)+')') for i in range(0, M+2)]
    r = [model.NewIntVar(0, M+1, 'r('+str(i)+')') for i in range(0, M+2)]
    return x, y, r, model


def Solve(M, N, Q, q, d, total):
    x, y, r, model = create_variables(M)
    model.Add(y[0] == 1)
    model.Add(y[M+1] == 1)
    model.Add(sum(x[i, M+1] for i in range(0,M+1)) == 1)
    model.Add(sum(x[0, i] for i in range(1,M+2)) == 1)
    for i in range(1,M+1):
        model.Add(sum(x[i, j] for j in range(0,i)) + sum(x[i, j] for j in range(i+1,M+2))== sum(x[j, i] for j in range(0,i)) + sum(x[j, i] for j in range(i+1,M+2)))
    for i in range(0, M+2):
        model.Add(sum(x[i, j] for j in range(0,i)) + sum(x[i, j] for j in range(i+1,M+2)) == y[i])

    for i in range(0, M+2):
        for j in range(0, M+2):
            if i != j:
                model.Add(x[i,j]*M + r[i] - r[j] <= M-1)

    for i in range(0, N):
        model.Add(sum(y[j % (M+1)]* Q[i][j % (M+1)] for j in range(1,M+2)) >= q[i])
    
    model.Minimize(sum(x[i, j]*d[i % (M+1)][j % (M+1)] for (i, j) in x))
    print('####')
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        print('S_min = ', int(solver.ObjectiveValue()))
        # print("z")
        # for i in range(1, Nodes+1):
        #     if solver.Value(z[i]) == 1:
        #         print(i)
        # print("x")
        # for (i, j) in d:
        #     if solver.Value(x[i, j]) == 1:
        #         print(i, j)
        # print("y")
        # for i in range(1, Nodes+1):
        #     if solver.Value(y[i]) >= 1:
        #         print(i,solver.Value(y[i]))
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
