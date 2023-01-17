with open ("./input_10_2_4.txt") as f:
    text = f.readlines()

n = int(text[0][:-1])

d = ("1 " + text[1][:-1]).split(" ")

d = [int(a) for a in d]

print(d)

m = int(text[2][:-1])

c = ("1 " + text[3][:-1]).split(" ")

print(c)

c = [int(a) for a in c]

s=[]

for i in range(5, len(text)):
    tmp = text[i].split(' ')
    tmp = [int(a) for a in tmp]
    s.append(tmp)

from ortools.sat.python import cp_model

left = 1
right = n
ans = n

data = {}
data['n'] = n
data['m'] = m

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        if self.__solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count

total_run_time = 10

while (left <= right):
    k = int((left + right) / 2)

    print("k=" + str(k))
    
    model = cp_model.CpModel()

    p = {}

    for i in range(1, data['n'] + 1):
        for j in range(1, data['n'] + 1):
            p[str(i) + "," + str(j)] = model.NewIntVar(0, 1, "p[" + str(i) + "," + str(j) + "]")

    x = {}

    for i in range(1, data['n'] + 1):
        x[str(i)] = model.NewIntVar(1, k, "x[" + str(i) + "]")

    mm = {}

    for i in range(1, data['n'] + 1):
        for j in range(1, data['m'] + 1):
            mm[str(i) + "," + str(j)] = model.NewIntVar(0, 1, "mm[" + str(i) + "," + str(j) + "]")

    for i in range(1, data['n'] + 1):
        exp = p[str(i) + "," + str(1)]*1
        for j in range(2, data['n'] + 1):
            exp = exp + p[str(i) + "," + str(j)]*j
        model.Add(exp == x[str(i)])

        exp = p[str(i) + "," + str(1)]
        for j in range(2, data['n'] + 1):
            exp = exp + p[str(i) + "," + str(j)]
        model.Add(exp == 1)

    for t in s:
        for l in range(1, data['n'] + 1):
            model.Add(p[str(t[0]) + "," + str(l)] + p[str(t[1]) + "," + str(l)] <= 1)

    for i in range(1, data['n'] + 1):
        exp = mm[str(i) + "," + str(1)]*c[1]
        for j in range(2, data['m'] + 1):
            exp = exp + mm[str(i) + "," + str(j)]*c[j]
        model.Add(exp >= d[i])

    tmp = {}
    for xx in range(1, data['n'] + 1):
        for yy in range(1, data['m'] + 1):
            tmp[str(xx) + "," + str(yy)] = {}
            for t in range(1, data['n'] + 1):
                tmp[str(xx) + "," + str(yy)][str(t)] = model.NewIntVar(0, 1, "tmp["+ str(xx) + "," + str(yy) + "][" + str(t) + "]")
            exp = tmp[str(xx) + "," + str(yy)][str(1)]
            for t in range(2, data['n'] + 1):
                exp = exp + tmp[str(xx) + "," + str(yy)][str(t)]
            model.Add(exp <= 1)
            for t in range(1, data['n'] + 1):
                model.AddMultiplicationEquality(tmp[str(xx) + "," + str(yy)][str(t)], [mm[str(t) + "," + str(yy)],p[str(t) + "," + str(xx)]])
            
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinterWithLimit(1)
    solver.parameters.max_time_in_seconds = int(total_run_time*0.3)
    total_run_time = int(total_run_time*0.7)
    status = solver.Solve(model, solution_printer)
    print("Status=", status)
    if (status == cp_model.FEASIBLE or status == cp_model.OPTIMAL) :
        ans = k
        right = k - 1
    else:
        left = k + 1

print(ans)

print(cp_model.FEASIBLE, cp_model.OPTIMAL, cp_model.UNKNOWN, cp_model.INFEASIBLE)