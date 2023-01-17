from dataclasses import dataclass
import numpy as np
import array
from numpy import Infinity
from ortools.linear_solver import pywraplp
day_off = []

f = open("TestCase/200_60_35_50.txt")
fa = f.readlines()
tmp = fa[0].split()
N = int(tmp[0])
D = int(tmp[1])
a = int(tmp[2])
b = int(tmp[3])
day_off_tmp = np.zeros([N,D], dtype = int)
for i in range(1,len(fa)) :
    tmp = fa[i].split()
    for j in range(len(tmp) - 1) :
        m = int(tmp[j])
        day_off_tmp[i-1,m-1] = 1
for i in range(N) :
    day_off.append(list(day_off_tmp[i]))


def creat_data_model() :
    data = {}
    data['N'] = N
    data['D'] = D
    data['day_off'] = day_off
    data['a'] = a
    data['b'] = b
    return data

def main() :
    global a
    global b
    data = creat_data_model()
    solver = pywraplp.Solver.CreateSolver('SCIP')
    Schedule = {}
    num_night = {}
    for i in range(data['N']) :
        for j in range(4*data['D']) :
            Schedule[i,j] = solver.IntVar(0,1, 'Schedule[' + str(i) + ',' + str(j) + ']')

    for i in range(data['N']) :
        num_night[i] = solver.IntVar(0, data['N'], 'num_night[' + str(i) + ']')

    max_night = solver.IntVar(0, data['D'], "max_night")
    print('Number of variables =', solver.NumVariables())
    for i in range(data['N']) :
        cons1 = solver.Constraint(0, data['D'])
        cons1.SetCoefficient(max_night, 1)
        cons1.SetCoefficient(num_night[i], -1)
    for i in range(data['N']) :
        solver.Add(solver.Sum([Schedule[i,4*j] for j in range(data['D'])]) == num_night[i])
    # for i in range(4*data['D']) :
    #     solver.Add(solver.Sum([Schedule[j,i] for j in range(data['N'])]) == data['a'])
    for i in range(4*data['D']) :
        cons0 = solver.Constraint(data['a'], data['b'])
        for j in range(data['N']) :
            cons0.SetCoefficient(Schedule[j,i], 1)
    for i in range(data['N']) :
        for j in range(data['D']) :
            if j==0 :
                cons1 = solver.Constraint(0, 1)
                cons1.SetCoefficient(Schedule[i,4*j], 1)
                cons1.SetCoefficient(Schedule[i,4*j+1], 1)
                cons1.SetCoefficient(Schedule[i,4*j+2], 1)
                cons1.SetCoefficient(Schedule[i,4*j+3], 1)
            else :
                cons1 = solver.Constraint(0, 1)
                cons1.SetCoefficient(Schedule[i,4*j-1], 1)
                cons1.SetCoefficient(Schedule[i,4*j], 1)
                cons1.SetCoefficient(Schedule[i,4*j+1], 1)
                cons1.SetCoefficient(Schedule[i,4*j+2], 1)
                cons1.SetCoefficient(Schedule[i,4*j+3], 1)
    for i in range(data['N']) :
        for j in range(data['D']) :
            if data['day_off'][i][j] == 1 :
                solver.Add(Schedule[i,4*j] == 0)
                solver.Add(Schedule[i,4*j+1] == 0)
                solver.Add(Schedule[i,4*j+2] == 0)
                solver.Add(Schedule[i,4*j+3] == 0)
    solver.Minimize(max_night)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL :
        print("Solution : ")
        print('Objective value = ', solver.Objective().Value())
        # sol = np.zeros([data['N'], 4*data['D']])
        # for i in range(data['N']) :
        #     for j in range(4*data['D']) :
        #         # print(Schedule[i,j].name(), '=', Schedule[i,j].solution_value() )
        #         sol[i,j] = Schedule[i,j].solution_value()
        # print(sol)

    else :
        print("No solution")
if __name__ == '__main__' :
    main()

