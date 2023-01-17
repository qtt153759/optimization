import time
from ortools.sat.python import cp_model
import os


def read_input(file):
    with open(os.path.join('data', file), 'r') as f:
        [n, k] = [int(x) for x in f.readline().split()]
        capacity = [int(x) for x in f.readline().split()]
        route = []
        for i in range(2 * n + 1):
            route.append([int(x) for x in f.readline().split()])
    return n, k, capacity, route


n, k, bus_capacity, route = read_input('N8-K2.txt')
m = 10 * n
# print(n)
# print(k)
# print(bus_capacity)
# print(route)

model = cp_model.CpModel()
'''
Variables:
x[k][i][j]: 1 if (bus k travels from point i to point j) else 0
y[i]: capacity of bus passing point i after visited i
z[i]: index of bus visiting point i
b[i][j]: 1 if (point i is visited before point j) else 0  
'''
x = [[[model.NewIntVar(0, 1, 'x[' + str(p) + ', ' + str(i) + ', ' + str(j) + ']')
       for j in range(2 * n + 1)] for i in range(2 * n + 1)] for p in range(k)]
y = [model.NewIntVar(0, max(bus_capacity), 'y['+str(i)+']')
     for i in range(2*n+1)]
z = [model.NewIntVar(0, k, 'z['+str(i)+']') for i in range(2*n+1)]
b = [[model.NewIntVar(0, 1, 'b['+str(i)+']['+str(j)+']')
      for j in range(2*n+1)] for i in range(2*n+1)]

A = []
for i in range(2 * n + 1):
    for j in range(2 * n + 1):
        if i != j and not (i <= n and j == 0) and not(i == 0 and j >= n+1) and not(i == j+n):
            A.append([i, j])
# print(A)


def Ai(x): return [i for i, j in A if j == x]
def Ao(x): return [j for i, j in A if i == x]


for i in range(1, 2 * n + 1):
    '''
    1 bus travel from point i
    '''
    model.Add(sum([x[p][i][j] for p in range(k) for j in Ao(i)]) == 1)

for i in range(1, 2 * n + 1):
    ''' 
    1 bus travel to point i
    '''
    model.Add(sum([x[p][j][i] for p in range(k) for j in Ai(i)]) == 1)

for p in range(k):
    for i in range(2 * n + 1):
        '''
        Same bus pass point i
        '''
        model.Add(sum([x[p][i][j] for j in Ao(i)]) ==
                  sum([x[p][j][i] for j in Ai(i)]))

for p in range(k):
    '''
    All buses are working
    '''
    model.Add(sum([x[p][0][j] for j in range(1, n + 1)]) == 1)
    model.Add(sum([x[p][j + n][0] for j in range(1, n + 1)]) == 1)

for p in range(k):
    for i, j in A:
        '''
        Capacity at any point <= capacity of the bus passing that point
        '''
        temp = model.NewBoolVar('temp')
        model.Add(x[p][i][j] == 1).OnlyEnforceIf(temp)
        model.Add(y[j] <= bus_capacity[p]).OnlyEnforceIf(temp)
        model.Add(x[p][i][j] != 1).OnlyEnforceIf(temp.Not())

for p in range(k):
    for i, j in A:
        '''
        If x[k][i][j] == 1 and 0<i<=n, then: bus's capacity += 1
                                  i>n, then: bus's capacity -= 1
        '''
        if 0 < j <= n:
            temp = model.NewBoolVar('temp')
            model.Add(x[p][i][j] == 1).OnlyEnforceIf(temp)
            model.Add(y[j] == y[i] + 1).OnlyEnforceIf(temp)
            model.Add(x[p][i][j] != 1).OnlyEnforceIf(temp.Not())
        if j > n:
            temp = model.NewBoolVar('temp')
            model.Add(x[p][i][j] == 1).OnlyEnforceIf(temp)
            model.Add(y[j] == y[i] - 1).OnlyEnforceIf(temp)
            model.Add(x[p][i][j] != 1).OnlyEnforceIf(temp.Not())
    for i, j in A:
        if i != 0 and j != 0:
            '''
            If bus k travels from i to j, then the index of bus passing i and j is k
            If x[k][i][j] == 1, then z[i] == z[j] == k
            '''
            temp = model.NewBoolVar('temp')
            model.Add(x[p][i][j] == 1).OnlyEnforceIf(temp)
            model.Add(z[j] == z[i]).OnlyEnforceIf(temp)
            model.Add(z[j] == p).OnlyEnforceIf(temp)
            model.Add(z[i] == p).OnlyEnforceIf(temp)
            model.Add(x[p][i][j] != 1).OnlyEnforceIf(temp.Not())

for i in range(1, n + 1):
    '''
    The index of bus passing point i must pass point i+N
    '''
    model.Add(z[i] == z[i + n])

for p in range(k):
    for i in range(1, 2*n+1):
        for j in range(1, 2*n+1):
            '''
            if bus p travel from i to j(x[p][i][j] == 1): point i is visited before point j
            else (x[p][i][j] == 0): 0 <= b[i][j]
            '''
            model.Add(x[p][i][j] - b[i][j] <= 0)

for i in range(1, n+1):
    '''
    point i must be visited before point i+N

    '''
    model.Add(b[i][i+n] == 1)


# for i in range(1, 2*n+1):
#     '''
#     b[i][i] == 0
#     '''
#     model.Add(b[i][i] == 0)

for i in range(1, n+1):
    '''
    point i+N must not be visited before point i
    '''
    model.Add(b[n + i][i] == 0)

for q in range(k):
    for i in range(1, 2*n+1):
        for j in range(1, 2*n+1):
            if j != i:
                for u in range(1, 2*n+1):
                    if u != i and u != j:
                        '''
                        If bus q travel from point i to point j, 
                            then all points u != i, j are visited before i and j,
                                                   or are visted after i and j
                        '''
                        temp = model.NewBoolVar('temp')
                        model.Add(x[q][i][j] == 1).OnlyEnforceIf(temp)
                        model.Add(b[u][i] == b[u][j]).OnlyEnforceIf(temp)
                        model.Add(b[i][u] == b[j][u]).OnlyEnforceIf(temp)
                        model.Add(b[j][i] == 0).OnlyEnforceIf(temp)
                        model.Add(x[q][i][j] != 1).OnlyEnforceIf(temp.Not())


f = model.NewIntVar(0, 999999*(2*n+1), 'objective func')
model.Add(f == sum([x[p][i][j]*route[i][j] for p in range(k)
          for i in range(2*n+1) for j in range(2*n+1)]))

model.Minimize(f)
solver = cp_model.CpSolver()
start = time.time()
solver.Solve(model)
end = time.time()
print('Can_be_solved =', solver.Solve(model) == cp_model.OPTIMAL)

def findNext(q, i):
    for j in Ao(i):
        if solver.Value(x[q][i][j]) > 0:
            return j


def routes(q):
    s = '0 - '
    i = findNext(q, 0)
    while i != 0:
        s = s + str(i) + ' - '
        i = findNext(q, i)
    s = s + str(i)
    return s

print(f'Optimal objective value = {solver.Value(f)}')
for p in range(k):
    print('route[', p, '] = ', routes(p))
    print('route[',p,'] = ')
    for i, j in A:
        if solver.Value(x[p][i][j]) > 0:
            print('(', i, '-', j, ')', solver.Value(y[j]), 'z =', solver.Value(
                z[i]), solver.Value(z[j]), 'distance =', route[i][j])
            # print('(',i,'-',j,')', y[j].solution_value())
print(end-start)
