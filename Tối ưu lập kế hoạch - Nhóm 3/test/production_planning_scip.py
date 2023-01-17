from ortools.linear_solver import pywraplp

def main(N, A, C, c, a, f, m):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    infinity = solver.infinity()

    k = {}
    for i in range(N):
        k[i] = solver.IntVar(m[i], infinity, f'k_{i}')

    solver.Add(sum((k[i]*c[i]) for i in range(N)) <= C)
    solver.Add(sum((k[i]*a[i]) for i in range(N)) <= A)

    solver.Maximize(sum((k[i]*f[i]) for i in range(N)))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        for i in range(N):
            print(k[i].solution_value())
    else:
        print('The problem does not have an optimal solution.')

if __name__ == '__main__':
    N=8
    A=916
    C=478
    c=[ 9, 8, 9, 9, 7, 9, 9, 8]
    a=[ 18, 15, 17, 17, 13, 18, 16, 15]
    f=[ 270, 238, 256, 260, 201, 276, 250, 239]
    m=[ 2, 3, 1, 2, 9, 9, 2, 1]
    
    main(N, A, C, c, a, f, m)