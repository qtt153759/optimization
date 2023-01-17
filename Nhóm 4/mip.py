from ortools.linear_solver import pywraplp
import time

f=open('data/N4-K2.txt', 'r')

[n, K] = [int(x) for x in f.readline().split()]
q = [int(x) for x in f.readline().split()]
c=[[] for i in range(2*n+1)]
for i in range(2*n+1):
    c[i]=[int(x) for x in f.readline().split()]

#tạo biến:
solver = pywraplp.Solver.CreateSolver('SCIP')
x = {}
for i in range(2*n+1):
    for j in range(2*n+1):
        if i != j:
            for k in range(K):
                x[i, j, k] = solver.IntVar(0, 1, 'x(' + str(i) + ',' + str(j) + ','+ str(k) +')')
INF = solver.infinity()

y = {}
for i in range(2*n+1):
    for k in range(K):
        y[i, k] = solver.IntVar(0, INF, 'y('+str(i)+','+str(k)+')')

sl = {}
for i in range(2*n+1):
    for k in range(K):
        sl[i, k] = solver.IntVar(0, INF, 'sl('+str(i)+','+str(k)+')')


# 
# moi xe k xuat phat tu 0 den i va i+n tro ve 0
for k in range(K):
    cstr = solver.Constraint(1, 1)
    for j in range(1,n+1):
        if j!=0:
            cstr.SetCoefficient(x[0, j, k], 1) 

    # cstr = solver.Constraint(0, 0)
    # for j in range(n+1,2*n+1):# cac diem sau n+1 khong the di tu 0
    #     if j!=0:
    #         cstr.SetCoefficient(x[0, j, k], 1) 

    cstr = solver.Constraint(1, 1)
    for i in range(n+1, 2*n+1):
        if i!=0:
            cstr.SetCoefficient(x[i, 0, k], 1)
    # cstr = solver.Constraint(0, 0)
    # for j in range(1,n+1):# cac diem truoc n+1 khong the ve 0
    #     if j!=0:
    #         cstr.SetCoefficient(x[j, 0, k], 1) 

# # ràng buộc luồng           
for j in range(1,2*n+1):
    if j!=0: #diem 0 co k luong 
        cstr = solver.Constraint(1, 1)
        for i in range(2*n+1):
            if i != j :
                for k in range(K):
                    cstr.SetCoefficient(x[j, i, k], 1)
                
        cstr = solver.Constraint(1, 1)
        for i in range(2*n+1):
            if i != j:
                for k in range(K):
                    cstr.SetCoefficient(x[i, j, k], 1)


### bat buoc i va i +n di chung 1 xe
for i in range(1, n+1):
    for k in range(K):
        cstr = solver.Constraint(0, 0)
        for j in range(2*n+1):
            for l in range(2*n+1):
                if j!=i and l!=i+n:
                    cstr.SetCoefficient(x[j, i, k], 1)
                    cstr.SetCoefficient(x[l, i+n, k], -1)    
# di vao 1 diem va ra tai 1 diem phai cung 1 xe
for i in range(2*n+1):
    for k in range(K):
        cstr = solver.Constraint(0, 0)
        for j in range(2*n+1):
            for l in range(2*n+1):
                if j!=i and l!=i:
                    cstr.SetCoefficient(x[j, i, k], 1)
                    cstr.SetCoefficient(x[i, l, k], -1)

##lien he x va y
for k in range(K):
    cstr = solver.Constraint(0, 0)
    cstr.SetCoefficient(y[0, k], 1)
for j in range(1,2*n+1):
    for i in range(0,2*n+1):
        if i != j:
            for k in range(K):
                cstr = solver.Constraint(-10000-c[i][j], INF)
                #cstr.SetCoefficient(y[i, k]+c[i][j]-y[j, k]+10000(1-x[i, j, k]), 1)>0
                cstr.SetCoefficient(y[i, k], 1)
                cstr.SetCoefficient(y[j, k], -1)
                cstr.SetCoefficient(x[i, j, k], -10000)

                
                cst = solver.Constraint(-INF, 10000-c[i][j])

                #cstr.SetCoefficient(y[i, k]+c[i][j]-y[j, k]+10000(x[i, j, k]-1), 1)<0
                cst.SetCoefficient(y[i, k], 1)
                cst.SetCoefficient(y[j, k], -1)
                cst.SetCoefficient(x[i, j, k], +10000)
#đi đến i đi trước i+n
for i in range(1, n+1):
    for k in range(K):
        cstr = solver.Constraint(-INF, -1)
        cstr.SetCoefficient(y[i, k],1)
        cstr.SetCoefficient(y[i+n, k],-1)
#chi phí đến mỗi đỉnh đều dương => từ 0 có thể đi hết qua các đỉnh
for i in range(1, 2*n+1):
    for k in range(K):
        cstr = solver.Constraint(1, INF)
        cstr.SetCoefficient(y[i, k],1)


#khoi tao số lượng khách mỗi xe
for k in range(K):
    cstr = solver.Constraint(0, 0)
    cstr.SetCoefficient(sl[0, k], 1)

# Mối quan hệ số lượng khách nếu có đường đi i đến j
for j in range(1,n+1):
    for i in range(2*n+1):
        if i != j:
            for k in range(K):
                cstr = solver.Constraint(-10000-1, INF)
                #sl[i, k]+1-sl[j, k]+10000(1-x[i, j, k])>0
                cstr.SetCoefficient(sl[i, k], 1)
                cstr.SetCoefficient(sl[j, k], -1)
                cstr.SetCoefficient(x[i, j, k], -10000)

                cst = solver.Constraint(-INF, 10000-1)
                #sl[i, k]+1-sl[j, k]+10000(x[i, j, k]-1)<0
                cst.SetCoefficient(sl[i, k], 1)
                cst.SetCoefficient(sl[j, k], -1)
                cst.SetCoefficient(x[i, j, k], +10000)

for j in range(n+1, 2*n+1):
    for i in range(2*n+1):
        if i != j:
            for k in range(K):
                cstr = solver.Constraint(-10000+1, INF)
                #sl[i, k]-1-sl[j, k]+10000(1-x[i, j, k])>0
                cstr.SetCoefficient(sl[i, k], 1)
                cstr.SetCoefficient(sl[j, k], -1)
                cstr.SetCoefficient(x[i, j, k], -10000)

                cst = solver.Constraint(-INF, 10000+1)
                #sl[i, k]-1-sl[j, k]+10000(x[i, j, k]-1)<0
                cst.SetCoefficient(sl[i, k], 1)
                cst.SetCoefficient(sl[j, k], -1)
                cst.SetCoefficient(x[i, j, k], +10000)
# số lượng khách xe k quá quá q[k]
for i in range(1, n+1):
    for k in range(K):
        cstr = solver.Constraint(0, q[k])
        cstr.SetCoefficient(sl[i, k],1)
 

obj = solver.Objective()
for i in range(2*n+1):
    for j in range(2*n+1):
        if i!=j:
            for k in range(K):
                obj.SetCoefficient(x[i, j , k], c[i][j])

       
obj.SetMinimization()

start = time.time()
result_status = solver.Solve()
end = time.time()

def printRes():
    print("Distance min=", obj.Value())
    print("Route:")
    res = [[] for i in range(K)]
    for p in range(K):
        for i in range(2*n+1):
            for j in range(2*n+1):
                if i != j and x[i, j, p].solution_value() > 0:
                    res[p].append((i, j))

    for i in range(len(res)):
        h = []
        h.append(res[i][0][0])
        h.append(res[i][0][1])
        m = res[i][1:]
        while len(m)>0:
            for j in range(len(m)):
                if m[j][0] == h[-1]:
                    h.append(m[j][1])
                    m.pop(j)
                    break
        print("Vehicle "+str(i+1)+":")
        print(*h, sep=" -> ")
    print("Time:", end - start)
            
printRes()