import ortools
from ortools.linear_solver import pywraplp
import argparse
import time

def create_data(k, alpha, data_file):#input_file

    f = open(data_file, "r")
    data_inputs = f.read().split("\n")

    sodinh_socach = data_inputs[0].split(' ')
    data_inputs = data_inputs[1:-1]

    canhs = []
    for i in data_inputs:
        canhs.append(i.split(" "))

    numVertices = int(sodinh_socach[0])
    numVar1Dim = numVertices + k
    
    # weight_Matrix define from data_input
    weight_Matrix = []
    for i in range(numVertices):
        weight_Matrix.append([])
        for j in range(numVertices):
            weight_Matrix[i].append(0)
    # from scipy.sparse import coo_matrix
    for i in canhs:
        weight_Matrix[int(i[0])][int(i[1])] = float(i[2])

    data = {"k": k, "alpha": alpha, "numVertices": numVertices, "numVar1Dim": numVar1Dim, "weight_canh": weight_Matrix, "name_file": data_file}
    return data




def IP_solver(data):
    numVertices = data["numVertices"]
    numVar1Dim = data["numVar1Dim"]
    alpha = data["alpha"]
    weight_Matrix = data["weight_canh"]

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    x = {}
    for i in range(numVertices):
        for j in range(numVar1Dim):
            x[i, j] = solver.IntVar(0, 1, "x[{}_{}]".format(i, j))


    #  // constraint (1): each vertex belongs to exactly one 
    # // mỗi đỉnh thuộc đúng một phân vùng

    for i in range(numVertices):
        ct1 = solver.RowConstraint(1, 1, "ver "+str(i)+"in one partition")#bang 1
        for j in range(numVertices, numVar1Dim):
            ct1.SetCoefficient(x[i, j],1)#heej soos


    # // constraint (2): boundary of volume of each partition
    # // ranh giới mỗi phân vùng

    for i in range(numVertices, numVar1Dim):
        for j in range(i + 1, numVar1Dim):
            ct2 = solver.RowConstraint(- alpha, alpha, "diff part: "+str(i)+"-"+str(j))
            for ku in range(numVertices):
                ct2.SetCoefficient(x[ku,i],1)
                ct2.SetCoefficient(x[ku,j],-1)

    #  // constraint (3): the consistency of x(i,j)
    # // tính nhất quán

    for i in range(numVertices):
        for j in range(numVertices):
            for ku in range(numVertices, numVar1Dim):
                ct3 = solver.RowConstraint(-1,1,"consistent: "+str(i)+"-"+str(j))
                ct3.SetCoefficient(x[i, j],1)
                ct3.SetCoefficient(x[i, ku],-1)
                ct3.SetCoefficient(x[j, ku],1)

                ct3 = solver.RowConstraint(-1,1,"consistent: "+str(j)+"-"+str(i))
                ct3.SetCoefficient(x[i, j],1)
                ct3.SetCoefficient(x[i, ku],1)
                ct3.SetCoefficient(x[j, ku],-1)

    # // set up objective and calculate total weight of the graph at the same time
    # // thiết lập mục tiêu và tính toán tổng trọng lượng của đồ thị cùng một lúc
    objective = solver.Objective()
    sumWeight = 0
    for i in range(numVertices):
        for j in range(i, numVertices):
            objective.SetCoefficient(x[i, j], weight_Matrix[i][j])
            sumWeight += weight_Matrix[i][j]
    objective.SetMaximization()
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        solution = {}
        solution["k"] = data["k"]
        solution["weight"] = sumWeight
        solution["alpha"] = data["alpha"]
        solution["name"] = data["name_file"]
        solution["obj"] = solver.Objective().Value()
        solution["timeElapsed"] = solver.wall_time()
        solution["iterations"] = solver.iterations()
        solution["nodes"] = solver.nodes()
        solution["violation"] = 0 #vi day la IP nen chi co the co hoac khong

    return solution

def export(output_file, solution):
    if solution is not None:
        f = open(output_file, "r")
        data_inputs = f.read()
        f.close()
        with open(output_file, "w+") as f:
            f.write(data_inputs)
            # a = solution["name"][11:-1]
            # f.write("Name: %s\n" % a)
            f.write("K: %d\n" % solution["k"])
            f.write("Alpha: %d\n" % solution["alpha"])
            f.write("weight: %f\n" % (solution["weight"] - solution["obj"]))
            f.write("Violation: %d\n" % solution["violation"])  
            f.write("timeElapsed: %f milliseconds\n\n" % solution["timeElapsed"])
            # f.write("Result objective value =: %f\n" % float(solution["obj"]))
            # f.write("Iterations: %d\n" % solution["iterations"])
            f.write("sum_W: %f\n" % solution["weight"])

    else:
        with open(output_file, "w+") as f:
            f.write("No solution")


def process(k , alpha, input_file , output_file):
    data = create_data(k, alpha, input_file)
    solution = IP_solver(data)
    export(output_file, solution)

# /Users/duongdong/Desktop/graph_partitioning-main/datas/output/IPSolver/small
# /Users/duongdong/Desktop/graph_partitioning-main/datas/output/IPSolver/small/dense_10.rtf
# /Users/duongdong/Desktop/graph_partitioning-main/datas/output/IPSolver/
if __name__ == "__main__":
    process(4, 5, "datas/huge/dense_100.txt", 'datas/output/IPSolver/huge/dense_100.txt') 



