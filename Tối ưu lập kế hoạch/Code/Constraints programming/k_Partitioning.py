"""OR-Tools solution to the N-queens problem."""
import sys
import time
from ortools.sat.python import cp_model
      
def main(k,alpha,input_file,output_file):
    # f = open("../k_way_data.txt", "r")
    # print(f.read())
    f = open(input_file, "r")
    data_inputs = f.read().split("\n")
    sodinh_socach = data_inputs[0].split(' ')
    data_inputs = data_inputs[1:-1]
    a = []
    for i in data_inputs:
      a.append(i.split(" "))
    numVertices = int(sodinh_socach[0])
    numVar1Dim = numVertices + k
    
    # weight_Matrix define from data_input
    weight_Matrix = []
    for i in range(numVertices):
      weight_Matrix.append([])
      for j in range(numVertices):
        weight_Matrix[i].append(0)
    # from scipy.sparse import coo_matrix
    totalWeight=0
    for i in a:
      weight_Matrix[int(i[0])][int(i[1])] = float(i[2])
      totalWeight+=float(i[2])

    
    # print("a",a)
    # print("weight",weight_Matrix)
    # Creates the solver.
    cpModel = cp_model.CpModel()
    x={}
    for i in range(numVertices):
      for j in range(numVertices):
        x[i,j]=(cpModel.NewBoolVar(f'x[{i},{j}]'))

    y={}
    for i in range(numVertices):
      for j in range(k):
        y[i,j]=(cpModel.NewBoolVar(f'y[{i},{j}]'))

    for j in range(k):
      cpModel.AddAtLeastOne(y[i,j] for i in range(numVertices))

    for i in range(numVertices):
      cpModel.AddExactlyOne(y[i,j] for j in range(k))
    # // constraint (1): each vertex belongs to exactly one partition
    #     for (int i=0;i<numVertices;++i){
    #         cpModel.addEquality(LinearExpr.sum(y[i]),1);
    #     }
    for i in range(k-1):
      for j in range(i+1,k):
        b = cpModel.NewBoolVar('b')
        cpModel.AddLinearConstraint(sum(y[t,i] for t in range(numVertices))-sum(y[t,j] for t in range(numVertices)),-alpha,alpha)


    # // constraint (2): boundary of volume of each partition
    #     for (int i = 0; i < k-1; ++i) {
    #         for (int j=i+1;j<k;++j){
    #             cpModel.addGreaterOrEqualWithOffset(LinearExpr.sum(yTranspose[i]),LinearExpr.sum(yTranspose[j]),alpha);
    #             cpModel.addGreaterOrEqualWithOffset(LinearExpr.sum(yTranspose[j]),LinearExpr.sum(yTranspose[i]),alpha);
    #         }
    #     }
    # Contrain 3 if x[i][j]=1 => y[i][l]-y[j][l]
    for i in range(numVertices-1):
      for j in range(i+1,numVertices):
        b = cpModel.NewBoolVar('b')
        cpModel.Add(x[i,j] == 1).OnlyEnforceIf(b)
        cpModel.Add(x[i,j] == 0).OnlyEnforceIf(b.Not())
        for l in range(k):
          cpModel.Add(y[j,l]-y[i,l]==0).OnlyEnforceIf(b)
          # cpModel.Add(y[j][l]-y[i][l] == 0).OnlyEnforceIf(b.Not())

    # // constraint (3): the consistency of x(i,j)
    #     for (int i=0;i<numVertices-1;i++){
    #         for (int j=i+1;j<numVertices;j++){
    #             for(int l=0;l<k;l++){
    #                 cpModel.addLessOrEqual(LinearExpr.
    #                         scalProd(new IntVar[]{x[i][j],y[j][l],y[i][l]},new int[]{1,1,-1}),1);
    #                 cpModel.addLessOrEqual(LinearExpr.
    #                         scalProd(new IntVar[]{x[i][j],y[i][l],y[j][l]},new int[]{1,1,-1}),1);
    #             }
    #         }
    #     }
    objective_terms = []
    for i in range(numVertices):
      for j in range(numVertices):
        objective_terms.append(x[i,j] * weight_Matrix[i][j])
    cpModel.Maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60*10
    # solution_printer = KPartitioningPrinter(y,x)
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(cpModel)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
      print(f'Maximum of objective function: {solver.ObjectiveValue()}\n')
      for i in range(numVertices):
        for j in range(numVertices):
          print(solver.Value(x[i,j]),end=" ")
        print()
    else:
      print('No solution found.')
    
      

    print('\nStatistics')
    print(f'  status   : {solver.StatusName(status)}')
    print(f'  conflicts: {solver.NumConflicts()}')
    print(f'  branches : {solver.NumBranches()}')
    print(f'  wall time: {solver.WallTime()} s')
    with open(output_file, "a") as f:
            f.write("k: %d\n" % k)
            f.write("alpha: %d\n" % alpha)
            f.write("total: %d\n" % totalWeight)
            f.write("weight =: %f\n" % (totalWeight- solver.ObjectiveValue()))
            f.write("timeElapsed : %f milliseconds \n" % (solver.WallTime()*1000))
            f.write("Violation: %d\n"  % 0)
   
if __name__ == '__main__':
    # By default, solve the 8x8 problem.
    main(6,10,"./datas/huge/dense_400.txt","./datas/huge/dense_400_constraint_output.txt")