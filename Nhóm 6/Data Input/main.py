import os
import logging
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time

class TimeCounter():
  def __init__(self):
    self.start_time = 0
    self.end_time = 0
  def start(self):
    self.start_time = time.time()
  def end(self):
    self.end_time = time.time()
  def get_duration(self):
    return self.end_time - self.start_time

class Experiment_Manager():
  def read_data(self, file_name):
    #DATA_FILE = './Data Input/2005PisingerSigurd1.bpp'
    rootLogger.info("Read data from: {:s}".format(file_name))
    with open(file_name,'r') as f:
      lines = f.readlines()
      N,K = [int(i) for i in lines[1].strip().split()[:2]]
      W_, H_ = [],[]
      h,w = [],[]
      c = []
      bins = lines[3].strip().split()
      costs = lines[4].strip().split()
      for i in range(K):
        H_.append(int(bins[2*i]))
        W_.append(int(bins[2*i+1]))
        c.append(int(costs[i]))
      for i in range(N):
        item = lines[i+5].strip().split()
        h.append(int(item[0]))
        w.append(int(item[1]))
    W, H = max(W_), max(H_)
    data = {'bin': {'h': H_, 'w': W_, 'c': c}}
    for i in range(N):
      data[f'cat{i}'] = {'w': w[i],'h':h[i],'items':N_ITEMS}
    return data
  def run_MIPS(self, data):
    rootLogger.info("-----------Mixed Integer Programming------------")
    ### MIPS parameters initialize
    h = [data[cat]['h'] for cat in data if cat!='bin' for i in range(data[cat]['items'])]
    w = [data[cat]['w'] for cat in data if cat!='bin' for i in range(data[cat]['items'])]
    W_ = data['bin']['w']
    H_ = data['bin']['h']
    N = len(w)
    K = len(W_)
    W = max(W_)
    H = max(H_)
    c = data['bin']['c']

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.SetTimeLimit(TIME_LIMIT*1000)

    # Initialize variables
    #Variable

    infinity = solver.infinity()

    # Bien xi, yi low-left.
    x = {}
    y = {}
    z = {}
    for i in range(N):
        x[i] = solver.IntVar(0, infinity,f'x[{i}]') 
    for i in range(N):
        y[i] = solver.IntVar(0, infinity,f'y[{i}]')
    #Decision variable for using bin/ trunk
    for k in range(K):
        z[k] = solver.IntVar(0, 1, f'z[{k}]')

    # biến l_ij, b_ij, f_ik, 
    l ={}
    for i in range(N):
        for j in range(N):
            if (i != j):
              l[i,j] = solver.IntVar(0, 1, f'l[{i},{j}]')
    b = {}
    for i in range(N):
        for j in range(N):
            if (i != j):
                b[i,j] = solver.IntVar(0, 1, f'b[{i},{j}]')

    f = {}
    for i in range(N):
        for k in range(K):
            f[i,k] = solver.IntVar(0, 1, f'f[{i},{k}]')

    rootLogger.info('Number of variables = {:d}'.format(solver.NumVariables()))

    ### Initialize constraints
    #Define constraint
    # Non overlab constraint
    for i in range(N):
        for j in range(i+1, N):
            for k in range(K):
                solver.Add(l[i,j]+l[j,i]+b[i,j]+b[j,i]+(1-f[i,k])+(1-f[j,k])>=1)

    # neu i nằm bên trái j và  thì l_ij = 1 thif xi + wi <= xj ( Và có sự ngầm hiểu là i, j cùng bin)
    # Nếu i, j ko cùng bin thì lij  = 0, và vẫn đúng vì W là max with của tất cả các bin
    for i in range(N):
        for j in range(N):
            if (i !=j):
                solver.Add(x[i]-x[j]+W*l[i,j]<=W-w[i])
    for i in range(N):
        for j in range(N):
            if (i !=j):
                solver.Add(y[i]-y[j]+H*b[i,j]<=H-h[i])

    # Ko item nào có kích cỡ vượt quá kích cỡ của bin cả?.
    for i in range(N):
        for k in range(K):
            solver.Add(x[i]<=W_[k]-w[i]+(1-f[i,k])*W)
            solver.Add(y[i]<=H_[k]-h[i]+(1-f[i,k])*H)     
    # Constraint mỗi item thì phải thuộc đúng 1 chiếc xe tải?.
    for i in range(N):
            solver.Add(solver.Sum(f[i,k] for k in range(K)) == 1) #???. >= 1 hay == 1?

    # Điều kiện cuối cùng lq đến biến của hàm mục tiêu la biến Z_k
    for i in range(N):
        for k in range(K):
            solver.Add(z[k]>=f[i,k])
    rootLogger.info('Number of constraints = {:d}'.format(solver.NumConstraints()))

    ### Define objective function
    solver.Minimize(sum(z[k]*c[k] for k in range(K)))

    ### Solve
    COUNTER.start()
    status = solver.Solve()
    COUNTER.end()

    ### Display and log results
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        rootLogger.info("Solved in {:.2f}s".format(COUNTER.get_duration()))
        rootLogger.info('Objective value = {:d}'.format(int(solver.Objective().Value())))
        if status == 0:
          rootLogger.info("OPTIMAL SOLUTION")
        else:
          rootLogger.info("FEASIBLE SOLUTION")
      # for k in range(K):
      #     print(z[k].solution_value())
        inBin = []
        for i in range(N):
            for k in range(K):
                if (f[i,k].solution_value() == 1):
                    inBin.append(k)
                    #print('item',i,k)
        rootLogger.info(inBin)

  def run_CP(self, data):
    rootLogger.info("-----------Constraint Programming------------")
    # bin width and height
    H = data['bin']['h']
    C = data['bin']['c']
    #W = data['bin']['w']
    W = [0]
    for i in range(len(data['bin']['h'])):
      W.append(W[i]+data['bin']['w'][i])
    # h,w,cat for each item
    h = [data[cat]['h'] for cat in data if cat!='bin' for i in range(data[cat]['items'])]
    w = [data[cat]['w'] for cat in data if cat!='bin' for i in range(data[cat]['items'])]
    cat = [cat for cat in data if cat!='bin' for i in range(data[cat]['items'])]
    n = len(h)  # number of items
    m = len(data['bin']['h'])      # number of bins

    #---------------------------------------------------
    # or-tools model 
    #---------------------------------------------------


    model = cp_model.CpModel()

    #
    # variables
    #

    # x1,x2 and y1,y2 are start and end
    x1 = [model.NewIntVar(0, W[-1]-w[i], f'x1[{i}]') for i in range(n)]
    x2 = [model.NewIntVar(w[i], W[-1], f'x2[{i}]') for i in range(n)]
    y1 = [model.NewIntVar(0, max(H) - h[i], f'y1[{i}]') for i in range(n)]
    y2 = [model.NewIntVar(h[i], max(H), f'y2[{i}]') for i in range(n)]

    # interval variables
    lit = [[model.NewBoolVar(f'lit[{i}][{j}]') for j in range(m)] for i in range(n)]
    # Box using
    u = [model.NewBoolVar(f'u[{i}]') for i in range(m)]


    #
    # constraints
    #
    #for i in range(n):
    #  #model.Add(x1[i] == x[i] + b[i]*W)
    #  model.Add(x2[i] == x1[i] + w[i])
    #  model.Add(y2[i] == y1[i] + h[i])
    for i in range(n):
      model.Add(sum(lit[i]) == 1)
      model.Add(sum(W[j] * lit[i][j] for j in range(m)) <= x1[i])
      model.Add(sum(W[j + 1] * lit[i][j] for j in range(m)) >= x2[i])
      model.Add(x1[i]+w[i] == x2[i])
      #model.Add(sum(H[j] * lit[i][j] for j in range(m)) <= x1[i])
      model.Add(y1[i]+h[i] == y2[i])
      model.Add(sum(H[j] * lit[i][j] for j in range(m)) >= y2[i])
    # no overlap for items in same bin
    x_interval = [model.NewIntervalVar(x1[i], w[i], x2[i], f'x_interval[{i}]') for i in range(n)]
    y_interval = [model.NewIntervalVar(y1[i], h[i], y2[i], f'y_interval[{i}]') for i in range(n)]
    model.AddNoOverlap2D(x_interval, y_interval)
    #obj = model.NewIntVar(0, sum(C), 'obj')
    #model.Add(obj>=sum(u[j]*C[j] for j in range(m)))
    for j in range(m):
        model.Add(sum(lit[i][j] for i in range(n))>0).OnlyEnforceIf(u[j])
        model.Add(sum(lit[i][j] for i in range(n))<=0).OnlyEnforceIf(u[j].Not())
    obj = model.NewIntVar(0,sum(C),'obj')
    model.Add(obj==sum(u[j]*C[j] for j in range(m)))
    model.Minimize(obj)
    # M = solver.IntVar(0.0, infinity, 'M')
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    COUNTER.start()
    status = solver.Solve(model)
    COUNTER.end()
    res = []
    rootLogger.info("Solved in {:.2f}s".format(COUNTER.get_duration()))
    rootLogger.info("Objective value = {:d}".format(int(solver.Value(obj))))
    for i in range(n):
      for j in range(m):
        if solver.Value(lit[i][j]) == 1:
          res.append(j)
    rootLogger.info(solver.StatusName())
    rootLogger.info(res)
  
  def run(self, data_file, MIP = True, CP = True):
      rootLogger.info("###################################")
      data = self.read_data(data_file)
      if MIP:
        self.run_MIPS(data)
      if CP:
        self.run_CP(data)

if __name__ == '__main__':
  TIME_LIMIT = 300
  N_ITEMS = 1

  COUNTER = TimeCounter()
  manager = Experiment_Manager()

  logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
  rootLogger = logging.getLogger()
  rootLogger.setLevel(logging.INFO)

  fileHandler = logging.FileHandler("{0}/{1}.log".format('./', f'experiments_N_ITEMS_{N_ITEMS}'), mode = 'w')
  fileHandler.setFormatter(logFormatter)
  rootLogger.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  rootLogger.addHandler(consoleHandler)

  rootLogger.info("Experiment start")
  for file_name in os.listdir('./data_v2'):
    #DATA_FILE = './Data Input/2005PisingerSigurd.bpp'
    try:
      manager.run(os.path.join('./data_v2/',file_name))
    except:
      continue