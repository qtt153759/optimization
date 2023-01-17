import copy
#compute states
def TRY(city):
    for i in range(K):
        y[i].append(city)
        if(city==N):
            if(y[0]!=[]):
                state.append(copy.deepcopy(y))
        else:
            TRY(city+1)
        y[i].pop()


from sys import maxsize
from itertools import permutations
 
def TSPalgo(graph, s, A):
 
    # store all vertex apart from source vertex
    vertex = A
    if vertex==[]:
        return 0
 
    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation=permutations(vertex)
    for i in next_permutation:
 
        # store current Path weight(cost)
        current_pathweight = 0
 
        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
 
        # update minimum
        min_path = min(min_path, current_pathweight)
         
    return min_path
 

def calculateDistance():
    lenState=len(state)
    
    for i in range(lenState):
        totalDis=0
        TSP=[]
        maxTSP=0
        for j in range(K):
            s=0
            TSP=TSPalgo(graph, s, state[i][j])    # Calculate distance of a person in each state
            maxTSP=max(maxTSP,TSP)              # Calculate the minimun distance value in each state
            totalDis+=TSP                          # Calculate total distance in a state
        dictDistance[totalDis]=maxTSP
        dictState[totalDis]=state[i]
 
 
if __name__ == "__main__":
    y=[[],[],[]]
    N, K = input("input N and K:").split()
    N=int(N)
    K=int(K)
    state=[]
    dictDistance=dict()  #restore total distances and the minimum distance value of the delivers
    dictState=dict()     #restore total distances and the states
    # distance matrix
    graph = [[0, 184, 222, 177,216,231], [184, 0, 45, 123,128,200],[222,45,0,129,121,203], [177,123,129,0,46,83],[216,128,121,46,0,83],[231,200,203,83,83,0]]


    TRY(1)
    calculateDistance()


    # sort minimum distance value by total distance
    sorted(dictDistance.items())
    
    # decrease number of states
    dictDistance0=dictDistance.copy()
    a=0
    for i in dictDistance0.keys():
        if(a>int(len(dictDistance0)/K)):
            del dictDistance[i]
        a+=1
        
    # print result
    dictDistance=sorted(dictDistance.items(), key=lambda item: item[1])
    print(dictState[int(dictDistance[0][0])])