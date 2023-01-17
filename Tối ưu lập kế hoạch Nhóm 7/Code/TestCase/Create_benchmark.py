import os
import random
import numpy as np
print("Nhap N,D,a,b : ")
inputData = input()
inputData = inputData.split(" ")
N = int(inputData[0])
D = int(inputData[1])
a = int(inputData[2])
b = int(inputData[3])    

if N < 5*a : 
    print("Not exist solution satisfied")
    exit()

f = open("TestCase/benchmark.txt","w+")
f.write(str(N) + " " + str(D) + " " + str(a) + " " + str (b) + "\n")


candidate = []
for i in range(N) :
    candidate.append(i)
offline = []
Schedule = np.zeros([N,D],dtype=int)
for i in range(D) : 
    random.shuffle(candidate)
    for j in range(4*a) :
        Schedule[candidate[j],i] = 1
    for j in offline :
        candidate.append(j)
    offline.clear()
    for j in candidate[0:a] :
        offline.append(j)
    del candidate[0:a]
for i in range(N) :
    rand = random.random()
    for j in range(D) :
        if Schedule[i,j] == 0 and random.random() < rand :
            f.write(str(j+1) + " ")
    f.write("-1")
    if i < N-1 :
        f.write("\n")
f.close()
new_name ="TestCase/" + str(N) + "_" + str(D) + "_" + str(a) + "_" + str (b) + ".txt"
os.rename("TestCase/benchmark.txt", new_name)
print("Create Success") 