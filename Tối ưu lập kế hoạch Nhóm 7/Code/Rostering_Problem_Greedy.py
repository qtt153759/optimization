import numpy as np
import random

f = open("TestCase/100_30_15_25.txt")
fa = f.readlines()
tmp = fa[0].split()
N = int(tmp[0])
D = int(tmp[1])
a = int(tmp[2])
b = int(tmp[3])
day_off = np.zeros([N,D], dtype = int)
for i in range(1,len(fa)) :
    tmp = fa[i].split()
    for j in range(len(tmp) - 1) :
        m = int(tmp[j])
        day_off[i-1,m-1] = 1

Schedule = np.zeros([N,D+1], dtype=int)
num_night = [0]*N
min_num_night = 1000000
num_night_to_now = [0] * N

def heuristic (index : int) :
    global num_night
    return num_night[index]

for j in range(1,D+1) :
    for i in range(N) :
        if day_off[i,j-1] == 1 :
            Schedule[i,j] = -1
for j in range(1,D+1) :
    hight_candidate_night = []
    if j < D :
        for i in range(N) :
            if day_off[i,j-1] == 0 and day_off[i,j] == 1 :
                hight_candidate_night.append(i)
    random.shuffle(hight_candidate_night)
    hight_candidate_night.sort(key = heuristic, reverse=False)
    l = min(a, len(hight_candidate_night))
    for i in hight_candidate_night[0:l] :
        Schedule[i,j] = 4
        num_night[i] += 1

    candidate_night = []
    l = 0
    for i in range(N) :
        if Schedule[i,j] == 4 :
            l+=1
        if j < D :
            if Schedule[i,j+1] < 4 and day_off[i,j-1] == 0 :
                candidate_night.append(i)
        else :
            if day_off[i,j-1] == 0 :
                candidate_night.append(i)
    if len(candidate_night) + l < a :
        break
    candidate_night.sort(key = heuristic, reverse=False)
    if l < a :
        for i in candidate_night[0:a-l] :
            Schedule[i,j] = 4
            num_night[i] += 1
    l = 0
    for i in range(N) :
        if Schedule[i,j] == 4 or day_off[i,j-1] == 1 :
            l+=1
    if N - l < 3*a :
        break

    if j == D :
        min_num_night = max(num_night)
if min_num_night < 1000000 :
    print("Result : ",  min_num_night)
else : 
    print("Cannot find any Solution :(((")