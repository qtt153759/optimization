import numpy as np
import numpy.random as npr

def check_result(C, A, c, a, q):
    if sum(np.array(c)*np.array(q))<C and sum(np.array(a)*np.array(q))<A:
        return True
    else:
        return False

def main(N, A, C, c, a, f, m, trial = 20):
    q = m.copy()
    temp = np.array(f)-(np.array(a)+np.array(c))
    # choosen_index = np.argsort(temp)[int(-N/2-1):]
    choosen_index = np.argsort(temp)[-2:]
    roulette_wheel = 1/temp[choosen_index]
    roulette_wheel = roulette_wheel/sum(roulette_wheel)
    print(roulette_wheel)
    for i in range(trial):
        print(q)
        # index = npr.choice(int(N/2+1), p=roulette_wheel)
        index = npr.choice(2, p=roulette_wheel)
        temp = q.copy()
        temp[choosen_index[index]]+=1
        if check_result(C, A, c, a, temp):
            q = temp

    print("Objective value: "+str(sum(np.array(f)*np.array(q))))
    print(q)

if __name__ == '__main__':
     
    N=8
    A=916
    C=478
    c=[ 9, 8, 9, 9, 7, 9, 9, 8]
    a=[ 18, 15, 17, 17, 13, 18, 16, 15]
    f=[ 270, 238, 256, 260, 201, 276, 250, 239]
    m=[ 2, 3, 1, 2, 9, 9, 2, 1]
    
    main(N, A, C, c, a, f, m)

