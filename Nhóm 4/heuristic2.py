import numpy as np
import os
import time
begin=time.time()
def read_input(file):
    with open(os.path.join('data', file), 'r') as f:
        [n, k] = [int(x) for x in f.readline().split()]
        capacity = [int(x) for x in f.readline().split()]
        route = []
        for i in range(2 * n + 1):
            route.append([int(x) for x in f.readline().split()])
    return n, k, capacity, route


N, k, c, d = read_input('N200-K15.txt')


# print(N, k, c)
d = np.array(d)
capacity = sorted(c, reverse=True)
pick_up_lst = [i for i in range(1, N+1)]
drop_off_lst = [i for i in range(N+1, 2*N+1)]

initial_lst = []
# list khoảng cách từ 0 đến các điểm pick up + drop off đến 0
for i in range(1, N+1):
    initial_lst.append(d[0, i]+d[i+N][0])
print(initial_lst)
res = np.array(sorted(range(len(initial_lst)),
               key=lambda x: initial_lst[x])[:k])+1
# sort node gân nhiều điểm pick up nhất


def total_dis_to_pickup_node(n):
    s = 0
    for i in range(1, N+1):
        if i not in res:
            s += d[n][i]
    return s


# print('res',res)
total_dis = [total_dis_to_pickup_node(i) for i in res]
# print(total_dis)
res = sorted(res, key=lambda x: total_dis_to_pickup_node(x), reverse=False)
route_lst = [[i, i+N] for i in res]
visited_node = []
for i in res:
    visited_node.append(i)
    visited_node.append(i+N)
for i in route_lst:
    i.insert(0, 0)
    i.append(0)
# líst các điểm gần 0 nhất

def check_capacity(lst, r):
    queue = []
    for i in range(len(lst)):
        if lst[i]-N in queue:
            queue.pop(0)
        else:
            queue.append(lst[i])
            if len(queue) > r:
                return False
    return True


# Heuristic2
visited_node = []

for i in res:
    visited_node.append(i)
    visited_node.append(i + N)
route_lst = [[i, i + N] for i in res]
for i in route_lst:
    i.insert(0, 0)
    i.append(0)
insert1 = 1
insert2 = 1
insert3 = 1
for i in range(1, N + 1):

    arg1 = 1
    arg2 = 1
    if i not in visited_node:
        s = float('inf')
        for r in range(len(route_lst)):
            fake_route = route_lst[r].copy()
            s1 = float('inf')
            s2 = float('inf')
            for j in range(1, len(fake_route) - 1):
                lst = fake_route.copy()
                lst.insert(j + 1, i)
                if check_capacity(lst, capacity[r]):
                    if d[fake_route[j]][i] + d[i][fake_route[j + 1]] < s1:
                        s1 = d[fake_route[j]][i] + d[i][fake_route[j + 1]]
                        arg1, arg2 = r, j + 1
                lst = fake_route.copy()
                lst.insert(j, i)
                if check_capacity(lst, capacity[r]):
                    if d[i][fake_route[j]] + d[fake_route[j - 1]][i] < s1:
                        s1 = d[i, fake_route[j]] + d[fake_route[j - 1]][i]
                        arg1, arg2 = r, j
            fake_route.insert(arg2, i)
            arg3 = 1
            for v in range(arg2, len(fake_route) - 1):
                if d[fake_route[v]][i + N] + d[i + N][fake_route[v + 1]] < s2:
                    s2 = d[fake_route[v]][i + N] + d[i + N][fake_route[v + 1]]
                    arg3 = v + 1
            for v in range(arg2 + 1, len(fake_route)):
                if d[i + N][fake_route[v]] + d[fake_route[v - 1]][i + N] < s2:
                    s2 = d[i + N][fake_route[v]] + d[fake_route[v - 1]][i + N]
                    arg3 = v
            if s1 + s2 < s:
                s = s1 + s2
                insert1 = r
                insert2 = arg2
                insert3 = arg3
        route_lst[insert1].insert(insert2, i)
        route_lst[insert1].insert(insert3, i + N)
    visited_node.append(i)
for i in range(len(route_lst)):
    print('capacity', capacity[i])
    print('path', route_lst[i])
s = 0
for i in route_lst:
    for j in range(0, len(i) - 1):
        s += d[i[j], i[j + 1]]
print('heuristic 2 solution:', s)
print('time : ', time.time()-begin )