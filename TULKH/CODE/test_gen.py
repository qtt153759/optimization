


from random import randint

t = [[6,3],[10,7],[20,15],[50,45], [100,70],[200,100],[500,250],[800,600],[2500,600]]

for i in range(len(t)):

    n = t[i][0]
    m = t[i][1]
    k = int(0.1*n*(n-1)/2)

    f = open("input_"+str(n)+"_"+str(m)+"_"+str(k)+".txt", "w")

    c = []
    total_capacity = 0

    for i in range(m):
        tmp = randint(50, 200)
        c.append(tmp)
        total_capacity = total_capacity + tmp

    f.write(str(n))

    f.write('\n')

    s = str(randint(1, total_capacity))

    for i in range(1, n):
        s = s + ' ' + str(randint(1, total_capacity))

    s = s + '\n'

    f.write(s)

    f.write(str(m))

    f.write('\n')

    s = str(c[0])

    for i in range(1, m):
        s = s + ' ' + str(c[i])

    f.write(s)

    f.write('\n')

    f.write(str(k))

    f.write('\n')

    for i in range(k):
        while True:
            x = randint(0, n-1)
            y = randint(0, n-1)
            if x != y :
                f.write(str(x) + " " + str(y) + '\n')
                break;

    f.close()