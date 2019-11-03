import numpy

def data(N):
    'return N random points (x,y)'
    d = []
    for i in range(N):
        x = numpy.random.uniform(-1, 1)
        y = numpy.random.uniform(-1, 1)
        d.append([x, y])
    return d

def generateRandomTargetFunction():
    x2 = numpy.random.uniform(-1, 1)
    y2 = numpy.random.uniform(-1, 1)
    x1 = numpy.random.uniform(-1, 1)
    y1 = numpy.random.uniform(-1, 1)
    a = abs(y1 - x2) / abs(x1 - y2)
    b = y2 - a * x2
    return  a,b

def applicateFunction(coord, targetFunction):
    'maps a point (x1,x2) to a sign -+1 following function f '
    x = coord[0] #coordinate of data set
    y = coord[1]
    print('x',x,'y',y)
    y1 = targetFunction(x)
    print('y',y1)

    if y1 > y:
        return +1
    else:
        return -1


def build_misclassified_set(t_set, w):
    '''returns a tuple of index of t_set items
such that t_set[index] is misclassified <=> yn != sign(w*point)'''
    res = tuple()
    print('tuple ne dayıcım',res)
    for i in range(len(t_set)):
        point = t_set[i][0]
        print('point',point)
        print('w',w)
        s = h(w, point)
        yn = t_set[i][1]
        if s != yn:
            print('s',s,'yn',yn,'x',point[1])
            res = i
            print('peki buraya hiç uğrar mısın canim',i)
    return res


def h(w, x):
    'Hypothesis function returns w0 x0 + w1 x1 ... + wn xn'
    res = 0
    for i in range(len(x)):
        res = res + w[i] * x[i]
        print('res',res,'w',w)
    if res > 0:
        return +1
    else:
        return -1


def PLA(N):
    dataSet = (data(N))  # our data set generated here
    funcCoordinates = generateRandomTargetFunction()  # random target function's norm coordinates
    print('funcCoordinates', funcCoordinates[0], funcCoordinates[1])
    targetFunction = lambda x: funcCoordinates[0] * x + funcCoordinates[1]  # our target function created "ax+b"

    t_set = []
    for i in range(len(dataSet)):
        coor = dataSet[i]
        y = applicateFunction(coor, targetFunction)  # map x to +1 or -1 for training sets
        t_set.append([[1, coor[0], coor[1]], y])

#    for i in range(len(t_set)):
#        print('training set',t_set[i])
    iterate=True
    count=0
    w=[0,0,0]
    while iterate:
        count=count+1
        misclassified_set = build_misclassified_set(t_set, w)
        if (misclassified_set is None) : break
        print('misclassified_set',misclassified_set)
        index = numpy.random.randint(0, (misclassified_set) )
        print('index',index)
        p = misclassified_set
        point = t_set[p][0]

        s = h(w, point)
        yn = t_set[p][1]
        if s != yn:
            xn = point
            w[0] = w[0] + yn * xn[0]
            w[1] = w[1] + yn * xn[1]
            w[2] = w[2] + yn * xn[2]
    return t_set, w, count, targetFunction


p=PLA(100)
for i in range(len(p)):
    print(p[i])



