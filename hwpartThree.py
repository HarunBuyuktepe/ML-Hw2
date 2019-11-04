import random


def dataSet(N=100):
    #Create random data set
    d = []
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        d.append([x, y])
    return d

def targetFunction():
    #Create target function
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)

    a = abs(x1 - x2) / abs(y1 - y2)
    b = y1 - a * x1
    return [a, b]  # a*x + b

def applicateFunction(trainingSet, tFunction):
    'maps a point (x1,x2) to a sign -+1 following function f '
    x1 = trainingSet[0]
    y1 = trainingSet[1]

    y = tFunction(x1)
    compare_to = y1
    return comp(y, compare_to)
    #training set's y value > compare function's result y is 1 or 0

def comp(x, y=0):
    #Compare x and y
    if x > y:
        return +1
    else:
        return -1

def splitMissPlacedPoint(t_set, w):
    #return misclassified point index in trainig set
    res = tuple()

    for i in range(len(t_set)):
        point = t_set[i][0]
        s = hypothesis(w, point)
        yn = t_set[i][1]
        if s != yn:
            res = res + (i,)
            #collect to misclassified data indexes
    return res

def hypothesis(w, x):
    #hypothesis function to classify training set
    res = 0
    for i in range(len(x)):
        res = res + w[i] * x[i]
    return comp(res)

def pla(N):
    iteration = 0
    dataset = dataSet(N)
    funcVals = targetFunction()
    tFunction = lambda x: funcVals[0] * x + funcVals[1]
    w = [0, 0, 0]
    t_set = []

    for i in range(len(dataset)):
        trainingSet = dataset[i]
        y = applicateFunction(trainingSet, tFunction)  # map x to +1 or -1 for training points
        t_set.append([[1, trainingSet[0], trainingSet[1]], y])

    iterate = True
    while iterate:
        iteration = iteration + 1
        misclassified_set = splitMissPlacedPoint(t_set, w)
        #detect mis classified set
        if len(misclassified_set) == 0: break
        # if there are no misclassified points break
        index = random.randint(0, len(misclassified_set) - 1)
        #choose random misclassified trainind set
        p = misclassified_set[index]
        point = t_set[p][0]

        s = hypothesis(w, point)
        yn = t_set[p][1] #give misclassified result

        if s != yn:#update weights if misclassified
            xn = point
            w[0] = w[0] + yn * xn[0]
            w[1] = w[1] + yn * xn[1]
            w[2] = w[2] + yn * xn[2]
    return w, iteration, tFunction,t_set

def differenceCalc(tFunction, w,limit):
    #return avarage difference
    count = 0
    diff = 0

    while count < limit:
        count = count + 1
        #examine with random values
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        vector = [1, x, y]

        sign_f = comp(tFunction(x), y)
        sign_g = hypothesis(w, vector)
        #check result and count
        if sign_f != sign_g: diff = diff + 1

    return diff / (count * 1.0)

def app(N_samples=1, N_points=10):
    iterations = []  # vector of iterations needed for each PLA
    diff = []  # vector of difference average between f and g
    samples = [[],[]]  # vector of 1 clasified, 0 misclassified
    b_misclassified = False
    countMiss=0
    for i in range(N_samples):
        # run PLA in sample
        w, iteration, tFunction,t_set = pla(N_points)
        iterations.append(iteration)
        diff.append(differenceCalc(tFunction, w,N_points))
        # check points are classified or not
        for i in range(len(t_set)):
            point = t_set[i][0]
            s = hypothesis(w, point)
            yn = t_set[i][1]
            if yn != s:
                samples.append(0)
                b_misclassified = True
                countMiss=countMiss+1
                break
        if not b_misclassified: samples.append(1)
        b_misclassified = False

    print('number of samples misclassified: %s ', countMiss)
    print()
    print('number of classified samples: %s ', samples.count(1)-countMiss)
    print()
    print('number of iteration avg: %s ' , (str(sum(iterations) / len(iterations) * 1.0)))
    print()
    print('average of difference in function g: %s' , (sum(diff) / (len(diff) * 1.0)))

print()
print()

print('8.')
app(1000,100)

print()
print()
print()
print()
print()
print()
print()
print()

print('9.')
app(1000,100)

print()
print()
print()
print()
print()
print()

print('10.')
app(1000,10)

print()
print()
print()
print()




