import numpy as np

from numpy.linalg import inv


av_Ein =0
for P in range(0,1000):
    #Target Line calculation
    line = {}#Target Line
    x1,y1 = np.random.uniform(-1,1,2)
    x2,y2 = np.random.uniform(-1,1,2)
    m = (y2-y1)/(x2-x1)
    line['x'] = m
    line['y'] = -1
    line['c'] = y1-m*x1
    X = np.random.uniform(-1,1,(100,2))
    X = np.c_[np.ones(100),X]
    #------
    #Target calculation

    Y =[]#Target
    for i in range(0,100):
        Y.append(line['c']*1+line['x']*X[i][1]+line['y']*X[i][2])

    #------
    #For calculating the weight function W = pseudoinverse(X) x Y

    X_tran = X.transpose()
    temp_1 = np.matmul(X_tran,X)
    temp_inv = inv(temp_1)
    temp_1 = np.matmul(temp_inv,X_tran)
    W = np.matmul(temp_1,Y)

    #end of calculation

    #E_in(insample error) claculation for this final hypothesis

    E_in=0#(No of times the final hypothesis not matching with target /len(Target))

    #No of misclassified insample points

    for k in range(0,100):
        if np.sign(X[k][0]*W[0]+X[k][1]*W[1]+X[k][2]*W[2]) != np.sign(Y[k]):
            E_in = E_in+1
        #E_in = E_in+math.pow((lm.predict([X[k]])-Y[k]),2)

    #---

    E_in = E_in/100#E_in calculation
    av_Ein = av_Ein+E_in

av_Ein = av_Ein/1000 #(AVg of all the insmple error)
print(av_Ein)