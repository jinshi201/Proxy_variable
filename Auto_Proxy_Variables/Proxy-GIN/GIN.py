import numpy as np
import pandas as pd
from indTest import *
import indTest.fastHSIC as hsic






def GIN(X,Z,data,alpha):
    omega = getomega(data,X,Z)
    tdata= data[X]
    result = np.dot(omega, tdata.T)


    for i in Z:
        #get the Z data from data,test the Z whether satify the GIN
        temp = np.array(data[i])
        flag =hsic.test(result.T,temp,alpha)

        if not flag:#not false == ture  ---> if false
            return False
    # if no any emelent in Z is voiate the GIN,this is satify GIN,return TRUE
    return True



def getomega(data,X,Z):
    cov_m =np.cov(data,rowvar=False)
    col = list(data.columns)

    Xlist = []
    Zlist = []
    for i in X:
        t = col.index(i)
        Xlist.append(t)
    for i in Z:
        t = col.index(i)
        Zlist.append(t)



    B = cov_m[Xlist]
    B = B[:,Zlist]



    A = B.T
    u,s,v = np.linalg.svd(A)
    lens = len(X)
    #print(v.T)
    omega =v.T[:,lens-1]


    omegalen=len(omega)
    omega=omega.reshape(1,omegalen)
    #print(omega)

    return omega

