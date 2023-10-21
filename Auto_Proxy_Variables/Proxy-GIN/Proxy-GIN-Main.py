import numpy as np
import pandas as pd
import GIN
import Gdata
import itertools

#Open the debug printing
debug = 1


#An example of our manuscript
def main():
    data = Gdata.Gdata1(10000)

    X = ['X']
    Y = 'Y'
    q = 1
    alpha = 0.01

    estimated_effect = Proxy_GIN(X,Y,q,data,alpha)

    print(estimated_effect)
    print('The true effect is 1.2')




def Proxy_GIN(X, Y, q, data, alpha):
    '''
    Main function (refer to "Algorithm 2 Proxy-GIN" in our manuscript)
    # X: treatments, datatype: list; e.g., X=['x1','x2']
    # Y: outcome, data type: string; e.g., Y='y'
    #q: the number of latent confounders
    #data: dataset, datatype: DataFrame
    #alpha: confidence threshold

    '''
    indexs = list(data.columns)

    EffectSet = []

    for Xk in X:
        B_candicate = indexs.copy()
        B_candicate.remove(Xk)
        B_candicate.remove(Y)

        if debug:
            print('Current candicate Set is ', B_candicate)

        CandiSet = itertools.permutations(list(B_candicate), 2*q)

        for NC in CandiSet:
            NC = list(NC)
            if debug:
                print('Current NC set is ', NC)


            A = []
            B = []

            for i in range(0,q):
                A.append(NC[i])

            for i in range(q,len(NC)):
                B.append(NC[i])

            TestA1 = B.copy()
            TestA2 = A.copy()

            TestB1 = A.copy()
            TestB2 = B.copy()


            TestA1.append(Xk)
            TestA1.append(Y)

            TestA2.append(Xk)

            flag1 = GIN.GIN(TestA1,TestA2,data,alpha)

            if debug:
                print('Current Test Rank constraints in Rule 1_1 ', TestA1, TestA2,flag1)


            TestB1.append(Xk)

            flag2 = GIN.GIN(TestB1,TestB2,data,alpha)

            if debug:
                print('Current Test Rank constraints in Rule 1_2 ', TestB1, TestB2,flag2)


            if flag1 and flag2:
                if debug:
                    print('Current A and B are ',A,B)
                effect = EstimateEffect(A,B,Xk,Y,data)
                break
            else:
                effect = -999

        EffectSet.append(effect)


    print(EffectSet)
    return EffectSet




def EstimateEffect(A,B,Xk,Y,data):
    M =data.cov()

    l1 = A.copy()
    l1.insert(0,Xk)
    l2 = B.copy()
    l2.insert(0,Y)

    M2 = M[l1].loc[l2]

    if debug:
        print('Estimate M2: \n',M2)

    t3 = np.linalg.det(M2)

    l3 = B.copy()
    l3.insert(0,Xk)


    M3 = M[l1].loc[l3]


    if debug:
        print('Estimate M3: \n',M3)

    t4 = np.linalg.det(M3)

    r = t3/t4

    return r










if __name__ == '__main__':
    main()
