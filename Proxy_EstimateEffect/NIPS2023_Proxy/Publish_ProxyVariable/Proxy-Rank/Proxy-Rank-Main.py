import CCARankTester as CRANK
import numpy as np
import Gdata
import itertools


#Open the debug printing
debug = 1

#An example of the paper
def main():
    data,True_effect = Gdata.simulation_Gaussian(10000)

    X = ['X1','X2','X3','X4','X5','X6']
    Y = 'Y'
    q = 1

    alpha = 0.005

    estimated_effect = ProxyRank(X, Y, q, data, alpha)

    print(estimated_effect)
    print(True_effect)




def ProxyRank(X, Y, q, data, alpha):
    '''
    Main function (refer to "Algorithm 1 Proxy-Rank" in our manuscript)
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

        #CandiSet = itertools.combinations(list(B_candicate), 2*q+1)
        CandiSet = itertools.permutations(list(B_candicate), 2*q+1)

        for NC in CandiSet:
            NC = list(NC)
            if debug:
                print('Current NC set is ', NC)

            RanktestY = []
            RanktestZ = []

            # |A|+|Q| to Y  (here dim of RanktestY is equal to q+1)
            for i in range(0,q+1):
                RanktestY.append(NC[i])

            for i in range(q+1,len(NC)):
                RanktestZ.append(NC[i])

            RanktestY.append(Xk)
            RanktestZ.append(Y)
            RanktestZ.append(Xk)

            #here we construct Y = [A,Q,Xk] and Z = [Xk,Y,B]


            flag1 = RankTestFunction(RanktestY, RanktestZ, data, q+1, alpha)

            if debug:
                print('Current Test Rank constraints in Rule 1_1 ', RanktestY, RanktestZ,flag1)


            RanktestY2 = []
            RanktestZ2 = []

            A = []
            B = []

            #add |A| to Y
            for i in range(0,q):
                RanktestY2.append(NC[i])
                A.append(NC[i])
            #Add |B| to Z
            for i in range(q+1,len(NC)):
                RanktestZ2.append(NC[i])
                B.append(NC[i])


            RanktestY2.append(Xk)

            #add Q to Z
            RanktestZ2.append(NC[q])

            #here we construct Y = [A,Xk] and Z = [B,Q]


            flag2 = RankTestFunction(RanktestY2, RanktestZ2, data, q, alpha)
            if debug:
                print('Current Test Rank constraints in Rule 1_2 ', RanktestY2, RanktestZ2, flag2)

            if flag1 and flag2:
                if debug:
                    print('Current A and B are ',A,B)
                effect = EstimateEffect(A,B,Xk,Y,data)
                break
            else:
                effect = -999





        '''
        Apply Rule 2 to find NC
        '''

        effect = Rule2(effect, B_candicate, q, Xk, Y, data, alpha)



        EffectSet.append(effect)




    print(EffectSet)
    return EffectSet




#Rank Test by CCA method
def RankTestFunction(Y, Z, data1, Hyp_det, alpha):

    data = data1.copy()

    RankTest = CRANK.CCARankTester(data, alpha)
    indexs = list(data.columns)
    TrY = []
    TrZ = []

    for i in Y:
        TrY.append(indexs.index(i))

    for i in Z:
        TrZ.append(indexs.index(i))

    flag = RankTest.test(TrY,TrZ,Hyp_det)



    return (not flag)

# Rule 2 for detecting vaild NCE and NCO set
def Rule2(effect, B_candicate, q, Xk, Y, data, alpha):
    if effect == -999:
        CandiSet = itertools.permutations(list(B_candicate), 2*q+2)

        for NC in CandiSet:
            NC = list(NC)
            if debug:
                print('For Rule 2, Current NC set is ', NC)

            A = []
            B = []

            for i in range(0,q+1):
                A.append(NC[i])

            for i in range(q+1,len(NC)):
                B.append(NC[i])

            RankTestA1 = A.copy()
            RankTestA2 = B.copy()
            RankTestB1 = A.copy()
            RankTestB2 = B.copy()



            RankTestA1.append(Xk)
            RankTestA2.append(Xk)

            flag1 = RankTestFunction(RankTestA1, RankTestA2, data, q+1, alpha)


            if debug:
                print('Current Test Rank constraints in Rule 2_1 ', RankTestA1, RankTestA2, flag1)

            RankTestB1.append(Xk)

            flag2 = RankTestFunction(RankTestB1, RankTestB2, data, q, alpha)

            if debug:
                print('Current Test Rank constraints in Rule 2_2 ', RankTestB1, RankTestB2, flag2)


            if flag1 and flag2:
                if debug:
                    print('Current A and B are ',A,B)
                effect = EstimateEffect(A,B,Xk,Y,data)
                break

    return effect




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
