import numpy as np
import pandas as pd


def SelectPdf(Num,data_type="standard_exponential"):
    if data_type == "exp-non-gaussian":
        noise = np.random.uniform(-1, 1, size=Num) ** 5

    elif data_type == "gaussian":
        noise = np.random.normal(loc = 0, scale = 1, size=Num)

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    elif data_type == "exponential":  #exp-exponential

        noise = pow(np.random.exponential(scale=1, size=Num),1)

    elif data_type == "standard_exponential":
        noise =np.random.standard_exponential(size=Num)

    else: #uniform
        noise =np.random.uniform(-1, 1, size=Num)

    return noise

def Toa():
    return 1



def ToBij():
    signed = 1
    if np.random.randn(1) >= 0:
        signed = -1
    return signed * np.random.uniform(.5, 2)


def simulation_Gaussian(Num):
    U = SelectPdf(Num)

    effect = []

    for _ in range(0,5):
        effect.append(ToBij())


    X1 = SelectPdf(Num)*Toa()+U*ToBij()
    X2 = SelectPdf(Num)*Toa()+U*ToBij()+X1*effect[0]
    X3 = SelectPdf(Num)*Toa()+U*ToBij()
    X4 = SelectPdf(Num)*Toa()+U*ToBij()
    X5 = SelectPdf(Num)*Toa()+U*ToBij()+X4*effect[2]
    X6 = SelectPdf(Num)*Toa()+U*ToBij()+X5*effect[3]
    Y = SelectPdf(Num)*Toa()+U*ToBij()+X6*effect[4]+X2*effect[1]


    data2 = pd.DataFrame(np.array([X1,X2,X3,X4,X5,X6,Y]).T,columns=['X1','X2','X3','X4','X5','X6','Y'])

    return data2,effect











#1-factor model
def Gdata1(Num=3000):

    U = SelectPdf(Num)


    Z = SelectPdf(Num)*Toa()+U*ToBij()



    X = SelectPdf(Num)*Toa()+U*ToBij()+ToBij()*Z


    W = SelectPdf(Num)*Toa()+U*ToBij()


    Y = SelectPdf(Num)*Toa()+U*ToBij()+X*1.2+W*ToBij()


    data = pd.DataFrame(np.array([X,Y,Z,W]).T,columns=['X','Y','Z','W'])

    #data = (data-data.mean())/data.std()


    return data




def Gdata2(Num=3000):

    U = SelectPdf(Num)
    U2 = SelectPdf(Num)


    Z = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()
    Z1 = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()

    W = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()
    W1 = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()

    T = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()



    X = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()
    X1 = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()




    Y = SelectPdf(Num)*Toa()+U*ToBij()+U2*ToBij()+X*1.2+X1*0.8


    data2 = pd.DataFrame(np.array([X,Y,Z,W,Z1,W1,X1,T]).T,columns=['X','Y','Z','W','Z1','W1','X1','T'])

    return data2

def main():
    pass

if __name__ == '__main__':
    main()
