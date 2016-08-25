import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


#X为特征值，y为预测值,

def Htheta(X,theta):
    h=X * theta
    return h

def sumOfSquares_Htheta_y(htheta,y):
    s=pow(htheta - y,2)
    return s

def costFunctionJ(X,y,theta): #线性回归，损失函数
    J=0
    m=X[:,1].size
    rows=X[:,1].size  #计算矩阵行数
    columns=X[1].size #计算矩阵列数
    i=1
    k=1
    summary=0
    h_theta = Htheta(X,theta)
    for i in range(rows):
        summary = summary + sumOfSquares_Htheta_y(h_theta[i,0],y[i, 0])
        #print("y=", y[i, 0], ",htheta=", h_theta,",summary=",summary)
    J=1/(2*m)*summary
    #print("j=",J)
    return J

def testCostFunctionJ(_theta):
    # 构造参数为[1 x]的矩阵,定义theta0为1,根据Htheta的计算公式 Htheta=theta0+theta1*X1++theta1*X2+。。。+thetaN*XN
    X = np.matrix('1 1; 1 2; 1 3;1 4;1 5')  #特征参数
    theta = np.matrix(_theta)
    y = np.matrix('1;2;3;4;5')   #预测值
    # print(X[:,1].size)
    J = costFunctionJ(X, y, theta)
    #print(J)
    return J

def drawCostFunctionJ():
    i = -3
    x = []
    y = []
    while (i <= 3):
        tmp = '0;' + str(i)
        htheta = testCostFunctionJ(tmp)
        x.append(i)
        y.append(htheta)
        pl.plot(i, htheta, 'b*')
        i = i + 0.1
        # print("i=", i, htheta)
    pl.plot(x, y, 'r-')
    pl.xlim(-7.0, 7.0)  # set axis limits
    pl.ylim(0.0, 7.0)  # set axis limits
    pl.show()

#Q(j):=Q(j)-a*[d/(d*Q(j))*J(Q(0),Q(1))] (for j=0 and j=1
def bathGradientDescentAlgorithm():
    # temp0:=Q0-a*d/d*Q0*J(Q0,Q1)
    # temp0:=Q1-a*d/d*Q1*J(Q0,Q1)
    # Q0:=temp0
    # Q1:=temp1
    # a为学习曲率, learning rate
    altha=1.0  #学习曲率

    return

#testCostFunctionJ()
#X = np.matrix([[1, 1], [1, 2], [1, 3]])
X = np.matrix('1 1; 1 2; 1 3')
theta = np.matrix('0;0.5')
y = np.matrix('1;2;3')
drawCostFunctionJ()

