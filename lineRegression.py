import numpy as np
import math
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
    print("j=",J)
    return J

def testCostFunctionJ(_theta):
    # 构造参数为[1 x]的矩阵,定义theta0为1,根据Htheta的计算公式 Htheta=theta0+theta1*X1++theta1*X2+。。。+thetaN*XN
    X = np.matrix('1 1; 1 2; 1 3')
    theta = np.matrix(_theta)
    y = np.matrix('1;2;3')
    # print(X[:,1].size)
    J = costFunctionJ(X, y, theta)
    print(J)
    return J

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
print(theta.T)
print(X.T)
htheta=X*theta
print("y=",X*theta)
pl.figure(1,figsize=(8,8)) # 创建图表1
ax1 = plt.subplot(211) # 在图表2中创建子图1
# pl.plot(-10,-10)
# pl.plot(10,10)
# pl.plot(X[:,1], y,'b*')#,label=$cos(x^2)$)
# pl.plot(X[:,1], y,'r')
# pl.show()
#
i=-100
for i in range(100):
    tmp='0;'+str(i)
    #print("tmp=",tmp)
    htheta=testCostFunctionJ(tmp)
    print("i=",i,htheta)
    pl.plot(i,htheta,'b*')

pl.show()
