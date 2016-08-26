import numpy as np
import matplotlib.pyplot as plt
import pylab as pl



#X为特征值，y为预测值,计算htheta的 值
def Htheta(X,theta):
    h=X * theta
    return h

def sumOfSquares_Htheta_y(htheta,y):
    s=pow(htheta - y,2)
    return s

def costFunctionJ(X,y,theta): #线性回归，损失函数
    J=0
    m=X[:,1].size
    rows=len(X)  #计算矩阵行数
    #columns=X[1].size #计算矩阵列数
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


#梯度下降
def bathGradientDescentAlgorithm(X,y):
    # temp0:=Q0-a*d/d*Q0*J(Q0,Q1)
    # temp0:=Q1-a*d/d*Q1*J(Q0,Q1)
    # Q0:=temp0
    # Q1:=temp1
    # a为学习曲率, learning rate
    # theta0 j=0 :  1/m (i=1 to m 求和)(htheta(x[i])-y[i])
    # theta1 j=1 :  1/m (i=1 to m 求和)(htheta(x[i])-y[i])*x[i]
    altha=0.001  #学习曲率
    m=len(X)
    htheta=0
    # 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
    epsilon = 0.0000000001
    cnt=0
    theta0=0
    theta1=0
    error0=0


    while True:
        cnt +=1

        for i in  range(m):
            htheta=theta0+theta1*X[i,1]-y[i,0]
            theta0-=altha*htheta*X[i,0]
            theta1-=altha*htheta*X[i,1]
           # print("cnt=",cnt,"htheta=",htheta,"theta0=",theta0,"theta1",theta1)
        theta=np.matrix(str(theta0)+";"+str(theta1))


        error1=costFunctionJ(X,y,theta)


        # for i in range(m):
        #     error1 += (y[i,0] - (theta0 + theta1 * X[i][1])) ** 2 / 2
        if abs(error1-error0)<epsilon:
            break
        else:
            error0=error1
       # print("error0",error0,"error1=",error1,"error0-error1=",abs(error1-error0))

    return theta0,theta1

#testCostFunctionJ()
#X = np.matrix([[1, 1], [1, 2], [1, 3]])
X = np.matrix('1 2; 1 4; 1 5;1 7;1 9;1 11;1 13;1 15;1 20')
theta = np.matrix('0;0.5')
y = np.matrix('1;1.3;1.5;1.7;1.9;2;2.1;2.5;3')
#drawCostFunctionJ()

print("X len =",len(X))


theta0,theta1=bathGradientDescentAlgorithm(X,y)
theta = np.matrix(str(theta0)+";"+str(theta1))
print("theta0=",theta0,"theta1",theta1)
pl.plot(theta0,theta1)
pl.plot(X[:,1],y,"r*")
pl.plot(X[:,1],X*theta,"b")
pl.show()