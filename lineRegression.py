import numpy as np
import math
import matplotlib.pyplot as pltJ



#X为特征值，y为预测值,
def costFunctionJ(X,y,theta): #线性回归，损失函数
    J=0
    m=X[:,1].size
    rows=X[:,1].size  #计算矩阵行数
    columns=X[1].size #计算矩阵列数
    i=1
    k=1
    summary=0
    for i in range(rows):
        h_theta = 0
        for k in range(columns):
            h_theta = h_theta + X[i, k] * theta[0, k]
            print("x=",X[i, k], "theta=",theta[0, k], "h_theta=",h_theta),
        summary = summary + (h_theta - y[i, 0])*(h_theta - y[i, 0])  # 计算平方和
        print("y=", y[i, 0], ",htheta=", h_theta,",summary=",summary),
    J=1/(2*m)*summary
    return J

def newAlgorithm(Xdata,Y):
    return

X=np.matrix([[1,1],[1,2],[1,3]])
theta=np.matrix([0,0])
y=np.matrix([[1],[2],[3]])
#print(X[:,1].size)
J=costFunctionJ(X,y,theta)
print(J)

