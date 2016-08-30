import pandas as pd
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# CRIM犯罪率
# ZN超过25000地区的居住面积所占比例
# INDUS非零售业的商业用地比例
# CHAS是否被Charles河流穿过（是，取值1；否，取值0
# NOX一氧化氮含量RM房子的平均屋子数
# AGE早于1940年建立的的住宅比例
# DIS距离五个上班区域的加权平均距离RAD反映到达放射形状的高速路的能力的指标
# TAX 每10000美元的财产税
# PTRATIO小学生-老师的比例
# B反映黑人比例的指标，黑人比例越靠近0.63越小
# LSTAT 地位低的人群比例

def loadDataSet():
    data = []
    labelMat = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT', 'MEDV']
    fr = open('./housing.data')

    print("Data Loading...")
    for line in fr.readlines():
        lineArr = line.strip().split()

        data.append([float(lineArr[0]), float(lineArr[1]),
                      float(lineArr[2]), float(lineArr[3]),
                      float(lineArr[4]), float(lineArr[5]),
                      float(lineArr[6]), float(lineArr[7]),
                      float(lineArr[8]), float(lineArr[9]),
                      float(lineArr[10]), float(lineArr[11]),
                      float(lineArr[12]),float(lineArr[13])])

    print("Data was load.")
    dataSet=pd.DataFrame(data,columns=labelMat)
    # print(dataSet.head(5))
    return dataSet


#均方误差
def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


import warnings # 用来忽略seaborn绘图库产生的warnings
warnings.filterwarnings("ignore")

def loadboston():
    boston = datasets.load_boston()
    print(boston.feature_names)

def main():
    loadboston()
    dataset=loadDataSet()
    paramX=dataset.iloc[:,0:13]
    paramY=dataset.iloc[:,13]

    X_train, X_test, y_train, y_test = train_test_split(paramX, paramY, random_state=1)
    linreg = linear_model.LinearRegression(normalize=True)
    linreg.fit(X_train, y_train)
    predict_y = linreg.predict(X_test) #theta0
    coef=linreg.coef_
    intercept=linreg.intercept_
    print("coef",coef)
    print("intercept=",intercept)
    score=linreg.score(X_test,y_test)
    print("score", score)
    print("均方误差:",MSE(y_test,predict_y))

    # mark=['+','d','*','s','x','.','8','h','o','p','^',',','4','6']
    sns.barplot(x=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT'],y=coef)

    # 每个特征与房价之间的关系的散列图
    # sns.set(style="ticks", color_codes=True)
    # sns.pairplot(dataset, x_vars=['CRIM', 'ZN', 'INDUS', 'CHAS'], y_vars='MEDV', size=8, aspect=0.8, kind='reg')
    # sns.pairplot(dataset, x_vars=['NOX', 'RM', 'AGE'], y_vars='MEDV', size=8, aspect=0.8,kind='reg')
    # sns.pairplot(dataset, x_vars=['DIS', 'RAD', 'TAX'], y_vars='MEDV', size=8, aspect=0.8, kind='reg')
    # sns.pairplot(dataset, x_vars=['PTRATIO', 'B', 'LSTAT'], y_vars='MEDV', size=8, aspect=0.8,kind='reg',markers="+")

    # cols = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange',
    #         'darkgreen','gold','purple','lightcoral','lightskyblue','lightsalmon','ivory']
    # for i,f in enumerate(dataset.iloc[:,0:13]):
    #     x=dataset.iloc[:,i]
    #     y=dataset.iloc[:,13]
    #     p=plt.subplot(4,4,1+i)
    #     plt.sca(p)
    #     plt.xlabel('MEDV')
    #     plt.ylabel(f)
    #     plt.title(f)
    #     plt.scatter(y,x,color=cols[i],label=f)
    #
    #
    #     print(f,"i=",i)

    plt.show()




    # predict_y = linreg.predict(X_test)
    # print("predict_y----------",predict_y)
if __name__ == '__main__':
    main()
