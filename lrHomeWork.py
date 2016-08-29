import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


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
    print(dataSet.head(5))
    return dataSet


#均方误差
def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


import warnings # 用来忽略seaborn绘图库产生的warnings
warnings.filterwarnings("ignore")

def main():
    dataset=loadDataSet()
    paramX=dataset.iloc[:,0:13]
    paramY=dataset.iloc[:,13]

    print(paramX)
    print(paramY)
    X_train, X_test, y_train, y_test = train_test_split(paramX, paramY, random_state=1)
    linreg = linear_model.LinearRegression(normalize=True)
    linreg.fit(X_train, y_train)
    predict_y = linreg.predict(X_test)
    score=linreg.score(X_test,y_test)
    print("predict_y",predict_y)
    print("score", score)
    print("均方误差:",MSE(y_test,predict_y))


    print(linreg.intercept_)
    print (linreg.coef_)



    # predict_y = linreg.predict(X_test)
    # print("predict_y----------",predict_y)
if __name__ == '__main__':
    main()
