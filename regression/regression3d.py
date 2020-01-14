#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateDataSet(num=100, maxnum=50):
    # y^ = 5 + x1 + 6*x2
    rd = np.random.randn(num)
    x1 = np.random.rand(num) * maxnum
    x2 = np.random.rand(num) * maxnum

    y = (rd + 5) + x1 + 6 * x2

    fw = open('data3d.txt', 'w')
    for i in range(num):
        fw.write(str(x1[i]) + '\t' + str(x2[i]) + '\t' + str(y[i]) + '\n')
    fw.close()

def loadDataSet():
    fr = open("data3d.txt")
    dataMat = []
    arrayOfLines = fr.readlines()
    for line in arrayOfLines:
        curLine = list(map(float, line.strip().split('\t')))
        dataMat.append(curLine)

    return dataMat

def reg3d(dataSet):
    dataSet = np.array(dataSet)
    m = len(dataSet)
    X = np.ones((3, m))
    X[1, :] = dataSet[:, 0]
    X[2, :] = dataSet[:, 1]
    # print('x=', X)

    y = dataSet[:, 2]
    # print('y =', y)

    X = np.matrix(X)
    y = np.matrix(y)

    XXt = X * X.T
    w = y * X.T * XXt.I
    w = w.tolist()[0]

    print('w =', w)
    return w
    
def loss(dataSet, ws):
    dataSet = np.array(dataSet)
    m = len(dataSet)
    X = np.ones((3, m))
    X[1, :] = dataSet[:, 0]
    X[2, :] = dataSet[:, 1]
    X = np.matrix(X)
    y = dataSet[:, 2]
    y = np.matrix(y)
    ws = np.matrix(ws)

    error = np.array((ws * X - y).tolist()[0])
    
    error_sq = error ** 2
    loss = np.sum(error_sq)

    return loss / m
    

def plotDataSet(dataSet, maxnum=50):
    dataSet = np.array(dataSet)
    
    ws = reg3d(dataSet)
    X1 = np.linspace(0, maxnum, 20)
    X2 = np.linspace(0, maxnum, 20)
    X1, X2 = np.meshgrid(X1, X2)
    y = ws[0] + ws[1] * X1 + ws[2] * X2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, y, alpha=0.8)
    ax.scatter(dataSet[:, 0], dataSet[:, 1], dataSet[:, 2], c='r', marker='^')

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")

    fig.show()
    ls = loss(dataSet, ws)
    print("loss = ", ls)
    input("Press to exit.")



if __name__ == "__main__":
    # generateDataSet()
    dataSet = loadDataSet()
    plotDataSet(dataSet)
