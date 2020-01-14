#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    fr = open("data_poly.csv")
    arrayOfLines = fr.readlines()

    dataSet = []
    for line in arrayOfLines:
        curLine = list(map(float, line.strip().split(',')))
        dataSet.append(curLine)

    return dataSet


def poly_regression(dataSet):
    dataSet = np.array(dataSet)
    m = len(dataSet)
    X = np.ones((4, m))
    X[1, :] = dataSet[:, 0]
    X[2, :] = dataSet[:, 0] ** 2
    X[3, :] = dataSet[:, 0] ** 3

    X = np.matrix(X)
    y = np.matrix(dataSet[:, 1])

    XXt = X * X.T
    weights = y * X.T * XXt.I

    y_hat = weights * X
    error = y_hat - y
    error_ary = np.array(error.tolist()[0])
    square_error_ary = error_ary ** 2
    los = np.sum(square_error_ary) / m

    weights = weights.tolist()[0]

    print("weights = ", weights)
    return weights, los

    


def plotDataSet(dataSet):
    dataSet = np.array(dataSet)

    ws, los = poly_regression(dataSet)
    print("loss = ", los)

    min_x = min(dataSet[:, 0])
    max_x = max(dataSet[:, 0])

    x = np.linspace(min_x, max_x, 100)
    y = ws[0] + ws[1] * x + ws[2] * x**2 + ws[3] * x**3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1])
    ax.plot(x, y, c='orange')

    fig.show()
    input("Press to exit.")


if __name__ == "__main__":
    dataSet = loadDataSet()
    plotDataSet(dataSet)
    
