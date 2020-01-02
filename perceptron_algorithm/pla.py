#!/usr/bin/python
# Filename: Perceptron Linear Algorithm (PLA)
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    fr = open('testSet.txt')
    arrayOfLines = fr.readlines()
    dataSet = []
    labels = []

    for line in arrayOfLines:
        arrOfLine = line.strip().split('\t')
        dataSet.append([1.0, float(arrOfLine[0]), float(arrOfLine[1])])
        labels.append(int(arrOfLine[-1]))

    return dataSet, labels

def splitDataSetByValue(dataSet, labels, value):
    rtnDataSet = []
    m = len(dataSet)

    for i in range(m):
        if labels[i] == value:
            rtnDataSet.append(dataSet[i])

    return rtnDataSet

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def errorPoint(dataSet, labels, weights):
    m = len(dataSet)
    for i in range(m):
        if (labels[i] * np.dot(dataSet[i], weights) ) <= 0:
            return i

    return -1

def plaTraining(dataSet, labels):
    weights = np.array([1, 1, 1])
    dataSet = np.array(dataSet)
    alpha = 0.01
    
    p = errorPoint(dataSet, labels, weights)
    while p != -1:
        weights = weights + alpha * labels[p] * dataSet[p]
        p = errorPoint(dataSet, labels, weights)

    return weights

def plotFigure(weights):
    dataSet, labels = loadDataSet()
    cc = set(labels)
    fig = plt.figure()
    axis = fig.add_subplot(111)

    for i in cc:
        subDataSet = splitDataSetByValue(dataSet, labels, i)
        subDataSet = np.array(subDataSet)
        axis.scatter(subDataSet[:,1], subDataSet[:,2], label="class = %d" % i)
    
    dataSet = np.array(dataSet)
    min_x = np.min(dataSet[:,1]) - 1
    max_x = np.max(dataSet[:, 1]) + 1
    x = np.linspace(min_x, max_x, 20)
    y = -(weights[0] + weights[1] * x) / weights[2]
    axis.plot(x, y, 'r')
    
    fig.legend()
    fig.show()


if __name__ == "__main__":
    dataSet, labels = loadDataSet()
    weights = plaTraining(dataSet, labels)
    print(weights)
    plotFigure(weights)
    input("Press to exit.")

