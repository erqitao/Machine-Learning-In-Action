#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0 / (1+np.exp(-inX))

def loadDataSet():
    fr = open("testSet.txt")
    rtnDataSet = []
    rtnLabel = []

    arrayOfLines = fr.readlines()
    for line in arrayOfLines:
        listOfLine = line.strip().split()
        rtnDataSet.append([1, float(listOfLine[0]), float(listOfLine[1])])
        rtnLabel.append(int(listOfLine[2]))

    return rtnDataSet, rtnLabel

def gradientAscent(dataSet, label):
    dataSetMat = np.mat(dataSet)
    labelMat = np.mat(label).transpose()
    m, n = np.shape(dataSetMat)
    maxCycle = 500
    alpha = 0.001
    weights = np.ones((n, 1))
    weights = np.mat(weights)

    for k in range(maxCycle):
        h = sigmoid(dataSetMat * weights)
        error = labelMat - h
        weights += alpha * dataSetMat.transpose() * error

    return weights

def stoGradientAscent0(dataSet, label):
    dataSet = np.array(dataSet)
    m, n = np.shape(dataSet)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataSet[i] * weights))
        error = label[i] - h
        weights += alpha * error * dataSet[i]

    return weights

def stoGradientAscent1(dataSet, label, numItem=150):
    dataSet = np.array(dataSet)
    m, n = np.shape(dataSet)
    weights = np.ones(n)

    for j in range(numItem):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4.0/(1+i+j) + 0.01
            randomIndex = int(np.random.uniform(0, len(dataIndex)))
            randIndex = dataIndex[randomIndex] # this is the real random index of dataSet
            h = sigmoid(np.sum(dataSet[randIndex] * weights))
            error = label[randIndex] - h
            tmp = weights + dataSet[randIndex]
            tmp = alpha * error * tmp

            weights = weights + alpha * error * dataSet[randIndex]
            del dataIndex[randomIndex]

    return weights

def classify(inX, weights):
    p = sigmoid(sum(inX * weights))
    if p >= 0.5:
        return 1
    else:
        return 0

def test():
    frTrain = open("HorseColicTraining.txt")
    frTest = open("HorseColicTest.txt")
    
    trainingSet = []
    trainingSetLabels = []

    trainingData = frTrain.readlines()
    for line in trainingData:
        listOfLine = line.strip().split()
        lineAry = []
        for i in range(21):
            lineAry.append(float(listOfLine[i]))
        trainingSet.append(lineAry)
        trainingSetLabels.append(int(float(listOfLine[21])))

    trainingWeights = stoGradientAscent1(trainingSet, trainingSetLabels)

    testData = frTest.readlines()
    numOfTestData = len(testData)
    errorCount = 0

    for line in testData:
        listOfLine = line.strip().split()
        lineAry = []
        for i in range(21):
            lineAry.append(float(listOfLine[i]))
        
        if classify(np.array(lineAry), trainingWeights) != int(float(listOfLine[21])):
                errorCount += 1
    print("error rate: %.3f" % (errorCount / numOfTestData))
    return (errorCount / numOfTestData)


def multiTest():
    numOfTests = 10
    errorRate = 0.0
    for i in range(numOfTests):
        errorRate += test()
    print("After %d times iterations, the average error rate is %.3f" % (numOfTests, errorRate / numOfTests))

def plotBestFit(wei):
    weights = np.array(wei)
    print(weights)
    dataSet, label = loadDataSet()
    dataSetAry = np.array(dataSet)
    m = np.shape(dataSetAry)[0]
    
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    
    for i in range(m):
        if label[i] == 1:
            xcord1.append(dataSet[i][1])
            ycord1.append(dataSet[i][2])
        else:
            xcord2.append(dataSet[i][1])
            ycord2.append(dataSet[i][2])


    x = np.linspace(-5, 5, 100)
    y = (-weights[0] - weights[1]*x) / weights[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue', marker='o')
    ax.plot(x, y)
    plt.show()

if __name__ == "__main__":
    #dataSet, label = loadDataSet()
    # weights = gradientAscent(dataSet, label)
    #weights = stoGradientAscent1(dataSet, label)
    #plotBestFit(weights)
    multiTest()
