#!/usr/bin/python3
# -*- coding:utf8 -*-
import numpy as np
import string
import re

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVect = [0, 1, 0, 1, 0, 1]

    return postingList, classVect


def createVocbList(dataSet):
    vocbSet = set()
    for document in dataSet:
        vocbSet = vocbSet | set(document)

    return list(vocbSet)


def setOfWords2Vec(vocbList, inputSet):
    returnVect = [0] * len(vocbList)
    for word in inputSet:
        if word in vocbList:
            returnVect[vocbList.index(word)] = 1

    return returnVect


def bagOfWords2Vec(vocbList, inputSet):
    returnVect = [0] * len(vocbList)
    for word in inputSet:
        if word in vocbList:
            returnVect[vocbList.index(word)] += 1

    return returnVect


def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / numTrainDocs

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = numWords
    p1Denom = numWords

    for i in range(numTrainDocs):
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        else:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    
    print("Words count of class 0(Total = %d) \n %s" % (p0Denom, p0Num))
    print("Words count of class 1(Total = %d) \n %s" % (p1Denom, p1Num))

    p0Vect = np.log(p0Num / p0Denom)
    p1Vect = np.log(p1Num / p1Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vect2Classify, p0Vect, p1Vect, pClass1):
    p0 = np.sum(p0Vect * vect2Classify) + (1-pClass1)
    p1 = np.sum(p1Vect * vect2Classify) + pClass1

    if p0 > p1:
        return 0
    else:
        return 1


def testingNB():
    dataSet, labels = loadDataSet()
    vocabulary = createVocbList(dataSet) 
    trainMatrix = []
    for x in dataSet:
        trainMatrix.append(setOfWords2Vec(vocabulary, x))
    
    p0Vect, p1Vect, pab = trainNB(trainMatrix, labels)
    testEntry = ['I', 'love', 'dog']
    testVect = setOfWords2Vec(vocabulary, testEntry)

    ans = classifyNB(testVect, p0Vect, p1Vect, pab)
    print(testEntry, "classifier result :", ans)

    testEntry = ['stupid', 'garbage']
    testVect = setOfWords2Vec(vocabulary, testEntry)
    ans = classifyNB(testVect, p0Vect, p1Vect, pab)
    print(testEntry, "classifier result :", ans)


def textParse(filename):
    # print("filename =", filename)
    fr = open(filename)
    data = fr.readlines()
    rtnWords = []

    r = "['\"!#$%&'()*+,-./:;<=>?@[\\]^`{|}~]+"

    for line in data:
        line = line.strip()
        line = re.sub(r, ' ', line)
        for word in line.split():
            if len(word) > 2:
                rtnWords.append(word.lower())
        
    return rtnWords


def spamTest():
    docList =[]
    classList = []
    fullText = []

    for i in range(1, 26):
        wordList = textParse("spam/%d.txt" % i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse("ham/%d.txt" % i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    vocabList = createVocbList(docList)
    # print(docList)
    # print(classList)
    # print("vocabList:\n", vocabList)

    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randomIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(randomIndex)
        del trainingSet[randomIndex]

    trainMat = []
    trainClasses = []
    
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # print("trainMat:\n", trainMat)
    # print("trainClasses:\n", trainClasses)
    p0Num, p1Num, pab = trainNB(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for i in testSet:
        testVect = setOfWords2Vec(vocabList, docList[i])
        if classifyNB(testVect, p0Num, p1Num, pab) != classList[i]:
            errorCount += 1
            print("Error!")
            print("the real class is", classList[i])
            print("file id:", i)
            print(docList[i])
    print("the error rate is %f" % (errorCount / len(testSet)))


if __name__ == "__main__":
    #testingNB()
    spamTest()
