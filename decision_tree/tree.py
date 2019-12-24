#!/usr/bin/python3
import math
import operator
import pickle

def calShannonEnt(dataSet):
	numOfDataSet = len(dataSet)
	labelCount = {}

	for fectVect in dataSet:
		currentLabel = fectVect[-1]
		if currentLabel not in labelCount.keys():
			labelCount[currentLabel] = 0
		labelCount[currentLabel] += 1
	
	shannonEnt = 0.0;
	for key in labelCount:
		prob = labelCount[key] / numOfDataSet
		shannonEnt -= prob * math.log(prob, 2)
	
	return shannonEnt


def createDataSet():
	dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']

	return dataSet, labels


def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for fectVect in dataSet:
		if fectVect[axis] == value:
			reducedFectVect = fectVect[:axis]
			reducedFectVect.extend(fectVect[axis+1:])
			retDataSet.append(reducedFectVect)

	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1

	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0

		for val in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, val)
			prob = len(subDataSet) / len(dataSet)
			newEntropy += prob * calShannonEnt(subDataSet)

		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i

	return bestFeature


def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]


def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	
	if classList.count(classList[0]) == len(classList):
		return classList[0]

	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	
	bestFect = chooseBestFeatureToSplit(dataSet)
	bestFectLabel = labels[bestFect]
	
	myTree = {bestFectLabel:{}}
	del labels[bestFect]
	
	fectVals = [example[bestFect] for example in dataSet]
	uniqueVals = set(fectVals)

	for val in uniqueVals:
		subLabels = labels[:]
		myTree[bestFectLabel][val] = createTree(splitDataSet(dataSet, bestFect, val), subLabels)
	
	return myTree


def classify(myTree, fectLabels, testVect):
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	fectIndex = fectLabels.index(firstStr)
	classLabel = "##"

	for key in secondDict.keys():
		if testVect[fectIndex] == key:
			if type(secondDict[key]).__name__ == "dict":
				classLabel = classify(secondDict[key], fectLabels, testVect)
			else:
				classLabel = secondDict[key]
	
	return classLabel


def storeTree(inputTree, filename):
	fw = open(filename, 'wb')
	fw.write(pickle.dumps(inputTree))
	fw.close()


def grabTree(filename):
	fr = open(filename, 'rb')
	data = pickle.loads(fr.read())
	fr.close()
	return data


dataSet, labels = createDataSet()
print(calShannonEnt(dataSet))


