from math import log

#计算熵
def calcShannonEnt(dataSet):
	numTotal = len(dataSet)
	labelCount = {} 
	for item in dataSet:
		currentLabel = item[-1]
		if currentLabel not in labelCount:
			labelCount[currentLabel] = 0
		labelCount[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCount:
		prob = float(labelCount[key])/numTotal
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

def splitDataSetByFeatureIndex(dataSet,featureIndex,value):
	subDataSet = []
	for item in dataSet:
		if(item[featureIndex] == value):
			tempDataSet = item[:featureIndex]
			tempDataSet.extend(item[featureIndex+1:])
			subDataSet.append(tempDataSet)
	return subDataSet


#创建数据集
def createDataSet():
	dataSet = [
		[1,1,'yes'],
		[1,1,'yes'],
		[1,0,'no'],
		[0,1,'no'],
		[0,1,'no'],
	]
	labels = ['no surfacing','flippers']
	return dataSet,labels

myData,labels=createDataSet()
# print(calcShannonEnt(myData))
# print(splitDataSetByFeatureIndex(myData,0,1));
dataSetofFirstFeature = splitDataSetByFeatureIndex(myData,1,0);
print(calcShannonEnt(dataSetofFirstFeature))
