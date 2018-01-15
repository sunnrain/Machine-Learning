#coding=utf-8
from numpy import * #计算科学包Numpy 模块
import operator  #运算符模块
from os import listdir  #导入函数listdir,可以列出给定目录的文件名
def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels
	
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]  #shape为numpy 的函数，查看矩阵或数组的维数，shape[0]查看行（第一维度）
	diffMat = tile(inX, (dataSetSize,1)) - dataSet   #tile为numpy 函数，将inX横向复制dataSet次，纵向复制1次。
	sqDiffMat = diffMat**2 #平方
	sqDistances = sqDiffMat.sum(axis=1) #求和，axis=1 表示对行求和 
	distances = sqDistances**0.5 #开平方，为欧氏距离
	sortedDistIndicies = distances.argsort() #返回从小到大排序的索引值
	classCount={}
	for i in range(k): 
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #voteIlabel作为字典classCount的key，通过get函数获取相应的value,默认为0
		#各个标签出现的频率，用字典的value表示
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
	#sorted(iterable,key,reverse)，sorted一共有iterable,key,reverse这三个参数。
	#其中iterable表示可以迭代的对象，例如可以是dict.items(), key是一个函数，用来选取参与比较的元素，reverse则是用来指定排序是倒序还是顺序
	#key = operator.itemgetter(1)为一个函数，对iterable对象每个作用，reverse=true则是倒序，reverse=false时则是顺序(递增）
	#按照字典classCount的value(频次)进行排序，为倒序（从高到低）,返回一个由元组组成的list
	return sortedClassCount[0][0]#第一个元组的第一个值，即频次最高的key
	
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()#readlines()自动将文件内容分析成一个行的列表
	numberOfLines = len(arrayOLines)
	returnMat = zeros([numberOfLines, 3]) #创建返回矩阵,zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()#strip 同时去掉左右两边的空格
		listFromLine = line.split('\t')#split指定分隔符对数据切片
		returnMat[index,:] = listFromLine[0:3]#选取前3个元素（特征）存储在返回矩阵中
		classLabelVector.append(int(listFromLine[-1]))
		#必须明确地通知解释器，告诉它列表中存储的元素值为整型，否则Python语言会将这些元素当作字符串处理。
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)#存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	#注意这是具体特征值相除，而对于某些数值处理软件包，/可能意味着矩阵除法，
	#但在NumPy库中，矩阵除法需要使用函数linalg.solve(matA,matB)。
	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classifier came back with: %d, the real answer is:%d" % (classifierResult, datingLabels[i]))
		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))						

#完整的约会网站预测：给定一个人，判断时候适合约会
def classifyperson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input("percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)
	print('You will probably like this person:', resultList[classifierResult - 1])

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect
	
def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')  #获取目录内容,为一个list
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]#解析文件,把txt去掉
		classNumStr = int(fileStr.split('_')[0])#解析文件名，识别类别（每个数字代表一类，每类有多个实例）
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print("\nthe total number of error is: %d" % errorCount)
	print("\nthe total error rate is：%f" % (errorCount/float(mTest)))
		
		
	