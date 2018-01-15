#coding=utf-8
from numpy import * 
import kNN
from os import listdir
group, labels = kNN.createDataSet()
print(group, labels)
print(kNN.classify0([0, 0], group, labels, 3))
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print (datingDataMat)
print (datingLabels[0:20])
# *import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# #add_subplot(mnp)添加子轴、图。subplot（m,n,p）或者subplot（mnp）此函数最常用：
# #subplot是将多个图画到一个平面上的工具。其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，
# #如果第一个数字是2就是表示2行图。
# #p是指你现在要把曲线画到figure中哪个图上，最后一个如果是1表示是从左到右第一个位置。  
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 
# 15.0*array(datingLabels), 15.0*array(datingLabels),)
# #以第二列和第三列为x,y轴画出散列点，给予不同的颜色和大小  
# #scatter（x,y,s=1,c="g",marker="s",linewidths=0）  
# #s:散列点的大小,c:散列点的颜色，marker：形状，linewidths：边框宽度
# #plt.show()

norMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(norMat)
print(ranges)
print(minVals)
kNN.datingClassTest()
# kNN.classifyperson()
testVector = kNN.img2vector('testDigits/0_13.txt')
print(testVector[0, 0:31])
print(testVector[0, 32:63])
# trainingFileList = listdir('trainingDigits')
# print(trainingFileList)
# shape(trainingFileList)
kNN.handwritingClassTest()