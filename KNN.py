# -*- coding:utf-8 -*-
#! /usr/bin/python3

'''
KNN 算法伪代码：
	对未知数据依次进行：
	1.计算已知类别数据中的点与当前点的距离
	2.按照距离递增依次排序
	3.选取与当前距离最小的K个点
	4.确定K个点所在类别的最高频率
	5.返回前K个点中出现频率最高的类别作为当前预测点的最终类别

KNN 算法场景：
	有监督，数值型

KNN 算法参数：
	K的取值
	距离的计算方式：欧式距离、余弦值、相关度、曼哈顿距离

KNN 算法优点：
	简单，易理解，可解释性强，健壮性高

KNN 算法缺点：
	需要一一计算未知实利与已知实例的情况
	样本分布不均匀时（与未知实例距离近的样本数据很少，距离远的非常多），误差较大

KNN 算法改进：
	考虑距离，根据距离加权
'''

import numpy as np
import matplotlib.pyplot as plt

class KNN(object):

	def loadInitDataSet(self):

		test_data = np.array([
			[1.0, 1.1],
			[1.0, 1.0],
			[0.0, 0.0],
			[0.0, 0.1],
			[2,2],
		])

		test_labels = ['A','A','B','B','A']
		return test_data, test_labels

	def knnClassifier(self, unknow_data, test_data, test_labels, k=3):
		rows, cols = np.shape(test_data)
		# unknow_data_matrix = np.tile(unknow_data, (rows, 1))
		# distance_matrix = test_data - unknow_data_matrix
		distance_matrix = test_data - unknow_data
		square_distance_matrix = distance_matrix**2
		# sum() 参数为空：整体求和; axis=0:对列求和；axis=1:对行求和
		# 当前坐标与已知坐标距离矩阵
		sum_square_distance_matrix = square_distance_matrix.sum(axis=1)
		# print(sum_square_distance_matrix)
		index_distance = sum_square_distance_matrix.argsort()
		# print(index_distance)
		label_num = {}
		for item in index_distance[:k]:
			current_label = test_labels[item]

			label_num[current_label] = label_num.get(current_label, 0) + 1
			'''
			if current_label in label_num.keys():
				label_num[current_label] += 1
			else:
				label_num[current_label] = 1
			'''
		predict_result = sorted(label_num.items(), key = lambda x:x[1], reverse=True)

		return predict_result



if __name__ == '__main__':
	knn = KNN()
	test_data, test_labels = knn.loadInitDataSet()
	unknow_data = [1, 2]
	K = 3
	predict_result = knn.knnClassifier(unknow_data, test_data, test_labels,k=K)

	plt.axis([0,4,0,4])
	for item in range(len(test_data)):
		plt.text(test_data[item][0], test_data[item][1],test_labels[item])

	print('分类情况：',predict_result)
	recommand_label = predict_result[0][0]
	print('推荐分类：',recommand_label)
	plt.text(unknow_data[0], unknow_data[1],recommand_label, color='red')
	plt.show()


