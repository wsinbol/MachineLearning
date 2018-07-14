'''
KNN算法伪代码：
	对未知数据依次进行：
	1.计算已知类别数据中的点与当前点的距离
	2.按照距离递增依次排序
	3.选取与当前距离最小的K个点
	4.确定K个点所在类别的最高频率
	5.返回前K个点中出现频率最高的类别作为当前预测点的最终类别
'''

import numpy as np

def createDataSet():
	data = np.array([
		[1.0,1.1],
		[1.0,1.0],
		[0.0,0.0],
		[0.0,0.1],
		])
	labels = ['A','A','B','B']
	return data,labels

def knn_classify(unknow_data, test_data, k, test_labels):
	row = test_data.shape[0] # 获取已知数组的行数
	unknow_data_array = np.tile(unknow_data, (row, 1)) # 构造同已知数组相同的结构
	distance = unknow_data_array - test_data # 求差
	print(distance)
	# print(distance.sum()) # 参数为空整体求和
	# print(distance.sum(axis = 0)) # 参数为0列求和
	distance_sum = distance.sum(axis = 1) # 参数为1行求和
	distance_list = distance_sum**2
	distance_index = distance_list.argsort() # 返回升序排列的索引

	result = {}
	for i in range(k):
		# top_label.append(test_labels[distance_index[i]]) # 
		result[test_labels[distance_index[i]]] = result.get(test_labels[distance_index[i]], 0) + 1
	top_label = sorted(result.items(), key = lambda x:x[1], reverse = True)
	return top_label[0][0]


def main():
	test_data,test_labels = createDataSet()
	unknow_data = [0,1]
	get_label = knn_classify(unknow_data, test_data,3,test_labels)
	print(get_label)


if __name__ == '__main__':
	main()
