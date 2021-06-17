"""
kNN——k近邻
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, data_train, labels_train):
        self.data_train = data_train
        self.labels_train = labels_train

    @staticmethod
    def euclidean_distance(x1, x2):
        """
        欧氏距离
        :param x1:
        :param x2:
        :return:
        """
        dis = np.sqrt(np.sum((x1 - x2) ** 2))
        return dis

    def predict(self, data_test, k):
        """
        对输入进行kNN预测
        :param data_test: 输入
        :param k: kNN——k
        :return:
        """
        labels_pre = []  # 预测结果
        for data_test_i in data_test:
            statistics = np.inf * np.ones((k, 2))  # 定义数组，用于统计k个数据点中各个类别的鸢尾花出现的次数
            for data_train_i, labels_test_i in zip(self.data_train, self.labels_train):
                dist = self.euclidean_distance(data_train_i, data_test_i)
                # 选择前k小的k条数据
                if dist < statistics[k - 1, 0]:
                    statistics[k - 1, 0] = dist  # 距离
                    statistics[k - 1, 1] = labels_test_i  # 标签
                # 排序
                statistics_tmp = statistics[:, ::-1].T  # 先逆序再转置
                sort_index = np.lexsort(statistics_tmp)  # np.lexsort()，以列为整体性进行排序
                # 重点是：从最后一行开始比较大小，正因为从最后一行开始排序，前面才用逆序
                statistics = statistics[sort_index]  # 拿着新的序列号，调整下原数组就是我们要的结果了
            # print(statistics)

            # 统计k条数据各标签出现的数量
            labels_count = {}
            for labels_i in statistics[:, 1]:
                labels_i = int(labels_i)
                if labels_i in labels_count:
                    labels_count[labels_i] = labels_count[labels_i] + 1
                else:
                    labels_count[labels_i] = 1

            # 选择出现次数最多的标签作为预测结果
            labels_pre_i = labels_count[list(labels_count.keys())[0]]
            max_count = 0
            for labels_i in labels_count.keys():
                if labels_count[labels_i] > max_count:
                    labels_pre_i = labels_i

            labels_pre.append(labels_pre_i)
        labels_pre = np.array(labels_pre)
        return labels_pre


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    iris_data = iris_dataset.data
    iris_labels = iris_dataset.target

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.3, random_state=30)

    kNN = KNN(X_train, y_train)
    # 选择不同的k
    for k_i in range(1, 6):
        print("+" * 30)
        y_pre = kNN.predict(X_test, k_i)
        accuracy = np.mean([y_pre == y_test])
        print("k=%d accuracy=%.2f%%" % (k_i, accuracy * 100))
        # print(y_test)
        # print(y_pre)
