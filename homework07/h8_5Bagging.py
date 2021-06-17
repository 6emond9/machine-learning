"""
Bagging

试编程实现Bagging，以决策树桩为基学习器，在西瓜数据集3.0α上训练一个Bagging集成，并与图8.6进行比较
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from numpy import linspace


class Bagging:
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    def bagging(self, T):
        """
        Bagging算法
        :param T: 训练轮数
        :return: 字典，包含集成学习器预测准确率及决策树
        """
        m, n = self.dataset.shape
        clf_dict = {}  # 保存集成学习器的信息的字典

        max_acc = 0.
        for i in range(T):
            # 自助采样
            x, y = [], []
            for j in range(m):
                rand_i = np.random.randint(0, m)
                x.append(self.dataset[rand_i, :-1])
                y.append(self.dataset[rand_i, -1])

            clf = DecisionTreeClassifier(max_depth=1)  # 分类器ht
            clf.fit(x, y)
            acc = clf.score(x, y)  # 第t个分类器的误差

            # 选取集成学习器
            if acc > max_acc:
                max_acc = acc
                clf_dict["acc"] = max_acc
                clf_dict["clf"] = clf

        return clf_dict

    def plot(self, clf):
        """
        绘制数据集
        :param clf: 集成学习器
        :return:
        """
        x_pos, x_neg = [], []
        y_pos, y_neg = [], []
        for data in self.dataset:
            if data[-1] > 0:
                x_pos.append(data[0])
                y_pos.append(data[1])
            else:
                x_neg.append(data[0])
                y_neg.append(data[1])

        x = linspace(0 + 0.05, 0.8 + 0.05, 100)
        y = linspace(0 + 0.05, 0.6 + 0.05, 100)

        # 分类线
        for index in range(len(clf.tree_.feature)):
            if clf.tree_.feature[index] == 0:
                z1 = np.ones((100,)) * clf.tree_.threshold[index]
                plt.plot(z1, y)
            elif clf.tree_.feature[index] == 1:
                z2 = np.ones((100,)) * clf.tree_.threshold[index]
                plt.plot(x, z2)

        # 数据标记
        plt.scatter(x_pos, y_pos, marker='+', label='好瓜', color='b')
        plt.scatter(x_neg, y_neg, marker='_', label='坏瓜', color='r')

        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.xlim(0, 0.9)  # 设置x轴范围
        plt.ylim(0, 0.7)  # 设置y轴范围
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.legend(loc='upper left')


if __name__ == '__main__':
    # [密度，含糖量，好瓜]
    dataSet = [
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.430, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]
    ]

    plt.figure()
    i = 1
    for t in [3, 5, 11]:  # 学习器的数量
        plt.subplot(3, 1, i)
        i = i + 1
        bagging = Bagging(dataSet)
        clf_t = bagging.bagging(t)
        accuracy = clf_t["acc"]
        print(f'集成学习器 Iter({t})（字典） =\t{clf_t}')
        print(f'集成学习器 Iter({t}) 准确率 =\t{accuracy}')
        # 绘图函数
        # print(clf_dict_t)
        bagging.plot(clf_t["clf"])
    plt.show()
