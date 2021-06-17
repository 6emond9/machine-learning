"""
AdaBoost

从网上下载或自己编程实现AdaBoost，以不剪枝决策树为基学习器，在西瓜数据集3.0α上训练一个AdaBoost集成，并与图8.4进行比较
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from numpy import linspace


class AdaBoost:
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    def adaboost(self, T):
        """
        每学到一个学习器，根据其错误率确定两件事。
        1.确定该学习器在总学习器中的权重。正确率越高，权重越大。
        2.调整训练样本的权重。被该学习器误分类的数据提高权重，正确的降低权重，目的是在下一轮中重点关注被误分的数据，以得到更好的效果。
        :param T: 训练轮数
        :return: 字典，包含了T个分类器
        """
        m, n = self.dataset.shape
        # 初始化样本权重，每个样本的初始权重是相同的
        D = np.ones((m,)) / m
        x = self.dataset[:, :-1]
        y = self.dataset[:, -1].reshape(m, 1)  # 数据的类标签
        clf_dict = {}  # 保存每次迭代器的信息的字典

        # print(D.T)
        for i in range(T):
            clf = DecisionTreeClassifier()  # 根据样本权重D从数据集D中训练出分类器ht
            clf.fit(x, y)
            error = 1. - clf.score(x, y, D)  # 第t个分类器的误差

            if error > 0.5:
                break
            elif error == 0.:
                error = 1e-6

            alpha = np.log((1 - error) / error) / 2  # 第t个分类器的权值
            # 更新样本分布
            y_pre = clf.predict(x).reshape(m, 1)
            # print(y_pre)
            a = np.exp(-alpha * y * y_pre)
            Z = np.sum(a)  # 规范化因子，确保D是一个分布
            D = (a / Z).flatten()
            # print(y.shape, y_pre.shape)
            # print(error)
            # print(alpha)
            # print(D.shape)
            # print(clf)

            clf_dict[i] = {}
            clf_dict[i]["alpha"] = alpha
            clf_dict[i]["clf"] = clf

        return clf_dict

    @staticmethod
    def ada_predict(data, clf_dict):
        """
        通过Adaboost得到的总的分类器来进行分类
        :param data: 输入数据
        :param clf_dict: 字典，包含了多个决策树
        :return: 预测值
        """
        score = 0.
        for key in clf_dict.keys():
            pre = clf_dict[key]["clf"].predict(data.reshape(1, -1))
            score += clf_dict[key]["alpha"] * pre  # 加权结合后的集成预测结果

        if score > 0.5:
            y_pre = 1
        else:
            y_pre = 0
        return y_pre

    def calc_ada_acc(self, clf_dict):
        """
        计算集成学习器准确绿率
        :param clf_dict: 字典，包含了多个决策树
        :return: 准确率
        """
        count = 0
        for data in self.dataset:
            x = data[:-1]
            y = data[-1]
            y_pre = self.ada_predict(x, clf_dict)
            if y_pre == y:
                count += 1
        return count / float(self.dataset.shape[0])

    def plot(self, clf_dict):
        """
        绘制数据集及分类信息
        :param clf_dict: 字典，包含了多个决策树
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

        # 分类边界
        for key in clf_dict.keys():
            for index in range(len(clf_dict[key]["clf"].tree_.feature)):
                if clf_dict[key]["clf"].tree_.feature[index] == 0:
                    z1 = np.ones((100,)) * clf_dict[key]["clf"].tree_.threshold[index]
                    plt.plot(z1, y)
                elif clf_dict[key]["clf"].tree_.feature[index] == 1:
                    z2 = np.ones((100,)) * clf_dict[key]["clf"].tree_.threshold[index]
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
        adaBoost = AdaBoost(dataSet)
        clf_dict_t = adaBoost.adaboost(t)
        accuracy = adaBoost.calc_ada_acc(clf_dict_t)
        print('集成学习器（字典）：\n', f"Iter({t}) = {clf_dict_t}")
        print('集成学习器准确率 ', f'Iter({t}) =\t{accuracy}')
        # 绘图函数
        # print(clf_dict_t)
        adaBoost.plot(clf_dict_t)
    plt.show()
