"""
K-means
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class KMeans:
    def __init__(self, data):
        self.data = np.array(data)
        self.m, self.n = self.data.shape
        self.max_iter = 100

    @staticmethod
    def calc_distance(x1, x2):
        """
        求p = 2 时的闵可夫斯基距离，即欧氏距离
        :param x1:
        :param x2:
        :return:
        """
        return np.linalg.norm(x1 - x2)

    def k_means(self, k):
        """
        聚类——k均值算法
        :param k:
        :return:
        """
        # 初始化簇心向量
        mean_vectors = self.data[np.random.randint(0, len(self.data), k)]

        # 聚类结果
        clusters = {}

        # 根据迭代次数重复k-means聚类过程
        iteration = 0
        while iteration <= self.max_iter:
            iteration = iteration + 1
            # 初始化簇
            for i in range(k):
                clusters[i] = []

            # 分别对每一条数据进行聚类判别
            for j in range(self.m):
                min_dist = np.inf
                min_index = -1
                # 选择此条数据的簇
                for i in range(k):
                    dist = self.calc_distance(self.data[j], mean_vectors[i])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = i
                clusters[min_index].append(j)

            # 更新均值向量
            change_flag = False
            for i in range(len(mean_vectors)):
                data = np.array(self.data[clusters[i]])
                miu = np.sum(data, axis=0) / len(data)

                # 判断均值向量是否改变
                if self.calc_distance(miu, mean_vectors[i]) > 1e-3:
                    mean_vectors[i] = miu
                    change_flag = True

            # 输出均值向量，仅三位小数
            for miu in mean_vectors:
                print('[%.3f, %.3f]' % (miu[0], miu[1]), end=' ')
            print()

            # 均值向量未改变，结束迭代，输出迭代次数
            if not change_flag:
                print(f'共迭代 {iteration} 次')
                break

        # 输出聚类结果
        print(clusters)
        return mean_vectors, clusters

    def plot(self, mean_vectors, clusters):
        """
        绘制数据散点图，并标记聚类结果及均值向量
        :param mean_vectors: 均值向量
        :param clusters: 聚类结果
        :return:
        """
        for key in clusters.keys():
            data = self.data[clusters[key]]
            plt.scatter(data[:, 0], data[:, 1], s=10, label=('类别' + str(key)))
        plt.scatter(mean_vectors[:, 0], mean_vectors[:, 1], s=80, c='r', marker='+')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.legend()
        plt.title(f'k = {len(mean_vectors)}')
        # plt.show()


if __name__ == '__main__':
    iris_data = datasets.load_iris().data[:, :2]

    # 选择三个不同的k值——2，3，4
    k_list = [2, 3, 4]
    plt.figure()
    for k_index in range(len(k_list)):
        print('+' * 60)
        print(f'k = {k_list[k_index]}')

        plt.subplot(len(k_list), 1, k_index + 1)
        clustering_k_means = KMeans(iris_data)
        mean_vectors_i, clusters_i = clustering_k_means.k_means(k_list[k_index])
        clustering_k_means.plot(mean_vectors_i, clusters_i)

    plt.suptitle('k-means')
    plt.show()
