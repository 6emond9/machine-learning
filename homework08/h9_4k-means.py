"""
k均值

试编程实现k均值算法，设置三组不同的k值、三组不同初始中心点，在西瓜数据集4.0上进行实验比较，并讨论什么样的初始中心有利于取得好结果。
"""
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, dataset):
        self.dataset = np.array(dataset)
        self.features = ['密度', '含糖率']
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
        # 随机生成初始均值向量
        mean_vectors = self.dataset[np.random.randint(0, len(self.dataset), k)]  # 随机选取k个初始样本
        # print(mean_vectors)

        # 聚类结果
        clusters = {}

        # 设置最大迭代次数
        for iter_i in range(1, self.max_iter):
            # 对聚类结果进行清空复位
            for i in range(len(mean_vectors)):
                clusters[i] = []

            # 分别对每一条数据进行聚类判别
            for j in range(len(self.dataset)):
                min_dist = np.inf
                min_index = -1
                # 选择此条数据的簇
                for i in range(len(mean_vectors)):
                    dist = self.calc_distance(self.dataset[j], mean_vectors[i])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = i
                clusters[min_index].append(j)

            # 更新均值向量
            change_flag = False
            for i in range(len(mean_vectors)):
                data = np.array(self.dataset[clusters[i]])
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
                print(f'共迭代 {iter_i} 次')
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
            data = self.dataset[clusters[key]]
            plt.scatter(data[:, 0], data[:, 1], label=key)
        plt.scatter(mean_vectors[:, 0], mean_vectors[:, 1], s=80, c='r', marker='+')
        plt.xlabel(self.features[0])
        plt.ylabel(self.features[1])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.legend()
        # plt.show()


if __name__ == '__main__':
    dataSet = [
        [0.697, 0.460],
        [0.774, 0.376],
        [0.634, 0.264],
        [0.608, 0.318],
        [0.556, 0.215],
        [0.403, 0.237],
        [0.481, 0.149],
        [0.437, 0.211],
        [0.666, 0.091],
        [0.243, 0.267],
        [0.245, 0.057],
        [0.343, 0.099],
        [0.639, 0.161],
        [0.657, 0.198],
        [0.360, 0.370],
        [0.593, 0.042],
        [0.719, 0.103],
        [0.359, 0.188],
        [0.339, 0.241],
        [0.282, 0.257],
        [0.748, 0.232],
        [0.714, 0.346],
        [0.483, 0.312],
        [0.478, 0.437],
        [0.525, 0.369],
        [0.751, 0.489],
        [0.532, 0.472],
        [0.473, 0.376],
        [0.725, 0.445],
        [0.446, 0.459],
    ]

    # 选择三个不同的k值——2，3，4
    for k_i in [2, 3, 4]:
        print('+' * 60)
        print(f'k = {k_i}')
        plt.figure(k_i)
        # 设置三组不同的初始中心点——随机生成
        for rand_i in range(3):
            print('*' * 40)
            print(f'clustering {rand_i + 1}')
            plt.subplot(3, 1, rand_i + 1)
            clustering_k_means = KMeans(dataSet)
            mean_vectors_i, clusters_i = clustering_k_means.k_means(k_i)
            clustering_k_means.plot(mean_vectors_i, clusters_i)
        plt.suptitle(f'k-means(k = {k_i})')
        plt.show()
