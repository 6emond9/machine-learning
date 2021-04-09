import numpy as np
import matplotlib.pyplot as plt


class LDA:
    # 3.0α西瓜数据集
    file_name = 'db/3.0xigua.txt'

    def load_dataset(self):
        """
        获取3.0α西瓜数据集
        :return: (feature array, label array)
        """
        dataset = np.loadtxt(self.file_name, delimiter=",", dtype=float)
        # 略过编号列
        data_array = dataset[:, 1:-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    @staticmethod
    def lda(data_array, label_array):
        """
        LDA 线性判别分析
        :param data_array: input feature array with shape (m, n)
        :param label_array: the label of data set with shape (m, 1)
        :return: parameter w
        """
        # 负类
        data_array0 = data_array[label_array < 0.5]
        # 正类
        data_array1 = data_array[label_array >= 0.5]

        # 均值
        mean0 = np.mean(data_array0, axis=0).reshape((-1, 1))
        mean1 = np.mean(data_array1, axis=0).reshape((-1, 1))
        # 协方差矩阵
        cov0 = np.cov(data_array0, rowvar=False)
        cov1 = np.cov(data_array1, rowvar=False)
        # Sw
        sw = np.mat(cov0 + cov1)
        w = np.dot(sw.I, mean0 - mean1)
        return w

    def run(self):
        data_array, label_array = self.load_dataset()
        w = self.lda(data_array, label_array)
        k = float(-w[0] / w[1])

        # 正负类点集
        pos_points = data_array[label_array > 0.5]
        neg_points = data_array[label_array <= 0.5]

        plt.figure()
        # 绘制正负类点集
        plt.scatter(neg_points[:, 0], neg_points[:, 1], marker='o', color='k', s=10, label='negative')
        plt.scatter(pos_points[:, 0], pos_points[:, 1], marker='o', color='r', s=10, label='positive')
        # 绘制分类线
        plt.plot([0, 1], [0, k], label='y')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("LDA")

        # 绘制各点到分类线的投影线及投影点
        for i in range(data_array.shape[0]):
            projection_x = (k * data_array[i, 1] + data_array[i, 0]) / (1 + k * k)
            if label_array[i] <= 0.5:
                plt.plot(projection_x, k * projection_x, 'ko', markersize=3)
            else:
                plt.plot(projection_x, k * projection_x, 'ro', markersize=3)
            plt.plot([data_array[i, 0], float(projection_x)], [data_array[i, 1], float(k * projection_x)], 'c--',
                     linewidth=0.3)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    LDA().run()
