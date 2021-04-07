import numpy as np
import matplotlib.pyplot as plt


class LDA:
    file_name = 'db/3.0xigua.txt'

    def load_dataset(self):
        """
        get watermelon data set 3.0 alpha

        :return: (feature array, label array)
        """
        dataset = np.loadtxt(self.file_name, delimiter=",", dtype=float)
        data_array = dataset[:, 1:-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    @staticmethod
    def lda(data_array, label_array):
        """
        Linear Discriminant Analysis
        :param data_array:
        :param label_array:
        :return: parameter w
        """
        data_array0 = data_array[label_array < 0.5]
        data_array1 = data_array[label_array >= 0.5]

        mean0 = np.mean(data_array0, axis=0).reshape((-1, 1))
        mean1 = np.mean(data_array1, axis=0).reshape((-1, 1))
        cov0 = np.cov(data_array0, rowvar=False)
        cov1 = np.cov(data_array1, rowvar=False)
        sw = np.mat(cov0 + cov1)
        w = np.dot(sw.I, mean0 - mean1)
        return w

    def run(self):
        data_array, label_array = self.load_dataset()
        w = self.lda(data_array, label_array)
        k = float(-w[0] / w[1])

        # positive points and negative points
        pos_points = data_array[label_array > 0.5]
        neg_points = data_array[label_array < 0.5]

        # plot decision boundary
        plt.figure()
        plt.scatter(neg_points[:, 0], neg_points[:, 1], marker='o', color='k', s=10, label='negative')
        plt.scatter(pos_points[:, 0], pos_points[:, 1], marker='o', color='r', s=10, label='positive')
        plt.plot([0, 1], [0, k], label='y')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("LDA")

        for i in range(data_array.shape[0]):
            projection_x = (k * data_array[i, 1] + data_array[i, 0]) / (1 + k * k)
            if label_array[i] < 0.5:
                plt.plot(projection_x, k * projection_x, 'ko', markersize=3)
            else:
                plt.plot(projection_x, k * projection_x, 'ro', markersize=3)
            plt.plot([data_array[i, 0], float(projection_x)], [data_array[i, 1], float(k * projection_x)], 'c--',
                     linewidth=0.3)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    LDA().run()
