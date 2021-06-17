import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    # 3.3α西瓜数据集
    file_name = 'db/3.0xigua.txt'

    def load_dataset_with_index(self, file_name=None):
        """
        从本地获取数据集（数据集有编号列），并拆分为数据向量和标签
        :return: (feature array, label array)
        """
        if file_name is None:
            file_name = self.file_name
        dataset = np.loadtxt(file_name, delimiter=",", dtype=float)
        # 插入全一列
        dataset = np.insert(dataset, 1, np.ones(dataset.shape[0]), axis=1)
        # 略过编号列
        data_array = dataset[:, 1:-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    def load_dataset_no_index(self, file_name=None):
        """
        从本地获取数据集（数据集无编号列），并拆分为数据向量和标签
        :return: (feature array, label array)
        """
        if file_name is None:
            file_name = self.file_name
        dataset = np.loadtxt(file_name, delimiter=",", dtype=float)
        # 插入全一列
        dataset = np.insert(dataset, 0, np.ones(dataset.shape[0]), axis=1)
        data_array = dataset[:, :-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    @staticmethod
    def sigmoid(z):
        """
        计算sigmoid函数输出
        :param z: 输入
        :return: 输出
        """
        return 1.0 / (1 + np.exp(-z))

    def predict01(self, data, beta):
        """
        根据已有的数据和参数模型作预测
        :param data: data_6_2.txt array
        :param beta: parameter β
        :return: predict label -> 0, 1
        """
        pre_array = self.sigmoid(np.dot(data, beta))
        pre_array[pre_array <= 0.5] = 0
        pre_array[pre_array > 0.5] = 1
        return pre_array

    def predict12(self, data, beta):
        """
        根据已有的数据和参数模型作预测
        :param data: data_6_2.txt array
        :param beta: parameter β
        :return: predict label -> 1, 2
        """
        pre_array = self.sigmoid(np.dot(data, beta))
        pre_array[pre_array <= 1.5] = 1
        pre_array[pre_array > 1.5] = 2
        return pre_array

    @staticmethod
    def cal_error_rate(label, pre_label):
        """
        计算预测错误率
        :param label: real label
        :param pre_label: predict label
        :return: error rate
        """
        m = len(pre_label)
        cnt = 0.0
        for i in range(m):
            if pre_label[i] != label[i]:
                cnt += 1.0
        return cnt / float(m)

    def newton_method(self, data_array, label_array):
        """
        牛顿法计算对率回归参数β
        :param data_array: input feature array with shape (m, n)
        :param label_array: the label of data_6_2.txt set with shape (m, 1)
        :return: returns the parameters obtained by newton method
        """
        m, n = data_array.shape
        # (m, 1)
        label_array = label_array.reshape(-1, 1)
        beta = np.ones((n, 1))

        z = np.dot(data_array, beta)
        old_l = 0.
        new_l = np.sum(-label_array * z + np.log(1 + np.exp(z)))
        iter_ = 0.
        while abs(old_l - new_l) > 1e-5:
            iter_ += 1
            # py0 = p(y=0|x) (m,1)
            py0 = self.sigmoid(-np.dot(data_array, beta))
            py1 = 1 - py0
            # reshape(m) -> (m,), np.diag -> (m,m)
            p = np.diag((py0 * py1).reshape(m))

            # β一阶导数 (1, n)
            d_beta = -np.sum(data_array * (label_array - py1), axis=0, keepdims=True)
            # β二阶导数 (n, n)
            d_beta2 = data_array.T.dot(p).dot(data_array)
            d_beta2_inv = np.linalg.inv(d_beta2)
            # 迭代更新β
            beta -= np.dot(d_beta2_inv, d_beta.T)
            # beta -= d_beta.dot(np.linalg.inv(d_beta2)).T

            z = np.dot(data_array, beta)
            old_l = new_l
            new_l = np.sum(-data_array * z + np.log(1 + np.exp(z)))
        # print("newton iteration is ", iter_)
        return beta

    def grad_descent(self, data_array, label_array):
        """
        梯度下降法计算对率回归参数β
        :param data_array: input feature array with shape (m, n)
        :param label_array: the label of data_6_2.txt set with shape (m, 1)
        :return: returns the parameters obtained by newton method
        """
        m, n = data_array.shape
        label_array = label_array.reshape(-1, 1)
        # 学习率
        lr = 0.05
        # 初始化β
        beta = np.ones((n, 1)) * 0.1
        z = data_array.dot(beta)

        for i in range(50):
            py0 = self.sigmoid(-z)
            py1 = 1 - py0
            # β一阶导数
            d_beta = -np.sum(data_array * (label_array - py1), axis=0, keepdims=True)
            # 迭代更新β
            beta -= d_beta.T * lr
            z = data_array.dot(beta)

        return beta

    def run(self):
        data_array, label_array = self.load_dataset_with_index()
        # 牛顿法
        # beta = self.newton_method(data_array, label_array)
        # 梯度下降法
        beta = self.grad_descent(data_array, label_array)
        pre_label = self.predict01(data_array, beta)
        # 错误率
        error_rate = self.cal_error_rate(label_array, pre_label)
        print("y = %fx1 + %fx2 + %f" % (beta[2], beta[1], beta[0]))
        # print("newton error rate is ", error_rate)
        print("grad_descent error rate is ", error_rate)

        # 正类负类点集
        pos_points = data_array[label_array > 0.5]
        neg_points = data_array[label_array < 0.5]

        # 绘制正负类点集与分类边界
        plt.figure()
        plt.scatter(pos_points[:, 1], pos_points[:, 2])
        plt.scatter(neg_points[:, 1], neg_points[:, 2])
        x1 = np.linspace(0, 1, 100)
        x2_newton = -(beta[0] + beta[1] * x1) / beta[2]
        plt.plot(x1, x2_newton, label="newton method")
        plt.xlabel("x1")
        plt.ylabel("x2_newton")
        plt.title("decision boundary")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    LogisticRegression().run()
