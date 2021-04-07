import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    file_name = 'db/3.0xigua.txt'

    def load_dataset_with_index(self, file_name=None):
        """
        get watermelon data set 3.0 alpha

        :return: (feature array, label array)
        """
        if file_name is None:
            file_name = self.file_name
        dataset = np.loadtxt(file_name, delimiter=",", dtype=float)
        dataset = np.insert(dataset, 1, np.ones(dataset.shape[0]), axis=1)
        data_array = dataset[:, 1:-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    def load_dataset_no_index(self, file_name=None):
        """
        get watermelon data set 3.0 alpha

        :return: (feature array, label array)
        """
        if file_name is None:
            file_name = self.file_name
        dataset = np.loadtxt(file_name, delimiter=",", dtype=float)
        dataset = np.insert(dataset, 0, np.ones(dataset.shape[0]), axis=1)
        data_array = dataset[:, :-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

    def predict01(self, data, beta):
        """
        predict

        :param data: data array
        :param beta: parameter β
        :return: predict label -> 0, 1
        """
        pre_array = self.sigmoid(np.dot(data, beta))
        pre_array[pre_array <= 0.5] = 0
        pre_array[pre_array > 0.5] = 1
        return pre_array

    def predict12(self, data, beta):
        """
        predict

        :param data: data array
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
        calculate error rate

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
        calculate logistic parameters by newton method

        :param data_array: input feature array with shape (m, n)
        :param label_array: the label of data set with shape (m, 1)
        :return: returns the parameters obtained by newton method
        """
        m, n = data_array.shape
        label_array = label_array.reshape(-1, 1)
        beta = np.ones((n, 1))

        z = np.dot(data_array, beta)
        old_l = 0.
        new_l = np.sum(-label_array * z + np.log(1 + np.exp(z)))
        iter_ = 0.
        while abs(old_l - new_l) > 1e-5:
            iter_ += 1
            # py0 = p(y=0|x) with shape (m,1)
            py0 = self.sigmoid(-np.dot(data_array, beta))
            py1 = 1 - py0
            # 'reshape(m)' get shape (m,), 'np.diag' get diagonal matrix with shape (m,m)
            p = np.diag((py0 * py1).reshape(m))

            # shape (m,n)
            # first derivative with shape (1, n)
            d_beta = -np.sum(data_array * (label_array - py1), axis=0, keepdims=True)
            # second derivative with shape (n, n)
            d_beta2 = data_array.T.dot(p).dot(data_array)
            d_beta2_inv = np.linalg.inv(d_beta2)
            # β iteration
            beta -= np.dot(d_beta2_inv, d_beta.T)
            # beta -= d_beta.dot(np.linalg.inv(d_beta2)).T

            z = np.dot(data_array, beta)
            old_l = new_l
            new_l = np.sum(-data_array * z + np.log(1 + np.exp(z)))
        # print("newton iteration is ", iter_)
        return beta

    def grad_descent(self, data_array, label_array):
        """
        calculate logistic parameters by gradient descent method

        :param data_array: input feature array with shape (m, n)
        :param label_array: the label of data set with shape (m, 1)
        :return: returns the parameters obtained by newton method
        """
        m, n = data_array.shape
        label_array = label_array.reshape(-1, 1)
        # learn rate
        lr = 0.05
        beta = np.ones((n, 1)) * 0.1
        z = data_array.dot(beta)

        for i in range(50):
            py0 = self.sigmoid(-z)
            py1 = 1 - py0
            d_beta = -np.sum(data_array * (label_array - py1), axis=0, keepdims=True)
            beta -= d_beta.T * lr
            z = data_array.dot(beta)

        return beta

    def run(self):
        data_array, label_array = self.load_dataset_with_index()
        # beta = self.newton_method(data_array, label_array)
        beta = self.grad_descent(data_array, label_array)
        pre_label = self.predict01(data_array, beta)
        error_rate = self.cal_error_rate(label_array, pre_label)
        print("y = %fx1 + %fx2 + %f" % (beta[2], beta[1], beta[0]))
        # print("newton error rate is ", error_rate)
        print("grad_descent error rate is ", error_rate)

        # positive points and negative points
        pos_points = data_array[label_array > 0.5]
        neg_points = data_array[label_array < 0.5]

        # plot decision boundary
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
