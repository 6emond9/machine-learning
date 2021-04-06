import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    # local dataset
    file_name = 'db/3.0xigua.txt'

    def load_dataset(self):
        """
        get watermelon data set 3.0 alpha
        :return: (feature array, label array)
        """
        dataset = np.loadtxt(self.file_name, delimiter="\t", dtype=float)
        dataset = np.insert(dataset, 1, np.ones(dataset.shape[0]), axis=1)
        data_array = dataset[:, 1:-1]
        label_array = dataset[:, -1]
        return data_array, label_array.astype(int)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

    def predict(self, data, beta):
        pre_array = self.sigmoid(np.dot(data, beta))
        pre_array[pre_array <= 0.5] = 0
        pre_array[pre_array > 0.5] = 1
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
        pre = self.predict(data_array, beta)
        error_rate = self.cal_error_rate(label_array, pre)

        z = np.dot(data_array, beta)
        old_l = 0
        new_l_mat = -label_array * z + np.log(1 + self.sigmoid(z))
        new_l = np.nansum(new_l_mat)
        iter_ = 0
        while abs(old_l - new_l) > 1e-5:
            iter_ += 1
            # py0 = p(y=0|x) with shape (m,1)
            py0 = self.sigmoid(-np.dot(data_array, beta))
            py1 = 1 - py0
            # 'reshape(m)' get shape (m,), 'np.diag' get diagonal matrix with shape (m,m)
            p = np.diag((py0 * py1).reshape(m))

            # shape (m,n)
            d_beta_mat = -data_array * (label_array - py1)
            # first derivative with shape (1, n)
            d_beta = np.nansum(d_beta_mat, axis=0, keepdims=True)
            # second derivative with shape (n, n)
            d_beta2 = data_array.T.dot(p).dot(data_array)
            d_beta2_inv = np.linalg.inv(d_beta2)
            # β iteration
            beta = beta - np.dot(d_beta2_inv, d_beta.T)

            z = np.dot(data_array, beta)
            old_l = new_l
            new_l_mat = -data_array * z + np.log(1 + self.sigmoid(z))
            new_l = np.nansum(new_l_mat)

            pre = self.predict(data_array, beta)
            error_rate = self.cal_error_rate(label_array, pre)
        print("newton iteration is ", iter_)
        return beta, error_rate

    def run(self):
        data_array, label_array = self.load_dataset()
        beta, error = self.newton_method(data_array, label_array)
        print("newton error rate is ", error)

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
