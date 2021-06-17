import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Bayes:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data, self.labels = zip(*self.dataset)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.m, self.n = self.data.shape

    @staticmethod
    def calc_mean_var(data_list):
        """
        计算均值和方差
        :param data_list:
        :return: 返回数据的均值和方差
        """
        mean = np.mean(data_list)
        var = np.var(data_list)
        return mean, var

    def bayes(self):
        """
        Naive Bayes算法
        :return:
        """
        data, labels = self.data, self.labels
        m, n = self.m, self.n

        data_dict = {}
        for i in range(m):
            if labels[i] not in data_dict.keys():
                data_dict[labels[i]] = {}
                data_dict[labels[i]]['index'] = []
            data_dict[labels[i]]['index'].append(i)

        for label in data_dict.keys():
            for feature_i in range(n):
                mean, var = self.calc_mean_var(data[data_dict[label]['index'], feature_i])
                data_dict[label][feature_i] = {}
                data_dict[label][feature_i]['mean'] = mean
                data_dict[label][feature_i]['var'] = var
        return data_dict

    def predict_bayes(self, x, data_dict):
        """
        预测数据函数
        :param x:
        :param data_dict:
        :return:
        """
        m, n = self.m, self.n

        p_dist = {}
        for label in data_dict.keys():
            p_dist[label] = {}
            for feature_i in range(n):
                p_dist[label][feature_i] = (1 / (np.sqrt(2 * np.pi * data_dict[label][feature_i]['var']))) * np.exp(
                    -(x[feature_i] - data_dict[label][feature_i]['mean']) ** 2 / (
                            2 * data_dict[label][feature_i]['var']))

        result = list(p_dist.keys())[0]
        p_max = 0.
        for label in p_dist.keys():
            p = len(data_dict[label]['index']) / float(m)
            for feature_i in range(n):
                p = p * p_dist[label][feature_i]
            if p > p_max:
                result = label
                p_max = p
        # print(result)
        return result


if __name__ == '__main__':
    iris_data = datasets.load_iris().data
    iris_labels = datasets.load_iris().target

    data_train, data_test, labels_train, labels_test = train_test_split(iris_data, iris_labels, test_size=0.3,
                                                                        random_state=30)

    bayes = Bayes(zip(data_train, labels_train))
    data_dist = bayes.bayes()

    labels_pre = []
    for data_i in data_test:
        labels_pre.append(bayes.predict_bayes(data_i, data_dist))

    print("真实标签：", list(labels_test))
    print("预测结果：", labels_pre)

    acc = np.mean([labels_pre == labels_test])
    print("准确率：%.2f%%" % (acc * 100))
