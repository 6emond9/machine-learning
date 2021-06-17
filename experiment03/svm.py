from time import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self):
        self.digits = ''
        self.img = ''

    def load_data(self):
        """
        通过sklearn库加载digits数据集，并输出数据集条目总数
        :return: (x, y)
        """
        self.digits = load_digits()
        data = self.digits.data
        label = self.digits.target
        print('The count of dataset:\t', len(data))
        # print(data.shape, label.shape)
        return data, label

    @staticmethod
    def divide_dataset(data, label):
        """
        利用sklearn.model_selection.train_test_split划分训练集、测试集
        :param data: 数据
        :param label: 标签
        :return: x_train, x_test, y_train, y_test
        """
        # 按照30:13的比例划分出训练集、测试集
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=13 / 43)
        print('The count of train:\t', len(x_train))
        print('The count of test:\t', len(x_test))
        return x_train, x_test, y_train, y_test

    @staticmethod
    def model_train(x_train, y_train):
        """
        训练SVM模型并返回
        :param x_train:
        :param y_train:
        :return: clf
        """
        # 设置SVM模型参数
        clf = svm.SVC(C=10, kernel='rbf', gamma=0.001)
        print('-' * 20)
        print('Start Learning...')
        t0 = time()
        # 训练SVM模型
        clf.fit(x_train, y_train)
        t1 = time()
        t = t1 - t0
        print('训练耗时：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))
        print('Learning is OK...')
        print('-' * 20)
        return clf

    @staticmethod
    def predict(clf, x_test, y_test):
        """
        利用训练完成的SVM模型预测结果
        :param clf: SVM模型分类器
        :param x_test:
        :param y_test:
        :return: y_pre, accuracy
        """
        y_pre = clf.predict(x_test)
        # 判断与训练集y是否相等并返回正确率
        # accuracy = np.mean(y_pre == y_test) * 100
        accuracy = 100 * clf.score(x_test, y_test)
        # print('predict result:\n', result)
        # print('test label:\n', y_test)
        print('accuracy:%.2f%%' % accuracy)
        return y_pre, accuracy

    @staticmethod
    def plot(x_test, y_test, y_pre):
        """
        绘制输出预测
        :param x_test:
        :param y_test:
        :param y_pre:
        :return:
        """
        x_test = [xi.reshape(8, 8) for xi in x_test]
        plt.figure()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            # 设置显示灰度图
            plt.imshow(x_test[i], cmap='gray', interpolation='nearest')
            if y_test[i] == y_pre[i]:
                plt.title('%d = %d' % (y_test[i], y_pre[i]))
            else:
                plt.title('%d != %d' % (y_test[i], y_pre[i]))
            # plt.title(str(y_test[i]) + ' ' + r'$\color{red}' + str(y_pre[i]) + '$')
        plt.show()

    def run(self):
        data, label = self.load_data()
        x_train, x_test, y_train, y_test = self.divide_dataset(data, label)
        clf = self.model_train(x_train, y_train)
        y_pre, accuracy = self.predict(clf, x_test, y_test)
        self.plot(x_test, y_test, y_pre)


if __name__ == '__main__':
    svm_num = SVM()
    svm_num.run()
