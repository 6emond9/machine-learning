"""
svr --- LIBSVM
"""
from libsvm.svmutil import *
import matplotlib.pyplot as plt


def get_data(file_name):
    """
    读取数据
    :param file_name: 数据文件
    :return: y, x
    """
    label_y, data_x = svm_read_problem(file_name)
    return label_y, data_x


def take_first(enum):
    return enum[0]


def svr(data_x, label_y):
    """
    svm模型训练
    :param data_x:
    :param label_y:
    :return: null
    """
    # model = svm_train(y, x, '-t 0')
    model = svm_train(label_y, data_x, '-s 3 -t 2 -c 2.2 -g 2.8 -p 0.01')
    py, mse, prob = svm_predict(label_y, data_x, model, '-b 0')

    data_x = [xi[1] for xi in data_x]

    coordinate = []
    for i in range(len(data_x)):
        coordinate.append([data_x[i], py[i]])
    coordinate.sort(key=take_first)
    x1 = [xi[0] for xi in coordinate]
    y1 = [xi[1] for xi in coordinate]

    plt.scatter(data_x, label_y, marker='o', color='b', s=10, label='row')
    plt.plot(x1, y1, marker='o', color='r', label='pre')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("svr")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    y, x = get_data('db/data_6_8.txt')
    svr(x, y)
