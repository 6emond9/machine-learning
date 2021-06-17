"""
svm --- LIBSVM
https://blog.csdn.net/River_J777/article/details/107469726
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from libsvm.svmutil import *


def get_data(file_name):
    """
    读取数据
    :param file_name: 数据文件
    :return: y, x
    """
    label_y, data_x = svm_read_problem(file_name)
    return label_y, data_x


def svm_linear(data_x, label_y):
    """
    svm线性核模型训练
    :param data_x: 数据属性
    :param label_y: 标签
    :return: 训练好的模型
    """
    print("model:")
    # 线性核
    model = svm_train(label_y, data_x, '-t 0 -c 100')
    svm_save_model('Linear.model', model)
    return model


def svm_gauss(data_x, label_y):
    """
    svm高斯核模型训练
    :param data_x: 数据属性
    :param label_y: 标签
    :return: 训练好的模型
    """
    print("model:")
    # 高斯核
    model = svm_train(label_y, data_x, '-t 2 -c 100')
    svm_save_model('Gauss.model', model)
    return model


def plot(data_x, label_y, model):
    """
    绘制分类图
    :param data_x: 数据属性
    :param label_y: 标签
    :param model: 训练好的模型
    :return: null
    """
    x_np = np.asarray([[xi[1], xi[2]] for xi in data_x])
    # print(x_np)

    n, m = 200, 200  # 横纵各采样多少个值
    x1_min, x2_min = np.min(x_np, axis=0)  # 列最小值
    x1_max, x2_max = np.max(x_np, axis=0)  # 列最大值

    t1 = np.linspace(x1_min, x1_max, n)  # 在min-max间线性采样
    t2 = np.linspace(x2_min, x2_max, m)

    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    print("predict:")
    y_fake = np.zeros((n * m,))
    y_predict, _, _ = svm_predict(y_fake, x_show, model)
    # print(y_predict)

    # 绘制分类图
    cm = mpl.colors.ListedColormap(['#A0A0FF', '#FFA0A0'])
    plt.pcolormesh(x1, x2, np.array(y_predict).reshape(x1.shape), cmap=cm)
    plt.scatter(x_np[:, 0], x_np[:, 1], c=label_y, s=3, marker='o')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("svm")
    plt.show()


if __name__ == '__main__':
    y, x = get_data('db/data_6_2.txt')
    model_linear = svm_linear(x, y)
    model_gauss = svm_gauss(x, y)
    plot(x, y, model_linear)
    plot(x, y, model_gauss)
