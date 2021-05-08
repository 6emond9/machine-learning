"""
标准BP算法
"""
import numpy as np


class BP:
    def __init__(self, data, q=4, lr=0.1, err=0.001):
        self.data = np.array(data)
        # 学习率，默认为0.1
        self.lr = lr
        # 输入神经元个数d
        self.d = len(self.data[0]) - 1
        # 隐层神经元个数d
        self.q = q
        # 输出神经元个数d
        self.l = len(np.unique(self.data[:, -1]))
        # 错误率
        self.err = err

    @staticmethod
    def sigmoid(x):
        """
        激活函数
        :param x: 输入x
        :return: 输出
        """
        return 1 / (1 + np.exp(-x))

    def parameter_init(self):
        # 获取数据输入部分
        x = self.data[:, :-1]
        # 在所有数据前插入一列-1作为哑结点
        x = np.insert(x, [0], -1, axis=1)
        # 类别1*l，输出神经元个数跟类别一致，所以取原来的真实输出与真实输出的相反值作为y
        y = np.array([self.data[:, -1], 1 - self.data[:, -1]]).transpose()
        d = self.d
        q = self.q
        l = self.l
        # 初始化输入层到隐层的权值
        v = np.random.random((d + 1, q))
        # 初始化隐层到输出层的权值
        w = np.random.random((q + 1, l))
        return x, y, d, q, l, v, w

    def standard_bp(self):
        x, y, d, q, l, v, w = self.parameter_init()
        print('随机初始化得到的输入层到隐层的连接权v\n{}'.format(v))
        print('随机初始化得到的隐层到输出层的连接权w\n{}'.format(w))

        lr = self.lr
        err = self.err
        # 当前迭代中的错误率
        e_k = 1.
        # 迭代次数
        counter = 0
        while e_k > err:
            counter += 1
            for i in range(self.data.shape[0]):
                # 隐层输入 1*q
                alpha = np.mat(x[i, :]) * v
                # 隐层输出 1*q
                b_init = self.sigmoid(alpha)
                # 1*(q+1)
                b = np.insert(b_init, [0], -1, axis=1)
                # 输出层输入 1*l
                beta = b * w
                # 输出层输出 1*l
                out_y = np.array(self.sigmoid(beta))

                # 权值更新
                # 1*l
                g = out_y * (1 - out_y) * (y[i, :] - out_y)
                # q*1  哑结点的连接权不需要更新，切片
                w_g = w[1:, :] * np.mat(g).T
                # 1*q
                e = np.array(b_init) * (1 - np.array(b_init)) * np.array(w_g.T)
                # (q+1)*1 * (1*l) = (q+1)*l
                d_w = lr * np.mat(b).T * np.mat(g)
                # (d+1)*1 * 1*q = (d+1)*q
                d_v = lr * np.mat(x[i, :]).T * np.mat(e)
                w += d_w
                v += d_v
                e_k = 0.5 * np.sum((y[i, :] - out_y) ** 2)
        print('共经过{}轮BP训练'.format(counter))
        print('得到的输入层到隐层的连接权v\n{}'.format(v))
        print('得到的隐层到输出层的连接权w\n{}'.format(w))


if __name__ == '__main__':
    # 西瓜数据集3.0
    D = np.array([
        [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
        [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
        [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
        [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
        [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
        [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
        [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
        [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
        [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
        [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
        [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
        [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
        [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
        [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
        [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
        [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
        [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]])

    bp = BP(D)
    bp.standard_bp()
