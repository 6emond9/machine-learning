"""
异或RBF神经网络
"""
import numpy as np


class RBF:
    def __init__(self, data, label, q=4, lr=0.1, err=0.001):
        self.data = np.array(data)
        self.lr = lr
        self.label = np.array(label)
        self.d = len(self.data[0]) - 1
        self.q = q
        self.l = len(np.unique(self.data[:, -1]))
        # 服从0到1均匀分布的隐层神经元对应的中心，q个2维，范围[0,1)
        self.center = np.random.rand(self.q, 2)
        self.err = err

    def parameter_init(self):
        # 初始化隐层到输出层的权值 q*1
        w = np.mat(np.random.random((self.q, 1)))
        # 初始化隐层径向基函数的尺度系数 q*1
        beta = np.mat(np.random.random((self.q, 1)))
        print('初始化隐层到输出层权值为\n{}'.format(w))
        print('初始化隐层径向基函数的尺度系数为\n{}'.format(beta))
        return w, beta

    def rbf(self):
        x = self.data
        y = self.label
        center = self.center
        rows = self.data.shape[0]
        lr = self.lr
        w, beta = self.parameter_init()
        # 当前迭代错误率
        e_k = 1.
        # 迭代次数
        counter = 0
        while e_k > self.err:
            counter += 1
            d_w = 0
            d_beta = 0
            for i in range(rows):
                dis = []
                gauss = []
                for j in range(self.q):
                    # 2范数，样本到第j个隐层中心的模长
                    dis_j = np.linalg.norm(x[i] - center[j])
                    dis.append([dis_j])  # q*1
                    gauss_j = np.array(np.exp(-beta[j] * dis_j))
                    gauss.append(gauss_j[0])  # q*1
                out_y = np.mat(w).T * np.mat(gauss)  # 1*1
                d_y = out_y - y[i]
                d_w += -lr * np.array(d_y) * gauss  # q*1
                d_beta += lr * np.array(w) * np.array(d_y) * np.array(dis) * gauss  # q*1
                e_k += 0.5 * np.sum((y[i] - out_y) ** 2)
            w += d_w / rows
            beta += d_beta / rows
            e_k = e_k / rows
        print('一共经过{}轮学习'.format(counter))
        print('学习后的隐层到输出层权值为\n{}'.format(w))
        print('学习后的隐层径向基函数的尺度系数为\n{}'.format(beta))
        return w, beta

    def test(self):
        x_test = np.random.randint(0, 2, (10, 2))
        y_test = np.logical_xor(x_test[:, 0], x_test[:, 1])
        w, beta = self.rbf()
        y_out = []
        for i in range(len(x_test)):
            dis = []
            gauss = []
            for j in range(self.q):
                # 2范数，样本到第j个隐层中心的模长
                dis_j = np.linalg.norm(x_test[i] - self.center[j])
                dis.append([dis_j])  # q*1
                gauss_j = np.array(np.exp(-beta[j] * dis_j))
                gauss.append(gauss_j[0])  # q*1
            out_y = np.mat(w).T * np.mat(gauss)  # 1*1
            y_out.append(np.array(out_y)[0])
        print('测试样本\n{}'.format(x_test))
        print('测试样本的异或值\n{}'.format(y_test))
        print('模型的输出值\n{}'.format(y_out))


if __name__ == '__main__':
    # 异或训练集
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # 异或输出
    y = [[0], [1], [1], [0]]

    rbf = RBF(x, y, 5, 0.1, 0.001)
    rbf.test()
