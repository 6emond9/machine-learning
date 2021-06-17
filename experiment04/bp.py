import os
import struct
import numpy as np


class BP:
    def __init__(self, structure, epochs, mini_batch_size, eta=0.01):
        """
        BP类初始化
        :param structure: BP结构——[输入层神经元数目, 隐层神经元数目, 输出层神经元数目]
        :param epochs: 迭代次数
        :param mini_batch_size: 每次输入训练数据数目
        :param eta: 学习率
        """
        self.num_layers = len(structure)
        self.weights = [np.random.randn(y, x) for x, y in zip(structure[:-1], structure[1:])]
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lr = eta

    @staticmethod
    def load_mnist(path, kind='train'):
        """
        Load MNIST data from 'path'
        """
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as label:
            magic, n = struct.unpack('>II', label.read(8))
            labels = np.fromfile(label, dtype=np.uint8)

        with open(images_path, 'rb') as image:
            magic, num, rows, cols = struct.unpack('>IIII', image.read(16))
            images = np.fromfile(image, dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    @staticmethod
    def preprocessing(images, labels):
        """
        数据预处理
        image——灰度值范围(0-255)，转换为(0-1)
        labels——n，转换为[0, 1, ..., n, ..., 9]'
        :param images: 图像数据
        :param labels: 标签
        :return: 处理后的 images, labels_
        """
        images = [xi.reshape(28 * 28, 1) for xi in images]
        # 灰度值范围(0-255)，转换为(0-1)
        images = [xi.reshape(xi.shape[0], 1) / 255.0 for xi in images]
        labels_ = []
        for yi in labels:
            tmp = np.zeros((10, 1))
            tmp[yi] = 1
            labels_.append(tmp)

        images = np.array(images)
        labels_ = np.array(labels_)

        return images, labels_

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y

    def forward(self, x):
        """
        前馈函数
        :param x: 输入
        :return: 输出
        """
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def backprop(self, x_train, y_train):
        """
        误差反传BP
        :param x_train:
        :param y_train:
        :return: nabla_b, nabla_w
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x_train
        activations = [activation]
        zs = []
        for b, w in zip(self.biases, self.weights):
            # print(w.shape, activation.shape, b.shape)
            z = np.dot(w, activation) + b
            # print(z.shape)
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # print(activations[0].shape, activations[1].shape, activations[2].shape)
        delta = self.cost_derivative(activations[-1], y_train) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return nabla_b, nabla_w

    def parameter_update(self, x_train, y_train):
        """
        参数更新
        :param x_train:
        :param y_train:
        :return:
        """
        # nabla_b = [np.zeros(b.shape) for b in self.biases]
        # nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(x_train, y_train):
            nabla_b, nabla_w = self.backprop(x, y)
            self.biases = [b - nb * self.lr for b, nb in zip(self.biases, nabla_b)]
            self.weights = [w - nw * self.lr for w, nw in zip(self.weights, nabla_w)]

    def predict(self, x_test, y_test):
        """
        在测试集上预测准确率
        :param x_test:
        :param y_test:
        :return: y_pre, accuracy
        """
        y_pre = []
        count = 0
        for x, y in zip(x_test, y_test):
            y_p = self.forward(x)
            y_p = y_p.tolist().index(max(y_p))
            y_pre.append(y_p)
            if y_p == y:
                count += 1
        accuracy = count / len(y_test)
        y_pre = np.array(y_pre)
        # accuracy = np.mean(y_pre == y_test)
        return y_pre, accuracy

    def run(self):
        images_train, labels_train = self.load_mnist('./db/', 'train')
        images_train, labels_train = self.preprocessing(images_train, labels_train)
        images_test, labels_test = self.load_mnist('./db/', 't10k')
        images_test, _ = self.preprocessing(images_test, labels_test)
        # print(images_train.shape, labels_train.shape)
        # (60000, 784, 1)(60000, 10, 1)
        # print(images_test.shape, labels_test.shape)
        # (10000, 784, 1)(10000, 10, 1)

        n = len(images_train)
        # 迭代次数
        for epoch in range(self.epochs):
            # 数据索引
            index = list(range(0, n))
            # 数据重新打乱
            np.random.shuffle(index)
            mini_batches = [index[k:k + self.mini_batch_size] for k in range(0, n, self.mini_batch_size)]
            for mini_batch in mini_batches:
                self.parameter_update(images_train[mini_batch], labels_train[mini_batch])  # 输入为训练集时调用函数更新网络参数
            # 在测试集上预测，输出准确率
            y_pre, accuracy = self.predict(images_test, labels_test)
            print("Epoch %d: accuracy = %.2f%%" % (epoch + 1, accuracy * 100))
        print("Epoch complete")


if __name__ == '__main__':
    net = BP([28 * 28, 40, 10], epochs=5, mini_batch_size=100, eta=3.0)
    net.run()
