"""
CNN
"""
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


class ConvNet(object):
    def __init__(self):
        """
        2层卷积，2层池化，3层全连接
        """
        self.img_row = 28
        self.img_col = 28

        self.filters1 = [32, 3, 3]
        self.filters2 = [64, 3, 3]
        self.fully_connected = [128, 10]

    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('The shape of dataTrain:', x_train.shape)
        print('The shape of dataTest:', x_test.shape)

        x_train = x_train.reshape(-1, self.img_row, self.img_col, 1).astype('float32')
        x_test = x_test.reshape(-1, self.img_row, self.img_col, 1).astype('float32')
        input_shape = (28, 28, 1)
        # 二值化
        x_train /= 255.
        x_test /= 255.
        y_train = keras.utils.to_categorical(y_train, num_classes=10).astype('float32')
        y_test = keras.utils.to_categorical(y_test, num_classes=10).astype('float32')

        return x_train, y_train, x_test, y_test, input_shape

    def feed_forward(self, x_train, y_train, x_test, y_test, input_shape, batch_size, epochs):
        model = Sequential()
        # 卷积层1
        model.add(
            Conv2D(self.filters1[0], (self.filters1[1], self.filters1[2]), activation='relu', input_shape=input_shape))
        # # 池化层1
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # 卷积层2
        model.add(Conv2D(self.filters2[0], (self.filters2[1], self.filters2[2]), activation='relu', ))
        # 池化层2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flatten层，压平输入数据，相当于图像降维
        model.add(Dropout(0.25))  # 神经元随机失活。迭代时设置0.25的神经元不工作，防止过拟合
        model.add(Flatten())
        # 全连接层1
        model.add(Dense(self.fully_connected[0], activation='relu'))
        model.add(Dropout(0.5))  # 随机失活
        # # 全连接层2
        # 建立输出层，一共有 10 个神经元，因为 0 到 9 一共有 10 个类别， activation 激活函数用 softmax 这个函数用来分类
        model.add(Dense(self.fully_connected[1], activation='softmax'))

        model.summary()

        # 编译模型
        model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        # 训练模型
        train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                  validation_data=(x_test, y_test))
        # 评估模型
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return train_history

    @staticmethod
    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def run(self):
        x_train, y_train, x_test, y_test, input_shape = self.load_dataset()
        train_history = self.feed_forward(x_train, y_train, x_test, y_test, input_shape, batch_size=64, epochs=8)
        self.show_train_history(train_history, 'loss', 'val_loss')
        self.show_train_history(train_history, 'accuracy', 'val_accuracy')


if __name__ == '__main__':
    CNN = ConvNet()
    CNN.run()
