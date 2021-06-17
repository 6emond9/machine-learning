from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# graphviz 可视化树
import graphviz
import numpy as np
import matplotlib.pyplot as plt


class Tree:
    def __init__(self):
        self.feature_names = ''
        self.class_names = ''

    def load_dataset(self):
        """
        获取数据集，并拆分为数据向量和标签
        :return: (feature array, label array)
        """
        data = load_iris()
        data_array = data['data']
        label_array = data['target']
        self.feature_names = data.feature_names
        self.class_names = data.target_names
        # print(data_array, label_array, feature_names, class_names)
        return data_array, label_array

    @staticmethod
    def divide_dataset(x, y):
        """
        利用numpy库实现样本随机抽样，划分出训练集和测试集
        :param x:
        :param y:
        :return:
        """
        n = x.shape[0]
        index = np.arange(n)
        # 打乱原数据
        np.random.shuffle(index)
        # 设置训练集比例
        rate = 0.7
        divide = int(n * rate)
        x_test = x[index[:divide], :]
        y_test = y[index[:divide]]
        x_train = x[index[divide:], :]
        y_train = y[index[divide:]]
        # print(x_test, y_test, x_train, y_train)
        return x_test, y_test, x_train, y_train

    def dec_tree(self, x_test, y_test, x_train, y_train):
        """
        利用sklearn.tree.DecisionTreeClassifier()生成一颗决策树，并对测试集进行预测
        :param x_test:
        :param y_test:
        :param x_train:
        :param y_train:
        :return:
        """
        # 建立决策树对象
        # clf = DecisionTreeClassifier(
        #     criterion='gini', splitter='random', max_depth=4, random_state=14, max_leaf_nodes=5)
        # criterion：特征选择标准，可选参数，默认是gini，可以设置为entropy。
        # splitter：特征划分点选择标准，可选参数，默认是best，可以设置为random。
        # max_features：划分时考虑的最大特征数，可选参数，默认是None。
        # max_depth：决策树最大深，可选参数，默认是None。
        # min_samples_split：内部节点再划分所需最小样本数，可选参数，默认是2。
        # min_weight_fraction_leaf：叶子节点最小的样本权重和，可选参数，默认是0。
        # max_leaf_nodes：最大叶子节点数，可选参数，默认是None。
        # class_weight：类别权重，可选参数，默认是None，也可以字典、字典列表、balanced。
        # random_state：可选参数，默认是None。 设置随机种子为14，使用相同的随机种子能包粽子几次的实验结果相同。但是，在自己的实验中，为保证随机性，可以设置不同的随机种子。
        # min_impurity_split：节点划分最小不纯度,可选参数，默认是1e-7。
        # presort：数据是否预排序，可选参数，默认为False，这个值是布尔值，默认是False不排序。

        clf = []
        score = []
        for i in range(5):
            clf.append(DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=i + 1, random_state=14,
                                              max_leaf_nodes=5))
            clf[i].fit(x_train, y_train)  # 利用训练集训练生成决策树
            score.append(clf[i].score(x_test, y_test))
        plt.plot(range(1, 6), score, color='red', label='max_depth')
        plt.xlabel('决策树最大深度')
        plt.ylabel('准确率')
        plt.legend()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.show()

        index = 0
        best_score = score[index]
        for i in range(len(score)):
            if score[i] > best_score:
                index = i
                best_score = score[i]
        print('max_depth', index + 1)
        print('best_score', best_score)

        best_clf = clf[index]
        # best_clf = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=3, random_state=14,
        #                                   max_leaf_nodes=5)

        dot_data = export_graphviz(best_clf, out_file=None,
                                   feature_names=self.feature_names,
                                   class_names=self.class_names,
                                   filled=True, rounded=True,
                                   special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.view()

        y_pre = best_clf.predict(x_test)  # 得到决策树预测结果
        print('预测值:\n', y_pre)
        print('真实值:\n', y_test)
        # 判断与训练集y是否相等并返回正确率
        acc = np.mean(y_pre == y_test) * 100
        print('准确率为 %.2f%%' % acc)

    def run(self):
        x, y = self.load_dataset()
        x_test, y_test, x_train, y_train = self.divide_dataset(x, y)
        self.dec_tree(x_test, y_test, x_train, y_train)


if __name__ == '__main__':
    tree = Tree()
    tree.run()
