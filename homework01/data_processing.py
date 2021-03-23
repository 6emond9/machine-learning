import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.model_selection import KFold
from sklearn import preprocessing, tree, svm, neural_network, neighbors
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


class DataProcess:
    file_name = './db/haberman.data'
    n_feature = 3
    n_splits = 10
    clfs = {
        'decisionTree': tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2,
                                                    min_samples_leaf=1, max_features=None, max_leaf_nodes=None,
                                                    min_impurity_decrease=0),
        'svm': svm.SVC(C=1.0, kernel='rbf', gamma='auto'),
        'neural_network': neural_network.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001,
                                                       learning_rate='adaptive', learning_rate_init=0.001,
                                                       max_iter=200),
        'KNeighbors': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    }

    def get_dataset(self):
        dataset = np.loadtxt(self.file_name, delimiter=",", dtype=str)
        data_str = dataset[:, :self.n_feature]
        target = dataset[:, self.n_feature:]
        data = []
        for x_str in data_str:
            data.append([float(x) for x in x_str])
        data = np.array(data)
        target = [int(x) - 1 for x in target]
        target = np.array(target)
        return data, target

    @staticmethod
    def preprocessing(data):
        return preprocessing.scale(data)

    def split(self, data, target):
        # shuffle and split training and test sets
        kf = KFold(n_splits=self.n_splits)
        split_results = []
        for train_index, test_index in kf.split(data):
            split_result = [data[train_index], data[test_index], target[train_index], target[test_index]]
            split_results.append(split_result)
        return split_results

    def classifier(self, x_train, y_train, x_test, y_test):
        # Learn to predict each class against the other
        clf = self.clfs['svm']
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_score = clf.decision_function(x_test)
        # print(y_score)
        # exit(0)
        score = clf.score(x_test, y_test)
        return y_pred, y_score, score

    @staticmethod
    def draw_pr(y_test, y_score):
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        # average_precision = average_precision_score(y_test, y_score)
        # f1 = 2 * precision * recall / (precision + recall)

        plt.figure()
        plt.step(recall, precision, where='post')
        plt.fill_between(recall, precision, step='post')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('P-R')
        plt.show()

    @staticmethod
    def draw_roc(y_test, y_score):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        # roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.step(fpr, tpr, where='post')
        plt.fill_between(fpr, tpr, step='post')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.show()

    def run(self):
        data, target = self.get_dataset()
        data = self.preprocessing(data)
        split_results = self.split(data, target)
        score = 0.
        for x_train, x_test, y_train, y_test in split_results:
            y_pred, y_score, score_i = self.classifier(x_train, y_train, x_test, y_test)
            # target_names = ['1', '2']
            confusion_matrix(y_test, y_pred)
            self.draw_pr(y_test, y_score)
            self.draw_roc(y_test, y_score)
            score += score_i
        score /= self.n_splits
        print('cross_val_score: {}'.format(score))
