import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors, tree, naive_bayes, svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


class DataProcess:
    file_name = './db/iris.data'
    n_feature = 4
    n_classes = 0
    n_splits = 10
    clfs = {
        'K_neighbor': neighbors.KNeighborsClassifier(),
        'decision_tree': tree.DecisionTreeClassifier(),
        'naive_gaussian': naive_bayes.GaussianNB(),
        'svm': svm.SVC(),
        'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
        'random_forest': RandomForestClassifier(n_estimators=50),
        'adaboost': AdaBoostClassifier(n_estimators=50),
        'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
    }

    def get_dataset(self):
        dataset = np.loadtxt(self.file_name, delimiter=",", dtype=str)
        data_str = dataset[:, :self.n_feature]
        target = dataset[:, self.n_feature:]
        data = []
        for x_str in data_str:
            data.append([float(x) for x in x_str])
        data = np.array(data)
        # Binarize the output
        target = label_binarize(target, classes=[0, 1, 2])
        self.n_classes = target.shape[1]
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
        y_score = clf.decision_function(x_test)
        return y_score

    def draw_pr(self, y_test, y_score):
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(self.n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        precision['macro'], recall['macro'], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision['macro'] = average_precision_score(y_test, y_score, average='macro')
        print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision["macro"]))
        plt.figure()
        # plt.subplot(1, 3, iter_)
        # iter_ += 1
        plt.step(recall['macro'], precision['macro'], where='post')
        # plt.fill_between(recall, precision, step='post', color='b', alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Average precision score, macro-averaged over all classes: AP={0:0.3f}'
                  .format(average_precision["macro"]))
        plt.show()

    def draw_roc(self, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        # plt.subplot(1, 3, iter_)
        # iter_ += 1
        lw = 1
        colors = ['blue', 'red', 'green', 'black', 'yellow']
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.3f})'
                     .format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        plt.show()

    def run(self):
        data, target = self.get_dataset()
        data = self.preprocessing(data)
        split_results = self.split(data, target)
        for x_train, x_test, y_train, y_test in split_results:
            y_score = self.classifier(x_train, y_train, x_test, y_test)
            print(y_score)
            # self.draw_pr(y_test, y_score)
            # self.draw_roc(y_test, y_score)
            exit(0)
