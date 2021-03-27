import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from sklearn import preprocessing, svm, linear_model
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score


class DataProcess:
    file_name = './db/haberman.data'
    n_feature = 3
    n_splits = 10
    alpha = 0.05
    clfs = {
        'svm': svm.SVC(C=1.0, kernel='rbf', gamma='auto', probability=True),
        # 'KNeighbors': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        'logistic_regression': linear_model.LogisticRegression(),
        # 'decision_tree': tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2,
        #                                             min_samples_leaf=1, max_features=None, max_leaf_nodes=None,
        #                                             min_impurity_decrease=0),
        # 'neural_network': neural_network.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001,
        #                                                learning_rate='adaptive', learning_rate_init=0.001,
        #                                                max_iter=200)
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

    def classifier(self, x_train, y_train, x_test, y_test, key):
        # Learn to predict each class against the other
        clf = self.clfs[key]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test)[:, 1]
        # print(y_score)
        # exit(0)
        return y_pred, y_score

    @staticmethod
    def paired_t_test(y1_score, y2_score, alpha):
        k = len(y1_score)
        d = [y1_score[i] - y2_score[i] for i in range(k)]
        d = np.array(d)
        # d_ 均值
        d_ = np.mean(d)
        # s 标准差
        s = np.std(d)
        # calculate the t statistic
        t_stat = abs(k ** 0.5 * d_ / s)
        # degrees of freedom
        df = k - 1
        cv = t.ppf(1.0 - alpha, df)
        pv = (1.0 - t.cdf(t_stat, df)) * 2.0
        return t_stat, df, cv, pv

    @staticmethod
    def draw_pr(y_test, y_score, key):
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        # average_precision = average_precision_score(y_test, y_score)
        f1 = 2 * precision * recall / (precision + recall)

        plt.figure()
        plt.step(recall, precision, where='post')
        # plt.fill_between(recall, precision, step='post')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('{} P-R'.format(key))
        plt.show()
        return precision, recall, f1

    @staticmethod
    def draw_roc(y_test, y_score, key):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.step(fpr, tpr, where='post')
        # plt.fill_between(fpr, tpr, step='post')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{} ROC'.format(key))
        plt.show()
        return roc_auc

    def run(self):
        data, target = self.get_dataset()   # 获取数据集
        data = self.preprocessing(data)     # 数据集预处理
        split_results = self.split(data, target)    # 10次10折交叉验证划分
        y_score_t_test = []
        y_test, y_pred, y_score = [], [], []

        for key in self.clfs:
            for x_train_i, x_test_i, y_train_i, y_test_i in split_results:
                y_pred_i, y_score_i = self.classifier(x_train_i, y_train_i, x_test_i, y_test_i, key)    # 分类器
                y_test.extend(y_test_i)
                y_pred.extend(y_pred_i)
                y_score.extend(y_score_i)
            score = cross_val_score(self.clfs[key], data, target, cv=10, scoring='accuracy').mean()     # 十折交叉验证
            print(key)
            # print(len(y_test), len(y_pred), len(y_score))
            print('confusion_matrix:')
            # target_names = ['1', '2']
            print(confusion_matrix(y_test, y_pred))     # 混淆矩阵
            # print(classification_report(y_test, y_pred), end='')
            precision, recall, f1 = self.draw_pr(y_test, y_score, key)  # 绘制P-R曲线
            roc_auc = self.draw_roc(y_test, y_score, key)               # 绘制ROC曲线

            # print('precision={}\nrecall={}\nf1={}\nAUC={}'.format(precision.shape, recall.shape, f1.shape, roc_auc))
            # print('precision={}\nrecall={}\nf1={}\nAUC={}'.format(precision, recall, f1, roc_auc))
            print('cross_val_score: {}'.format(score))
            print('-' * 50)

            y_test.clear()
            y_pred.clear()
            y_score_t_test.append(y_score.copy())
            y_score.clear()

        y1_score = y_score_t_test[0]
        y2_score = y_score_t_test[1]
        # print(len(y1_score), len(y2_score))
        # exit(0)
        t_stat, df, cv, pv = self.paired_t_test(y1_score, y2_score, alpha=self.alpha)   # 假设检验
        print('t_stat={}\tdf={}\tcv={}\tpv={}'.format(t_stat, df, cv, pv))
        # interpret via critical value
        # if abs(t_stat) <= cv:
        #     print('Accept null hypothesis that the means are equal.')
        # else:
        #     print('Reject the null hypothesis that the means are equal.')
        # interpret via p-value
        if pv > self.alpha:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')
