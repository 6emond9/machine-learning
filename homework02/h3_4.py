from homework02.h3_3logistic_regression import LogisticRegression
from sklearn.model_selection import KFold


def logistic_regression1():
    data_array, label_array = LogisticRegression().load_dataset_no_index('./db/transfusion.data')
    kf1 = KFold(n_splits=10, shuffle=False, random_state=None)
    kf2 = KFold(n_splits=data_array.shape[0], shuffle=False, random_state=None)
    error_rate1, error_rate2 = 0., 0.
    for train_index, test_index in kf1.split(data_array):
        # beta = LogisticRegression().newton_method(data_array[train_index], label_array[train_index])
        beta = LogisticRegression().grad_descent(data_array[train_index], label_array[train_index])
        pre_label = LogisticRegression().predict01(data_array[test_index], beta)
        error_rate1 += LogisticRegression().cal_error_rate(label_array[test_index], pre_label)
    error_rate1 /= 10.

    for train_index, test_index in kf2.split(data_array):
        # beta = LogisticRegression().newton_method(data_array[train_index], label_array[train_index])
        beta = LogisticRegression().grad_descent(data_array[train_index], label_array[train_index])
        pre_label = LogisticRegression().predict01(data_array[test_index], beta)
        error_rate2 += LogisticRegression().cal_error_rate(label_array[test_index], pre_label)
    error_rate2 /= data_array.shape[0]

    print("10次10折交叉验证-错误率:\t%f" % error_rate1)
    print("留一法-错误率:\t%f" % error_rate2)


def logistic_regression2():
    data_array, label_array = LogisticRegression().load_dataset_no_index('./db/haberman.data')
    kf1 = KFold(n_splits=10, shuffle=False, random_state=None)
    kf2 = KFold(n_splits=data_array.shape[0], shuffle=False, random_state=None)
    error_rate1, error_rate2 = 0., 0.
    for train_index, test_index in kf1.split(data_array):
        # beta = LogisticRegression().newton_method(data_array[train_index], label_array[train_index])
        beta = LogisticRegression().grad_descent(data_array[train_index], label_array[train_index])
        pre_label = LogisticRegression().predict12(data_array[test_index], beta)
        error_rate1 += LogisticRegression().cal_error_rate(label_array[test_index], pre_label)
    error_rate1 /= 10.

    for train_index, test_index in kf2.split(data_array):
        # beta = LogisticRegression().newton_method(data_array[train_index], label_array[train_index])
        beta = LogisticRegression().grad_descent(data_array[train_index], label_array[train_index])
        pre_label = LogisticRegression().predict12(data_array[test_index], beta)
        error_rate2 += LogisticRegression().cal_error_rate(label_array[test_index], pre_label)
    error_rate2 /= data_array.shape[0]

    print("10次10折交叉验证-错误率:\t%f" % error_rate1)
    print("留一法-错误率:\t%f" % error_rate2)


if __name__ == '__main__':
    logistic_regression1()
    logistic_regression2()
