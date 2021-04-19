"""
Algorithm comparison
"""
from scipy.stats import t
from h4_3tree_generate import *
from h4_4gini import *

import numpy as np
import plot_tree


def paired_t_test(y1_score, y2_score, alpha):
    """
    成对t检验
    :param y1_score: y1
    :param y2_score: y2
    :param alpha: α
    :return: t_stat, df, cv, pv
    """
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


if __name__ == '__main__':
    file = './db/4.2data.txt'
    dataset = load_dataset(file)
    dataset.drop(columns=['编号'], inplace=True)
    attr = [column for column in dataset.columns if column != tuple(label_keys.keys())[0]]
    accuracy1 = []
    accuracy2 = []
    for i in range(dataset.shape[0]):
        test = dataset.iloc[[i]].copy()
        j_list = []
        for j in range(dataset.shape[0]):
            if j != i:
                j_list.append(j)
        train = dataset.iloc[j_list].copy()

        # 1.未剪枝
        root1 = tree_generate_by_gain(train, attr)
        root2 = tree_generate_by_gini(train, attr)
        # # 2.预剪枝
        # tmp = root1.value
        # root1.value = choose_largest_example(train)
        # tree_accuracy = cal_accuracy(root1, test)
        # root1.value = tmp
        # root1 = pre_reduce_branch(root1, train, tree_accuracy, root1, test)
        # tmp = root2.value
        # root2.value = choose_largest_example(train)
        # tree_accuracy = cal_accuracy(root2, test)
        # root2.value = tmp
        # root2 = pre_reduce_branch(root2, train, tree_accuracy, root2, test)
        tree_accuracy = cal_accuracy(root1, test)
        # # 3.后剪枝
        # root1, _ = post_reduce_branch(root1, train, tree_accuracy, root1, test)
        # root2, _ = post_reduce_branch(root2, train, tree_accuracy, root2, test)

        accuracy1.append(cal_accuracy(root1, test))
        accuracy2.append(cal_accuracy(root2, test))

        # plot_tree.plot_tree(root1)
        # plot_tree.plot_tree(root2)
    print(accuracy1)
    print(accuracy2)

    alpha = 0.05
    t_stat, df, cv, pv = paired_t_test(accuracy1, accuracy2, alpha)  # 假设检验
    print('t_stat={}\tdf={}\tcv={}\tpv={}'.format(t_stat, df, cv, pv))
    # interpret via critical value
    # if abs(t_stat) <= cv:
    #     print('Accept null hypothesis that the means are equal.')
    # else:
    #     print('Reject the null hypothesis that the means are equal.')
    # interpret via p-value
    if pv > alpha:
        print('Accept null hypothesis that the means are equal.')
    else:
        print('Reject the null hypothesis that the means are equal.')
