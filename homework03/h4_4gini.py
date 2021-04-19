"""
依据基尼指数划分生成决策树
"""
from h4_3tree_generate import Node, load_dataset, is_same_value, choose_largest_example
import plot_tree

# 离散属性
discrete_keys = {
    '色泽': ['青绿', '乌黑', '浅白'],
    '根蒂': ['蜷缩', '稍蜷', '硬挺'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '稍糊', '模糊'],
    '脐部': ['凹陷', '稍凹', '平坦'],
    '触感': ['硬滑', '软粘'],
}
# 分类标签
label_keys = {'好瓜': ['是', '否']}

# 全局变量——当前结点深度
depth = 0


def cal_gini(D):
    """
    计算基尼值
    :param D: 数据集
    :return: 基尼值
    """
    gini = 1.
    keys = label_keys[tuple(label_keys.keys())[0]]
    # 空数据集
    if D.shape[0] == 0:
        return gini

    count = D[tuple(label_keys.keys())[0]].value_counts()
    num = D.shape[0]

    for key in keys:
        if key in count.keys():
            prob = count[key] / num
            gini -= prob ** 2

    return gini


def cal_gini_index(D, a):
    """
    计算数据集D在属性a上的基尼指数
    :param D: 数据集
    :param a: 属性
    :return: 基尼指数
    """
    gini_index = 0.
    D_size = D.shape[0]
    for value in discrete_keys[a]:
        Dv = D.loc[D[a] == value].copy()
        Dv_size = Dv.shape[0]
        gini_Dv = cal_gini(Dv)
        gini_index += Dv_size / D_size * gini_Dv
    return gini_index


def choose_best_attribute_by_gini(D, A):
    """
    依据基尼指数选择最优划分属性
    :param D: 数据集
    :param A: 属性集
    :return: 最优划分属性
    """
    best_attr = ''
    max_gini_index, gini_index = -1., -1.
    for key in A:
        gini_index = cal_gini_index(D, key)

        if max_gini_index < gini_index:
            best_attr = key
            max_gini_index = gini_index

    return best_attr


def decide_tree_predict(node, test_data):
    """
    依据决策树对数据进行预测
    :param node: 决策树结点
    :param test_data: 测试数据
    :return: '是' or '否'
    """
    if node.value == '好瓜':
        return '是'
    elif node.value == '坏瓜':
        return '否'

    for i in range(len(node.children)):
        if test_data[node.value] == node.branch[i]:
            return decide_tree_predict(node.children[i], test_data)


def cal_accuracy(node, test_data):
    """
    计算决策树在验证集上的准确度
    :param node: 决策树结点
    :param test_data: 测试数据集
    :return: 准确度
    """
    num = 0.
    count = test_data.shape[0]
    for i in range(count):
        data = test_data.iloc[i].copy()
        pre = decide_tree_predict(node, data)
        if data[tuple(label_keys.keys())[0]] == pre:
            num += 1
    return num / count


def tree_generate_by_gini(D, A):
    """
    递归生成决策树
    返回情形：
    1.当前结点包含的样本全属于一个类别
    2.当前属性值为空，或是所有样本在所有属性值上取值相同，无法划分
    3.当前结点包含的样本集合为空，不可划分
    :param D: 数据集
    :param A: 属性集
    :return: 结点
    """
    global depth

    # 生成结点
    node = Node()
    depth += 1
    node.depth = depth

    if node.depth == 1:
        root = node

    value_count = D[tuple(label_keys.keys())[0]].value_counts()
    # 1.当前结点包含的样本全属于一个类别
    if len(value_count) == 1:
        node.value = '好瓜' if D['好瓜'].values[0] == '是' else '坏瓜'
        return node

    # 2.当前属性值为空，或是所有样本在所有属性值上取值相同，无法划分
    if len(A) == 0 or is_same_value(D, A):
        node.value = choose_largest_example(D)
        return node

    # 选取最优化分属性
    best_attr = choose_best_attribute_by_gini(D, A)
    # print('%d\t%s' % (node.depth, best_attr))

    # 1.未剪枝
    # 最优划分属性为离散属性时
    node.value = best_attr
    for value in discrete_keys[best_attr]:
        Dv = D.loc[D[best_attr] == value].copy()
        # 3.当前结点包含的样本集合为空，不可划分
        if Dv.shape[0] == 0:
            tmp_node = Node()
            tmp_node.depth = depth + 1
            tmp_node.value = choose_largest_example(D)
            node.branch.append(value)
            node.children.append(tmp_node)
            return node
        else:
            Av = [key for key in A if key != node.value]
            node.branch.append(value)
            node.children.append(tree_generate_by_gini(Dv, Av))
            depth -= 1

    return node


def pre_reduce_branch(node, D, pre_accuracy, root, test):
    """
    决策树预剪枝
    :param node: 决策树结点
    :param D: 训练集
    :param pre_accuracy: 剪枝前准确度
    :param root: 决策树根结点
    :param test: 验证集
    :return: 决策树结点
    """
    if node.value == '好瓜' or node.value == '坏瓜':
        return node

    node_value = []
    for i in range(len(node.children)):
        node_value.append(node.children[i].value)
        Dv = D.loc[D[node.value] == node.branch[i]].copy()
        node.children[i].value = choose_largest_example(Dv)

    accuracy = cal_accuracy(root, test)
    # print(last_accuracy)
    # print(accuracy)

    if accuracy > pre_accuracy:
        for i in range(len(node.children)):
            node.children[i].value = node_value[i]
    else:
        node.value = choose_largest_example(D)
        node.branch.clear()
        node.children.clear()
        # print('%d\t%s' % (node.depth, node.value))
        # print(pre_accuracy)
        # print(accuracy)
        return node

    for i in range(len(node.children)):
        Dv = D.loc[D[node.value] == node.branch[i]].copy()
        node.children[i] = pre_reduce_branch(node.children[i], Dv, accuracy, root, test)

    return node


def post_reduce_branch(node, D, pre_accuracy, root, test):
    """
    决策树后剪枝
    :param node: 决策树结点
    :param D: 训练集
    :param pre_accuracy: 剪枝前准确度
    :param root: 决策树根结点
    :param test: 验证集
    :return: 决策树结点, 剪枝后准确度
    """
    for i in range(len(node.children)):
        if node.branch[i] != '好瓜' and node.branch[i] != '坏瓜':
            Dv = D.loc[D[node.value] == node.branch[i]].copy()
            node.children[i], pre_accuracy = post_reduce_branch(node.children[i], Dv, pre_accuracy, root, test)

    value_tmp = node.value
    node.value = choose_largest_example(D)
    accuracy = cal_accuracy(root, test)
    if accuracy > pre_accuracy:
        # print('%d\t%s' % (node.depth, value_tmp))
        # print(pre_accuracy)
        # print(accuracy)
        node.branch.clear()
        node.children.clear()
        pre_accuracy = accuracy
    else:
        node.value = value_tmp

    return node, pre_accuracy


if __name__ == '__main__':
    file = './db/4.2data.txt'
    dataset = load_dataset(file)
    dataset.drop(columns=['编号'], inplace=True)
    attr = [column for column in dataset.columns if column != tuple(label_keys.keys())[0]]
    train = dataset.iloc[:10].copy()
    test = dataset.iloc[10:].copy()
    root = tree_generate_by_gini(train, attr)

    # # 2.预剪枝
    # print('=' * 18)
    # tmp = root.value
    # root.value = choose_largest_example(train)
    # tree_accuracy = cal_accuracy(root, test)
    # root.value = tmp
    # root = pre_reduce_branch(root, train, tree_accuracy, root, test)

    # # 3.后剪枝
    # print('=' * 18)
    # tree_accuracy = cal_accuracy(root, test)
    # root, last_accuracy = post_reduce_branch(root, train, tree_accuracy, root, test)

    plot_tree.plot_tree(root)
