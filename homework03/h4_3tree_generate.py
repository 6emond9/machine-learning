"""
依据信息增益划分生成决策树
"""
import pandas as pd
import math
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


# 结点类
class Node(object):
    def __init__(self):
        # 结点在决策树的深度
        self.depth = 0
        # 结点文本内容
        self.value = ''
        # 结点向下分支[str1, str2 ... ]
        self.branch = []
        # 结点的子结点[Node1, Node2, ... ]
        self.children = []


def load_dataset(file_name):
    """
    加载本地数据集
    :return: 数据集
    """
    return pd.read_csv(file_name)


def is_same_value(D, A):
    """
    判断数据集D在属性集A上是否划分为相同类别
    :param D: 数据集
    :param A: 属性集
    :return: Ture——相同类别，False——不同类别
    """
    for key in A:
        # if key not in discrete_keys and len(D[key].value_counts()) > 1:
        #     return False
        # if key in discrete_keys and len(D[key].value_counts()) > 1:
        #     return False
        if len(D[key].value_counts()) > 1:
            return False
    return True


def choose_largest_example(D):
    """
    选择数据集D中数量最多的类别
    :param D: 数据集
    :return: 最多类
    """
    count = D['好瓜'].value_counts()
    if len(count) == 0:
        return
    elif len(count) == 1:
        return '好瓜' if count.keys()[0] == '是' else '坏瓜'
    else:
        return '好瓜' if count['是'] > count['否'] else '坏瓜'


def cal_ent(D):
    """
    计算数据集D的信息熵ent
    :param D: 数据集
    :return: 信息熵
    """
    ent = 0.
    keys = label_keys['好瓜']
    # 空数据集
    if D.shape[0] == 0:
        return ent

    count = D['好瓜'].value_counts()
    num = D.shape[0]

    for key in keys:
        if key in count.keys():
            prob = count[key] / num
            ent -= prob * math.log(prob, 2)

    return ent


def cal_gain_discrete(D, a):
    """
    计算数据集D在离散属性a上的信息增益gain
    :param D: 数据集
    :param a: 离散属性
    :return: 信息增益
    """
    gain = cal_ent(D)
    D_size = D.shape[0]
    for value in discrete_keys[a]:
        Dv = D.loc[D[a] == value].copy()
        Dv_size = Dv.shape[0]
        ent_Dv = cal_ent(Dv)
        gain -= Dv_size / D_size * ent_Dv
    return gain


def cal_gain_continuous(D, a):
    """
    计算数据集D在连续属性a上的信息增益gain
    :param D: 数据集
    :param a: 连续属性
    :return: 信息增益
    """
    # 划分值列表
    T = []
    for value in D[a]:
        T.append(value)
    T.sort()
    for i in range(len(T) - 1):
        T[i] = (T[i] + T[i + 1]) / 2
    # 删除多余的最后一个值
    T.pop()

    ent = cal_ent(D)
    max_gain = 0.
    partition = T[0]
    for t in T:
        left, right = pd.DataFrame(), pd.DataFrame()
        for value in D[a]:
            if value <= t:
                left = pd.concat([left, D.loc[D[a] == value]])
            else:
                right = pd.concat([right, D.loc[D[a] == value]])

        gain = ent - left.shape[0] / D.shape[0] * cal_ent(left) - right.shape[0] / D.shape[0] * cal_ent(right)

        if gain > max_gain:
            partition = t
            max_gain = gain

    return max_gain, partition


def choose_best_attribute_by_gain(D, A):
    """
    数据集D在属性集A上的最优化分属性
    :param D: 数据集
    :param A: 属性集
    :return: 离散属性——最优化分属性；连续属性——最优化分属性,划分值
    """
    best_attr = ''
    max_partition, partition = -1., -1.
    max_gain, gain = -1., -1.
    for key in A:
        # 划分属性为离散属性时
        if key in discrete_keys:
            gain = cal_gain_discrete(D, key)
        # 划分属性为连续属性时
        else:
            gain, partition = cal_gain_continuous(D, key)

        if max_gain < gain:
            best_attr = key
            max_gain = gain
            max_partition = partition

    return best_attr, max_partition


def tree_generate_by_gain(D, A):
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
    depth += 1
    # 生成结点
    node = Node()
    node.depth = depth
    value_count = D['好瓜'].value_counts()
    # 1.当前结点包含的样本全属于一个类别
    if len(value_count) == 1:
        node.value = '好瓜' if D['好瓜'].values[0] == '是' else '坏瓜'
        return node

    # 2.当前属性值为空，或是所有样本在所有属性值上取值相同，无法划分
    if len(A) == 0 or is_same_value(D, A):
        node.value = choose_largest_example(D)
        return node

    # 选取最优化分属性
    best_attr, partition = choose_best_attribute_by_gain(D, A)
    # print('%d\t%s' % (node.depth, best_attr))

    # 最优划分属性为离散属性时
    if best_attr in discrete_keys:
        node.value = best_attr + '=?'
        for value in discrete_keys[best_attr]:
            Dv = D.loc[D[best_attr] == value].copy()
            # 3.1当前结点包含的样本集合为空，不可划分
            if Dv.shape[0] == 0:
                tmp_node = Node()
                tmp_node.depth = depth + 1
                tmp_node.value = choose_largest_example(D)
                node.branch.append(value)
                node.children.append(tmp_node)
                return node
            else:
                Av = [key for key in A if key != best_attr]
                node.branch.append(value)
                node.children.append(tree_generate_by_gain(Dv, Av))
                depth -= 1
    # 最优划分属性为连续属性时
    else:
        node.value = best_attr + '<=' + str(round(partition, 3))

        left = D.loc[D[best_attr] <= partition].copy()
        # 3.2当前结点包含的样本集合为空，不可划分
        if left.shape[0] == 0:
            tmp_node = Node()
            tmp_node.depth = depth + 1
            tmp_node.value = choose_largest_example(D)
            node.branch.append('是')
            node.children.append(tmp_node)
            return node
        else:
            node.branch.append('是')
            node.children.append(tree_generate_by_gain(left, A))
            depth -= 1

        right = D.loc[D[best_attr] > partition].copy()
        # 3.3当前结点包含的样本集合为空，不可划分
        if right.shape[0] == 0:
            tmp_node = Node()
            tmp_node.depth = depth + 1
            tmp_node.value = choose_largest_example(D)
            node.branch.append('否')
            node.children.append(tmp_node)
            return node
        else:
            node.branch.append('否')
            node.children.append(tree_generate_by_gain(right, A))
            depth -= 1

    return node


if __name__ == '__main__':
    file = './db/4.3data.txt'
    dataset = load_dataset(file)
    dataset.drop(columns=['编号'], inplace=True)
    attribute = [column for column in dataset.columns if column != '好瓜']
    # 测试连续属性信息增益
    # print(cal_gain_continuous(dataset, '密度'))
    # print(cal_gain_continuous(dataset, '含糖率'))
    root = tree_generate_by_gain(dataset, attribute)
    # 输出决策树基本信息以判断决策树生成正确与否
    # print(get_num_leaves(root))
    # print(get_tree_depth(root))
    plot_tree.plot_tree(root)
