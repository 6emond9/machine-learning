import matplotlib.pyplot as plt
import matplotlib as mpl

# 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']
# 字体大小
mpl.rcParams['font.size'] = 12
# 正常显示负号
mpl.rcParams['axes.unicode_minus'] = False

# 绘制决策树——决策结点
decision_node = dict(boxstyle="sawtooth", fc="0.8")
# 绘制决策树——叶结点
leaf_node = dict(boxstyle="round4", fc="0.8")
# 全局变量——画布全长，全宽
total_w, total_d = 0., 0.
# 全局变量——当前决策结点在画布位置
node_x, node_y = 0., 0.
# 全局变量——当前叶结点在画布位置
leaf_node_x, leaf_node_y = 0., 0.
# 全局变量——当前结点的父结点在画布位置[(x1,y1), (x2,y2) ... ]
parent_node = []


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制决策树结点
    :param node_txt: 结点要显示的文本
    :param center_pt: 注释内容的中心坐标
    :param parent_pt: 箭头起始的坐标
    :param node_type: 给标题增加外框
    :return: None
    """
    # node_txt为要显示的文本，xy是箭头尖的坐标，xytest是注释内容的中心坐标
    # xycoords和textcoords是坐标xy与xytext的说明（按轴坐标）
    # 若textcoords=None，则默认textcoords与xycoords相同，若都未设置，默认为data
    # va/ha设置节点框中文字的位置
    # va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')
    # bbox给标题增加外框
    # arrowprops箭头参数,参数类型为字典dict
    plt.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                 textcoords='axes fraction', va="center", ha="center", bbox=node_type,
                 arrowprops=dict(arrowstyle="<-"))


def plot_mid_text(center_pt, parent_pt, txt_string):
    """
    绘制决策树分支文本
    :param center_pt: 注释内容的中心坐标
    :param parent_pt: 箭头起始的坐标
    :param txt_string: 分支文本内容
    :return: None
    """
    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    plt.text(x_mid, y_mid, txt_string, va="center", ha="center")


def get_num_leaves(node):
    """
    获取当前结点向下的叶节点数目
    :param node: 结点
    :return: 叶节点数目
    """
    num_leaves = 0
    if node.value == '好瓜' or node.value == '坏瓜':
        return 1
    else:
        for i in range(len(node.children)):
            num_leaves += get_num_leaves(node.children[i])
    return num_leaves


def get_tree_depth(node):
    """
    获取当前结点向下形成的决策树深度
    :param node: 结点
    :return: 深度
    """
    max_depth, tmp_depth = node.depth, node.depth
    for i in range(len(node.children)):
        tmp_depth = get_tree_depth(node.children[i])
        if tmp_depth > max_depth:
            max_depth = tmp_depth
    return max_depth


def dfs_tree(node, branch=''):
    """
    深度优先遍历决策树
    :param node: 结点
    :param branch: 分支文本
    :return: None
    """
    # 全局变量
    global node_x, node_y, leaf_node_x, leaf_node_y, total_w, total_d, parent_node
    num_w = get_num_leaves(node) - 1

    # 叶结点
    if node.value == '好瓜' or node.value == '坏瓜':
        plot_mid_text((leaf_node_x, node_y), parent_node[-1], branch)
        plot_node(node.value, (leaf_node_x, node_y), parent_node[-1], leaf_node)
        leaf_node_x += 1.0 / total_w
    # 决策结点
    else:
        node_x = leaf_node_x + 0.5 * num_w / total_w
        plot_mid_text((node_x, node_y), parent_node[-1], branch)
        plot_node(node.value, (node_x, node_y), parent_node[-1], decision_node)
        parent_node.append((node_x, node_y))
        for i in range(len(node.children)):
            node_y -= 1.0 / total_d
            dfs_tree(node.children[i], node.branch[i])
            node_y += 1.0 / total_d
        parent_node.pop()


def plot_tree(tree):
    """
    绘制决策树
    :param tree: 决策树根结点
    :return: None
    """
    # 全局变量
    global node_x, node_y, leaf_node_x, leaf_node_y, total_w, total_d, parent_node
    # 画布大小
    total_w = float(get_num_leaves(tree)) - 1
    total_d = float(get_tree_depth(tree)) - 1
    # 初始结点位置
    node_x = 0.5
    node_y = 1.0
    leaf_node_x = 0.0
    leaf_node_y = 1.0
    parent_node.append((node_x, node_y))

    plt.figure(1, figsize=(7.2, 4.8), facecolor='white')
    dfs_tree(tree)

    parent_node.pop()
    # 去掉坐标轴以及刻度
    plt.axis('off')
    plt.show()
