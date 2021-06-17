"""
拉普拉斯修正的朴素贝叶斯分类器

试编程实现拉普拉斯修正的朴素贝叶斯分类器，并以西瓜数据集3,.0为训练集，对P.151“测1”样本进行判别。
"""
import numpy as np
import json


class Bayes:
    def __init__(self, dataset):
        """
        初始化
        :param dataset: 数据集
        """
        self.dataset = np.array(dataset)
        self.features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']
        self.feature_dic = {
            '色泽': ['浅白', '青绿', '乌黑'],
            '根蒂': ['硬挺', '蜷缩', '稍蜷'],
            '敲声': ['沉闷', '浊响', '清脆'],
            '纹理': ['清晰', '模糊', '稍糊'],
            '脐部': ['凹陷', '平坦', '稍凹'],
            '触感': ['硬滑', '软粘']
        }

    def calc_discrete_prob(self, feature_index, value, value_count, class_label):
        """
        计算离散量在拉普拉斯修正下的类条件概率
        :param feature_index: 该特征在特征列表的索引
        :param value: 数据在该特征下的值 x
        :param value_count: 数据在该特征下的取值数目 N
        :param class_label: 类标记 c
        :return: 拉普拉斯修正的类条件概率 P(x|c)
        """
        # 数据集中该特征feature取值为value的数目
        count_feature = 0
        # 数据集中类标记为class_label的数目
        count_class = 0
        for data in self.dataset:
            if data[-1] == class_label:
                count_class += 1
                if data[feature_index] == value:
                    count_feature += 1
        # 拉普拉斯修正下的类条件概率
        prob = (count_feature + 1.) / (count_class + value_count)
        # print(self.features[feature_index], value, class_label)
        # print(prob, count_feature, count_class, value_count)
        # exit()
        return prob

    def calc_continuous(self, feature_index, class_label):
        """
        计算连续量对应的数据————均值和方差
        :param feature_index: 该特征在特征列表的索引
        :param class_label: 类标记 c
        :return: 均值,方差
        """
        # 数据集中特征feature对应的类标记c为class_label的数据
        data_list = []
        for data in self.dataset:
            if data[-1] == class_label:
                # 将符合条件的数据由str类型作为float类型加入列表
                data_list.append(data[feature_index].astype("float64"))
        data_list = np.array(data_list)
        mean = data_list.mean()  # 均值
        var = data_list.var()  # 方差
        # print(self.features[feature_index], class_label)
        # print(data_list)
        # print(feature_index, class_label)
        # print(mean, var)
        # exit()
        return mean, var

    @staticmethod
    def calc_continuous_prob_density(x, mean, var):
        """
        计算连续量x对应的类条件概率P(x|c)————概率密度函数
        :param x: 连续量取值
        :param mean: 数据集中对应特征及类标记下数据的均值
        :param var: 数据集中对应特征及类标记下数据的方差
        :return: 类条件概率P(x|c)
        """
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-(x - mean) ** 2 / (2 * var))

    def naive_bayes(self):
        """
        朴素贝叶斯算法
        :return: 数据集生成的朴素贝叶斯计算数据(字典格式)
        """
        # 朴素贝叶斯计算数据字典
        bayes_dict = {}

        pos_count, neg_count, count = 0, 0, 0
        for data in self.dataset:
            count += 1
            if data[-1] == '1':
                pos_count += 1
            elif data[-1] == '0':
                neg_count += 1
            # 注：读取的数据集self.dataset中各数据均为str类型
        # print(count, pos_count, neg_count)
        bayes_dict["好瓜"] = {}
        bayes_dict["好瓜"]["是"] = (pos_count + 1.) / (count + 2.)  # P(好瓜==是)
        bayes_dict["好瓜"]["否"] = (neg_count + 1.) / (count + 2.)  # P(好瓜==否)

        for feature in self.features:
            bayes_dict[feature] = {}
            feature_index = self.features.index(feature)

            if feature != '密度' and feature != '含糖量':
                # 离散属性
                # 数据在特征feature下对应的取值列表
                value_list = self.feature_dic[feature]
                for value in value_list:
                    positive_prob = self.calc_discrete_prob(feature_index, value, len(value_list), '1')
                    negative_prob = self.calc_discrete_prob(feature_index, value, len(value_list), '0')
                    bayes_dict[feature][value] = {}
                    bayes_dict[feature][value]["是"] = positive_prob  # P(x|是)
                    bayes_dict[feature][value]["否"] = negative_prob  # P(x|否)
            else:
                # 连续属性
                positive_mean_var = self.calc_continuous(feature_index, '1')
                negative_mean_var = self.calc_continuous(feature_index, '0')
                bayes_dict[feature]["是"] = {}  # P(x|是)对应的均值及方差
                bayes_dict[feature]["是"]["均值"] = positive_mean_var[0]
                bayes_dict[feature]["是"]["方差"] = positive_mean_var[1]
                bayes_dict[feature]["否"] = {}  # P(x|否)对应的均值及方差
                bayes_dict[feature]["否"]["均值"] = negative_mean_var[0]
                bayes_dict[feature]["否"]["方差"] = negative_mean_var[1]
        return bayes_dict

    def predict(self, bayes_dict, data):
        """
        对输入数据data根据朴素贝叶斯数据bayes_dict计算分类结果
        :param bayes_dict: 朴素贝叶斯数据
        :param data: 输入数据
        :return: 分类结果,预测为正类的概率,预测为负类的概率
        """
        positive_data = []  # 预测为正类对应的各类条件概率列表
        negative_data = []  # 预测为负类对应的各类条件概率列表
        positive_data.append(bayes_dict["好瓜"]["是"])  # P(好瓜==是)
        negative_data.append(bayes_dict["好瓜"]["否"])  # P(好瓜==否)
        for feature in self.features:
            feature_index = self.features.index(feature)
            value = data[feature_index]  # 输入数据在特征feature下的值
            if feature != '密度' and feature != '含糖量':
                # 离散属性
                positive_data.append(bayes_dict[feature][value]["是"])
                negative_data.append(bayes_dict[feature][value]["否"])
            else:
                # 连续属性
                positive_prob = self.calc_continuous_prob_density(value, bayes_dict[feature]["是"]["均值"],
                                                                  bayes_dict[feature]["是"]["方差"])
                negative_prob = self.calc_continuous_prob_density(value, bayes_dict[feature]["否"]["均值"],
                                                                  bayes_dict[feature]["否"]["方差"])
                positive_data.append(positive_prob)
                negative_data.append(negative_prob)
        prob_pre_positive = 1.  # 预测为正类的概率
        prob_pre_negative = 1.  # 预测为负类的概率
        for prob in positive_data:
            prob_pre_positive *= prob
        for prob in negative_data:
            prob_pre_negative *= prob
        if prob_pre_positive > prob_pre_negative:
            pre = "是"
        else:
            pre = "否"
        # print(pre, prob_pre_positive, prob_pre_negative)
        return pre, prob_pre_positive, prob_pre_negative


if __name__ == '__main__':
    # 数据集
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]
    ]
    # 测试数据
    test_data = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
    bayes = Bayes(dataSet)
    # data_array, label_array = bayes.load_data()
    # print(data_array)
    # print(label_array)
    naive_bayes_dict = bayes.naive_bayes()
    # 将最后的字典型结果转换为json格式（单引号' -> 双引号", 中文不编码为\u类型）
    print(json.dumps(naive_bayes_dict, ensure_ascii=False))
    pre_label, _, _ = bayes.predict(naive_bayes_dict, test_data)
    print('预测结果为：', pre_label)
