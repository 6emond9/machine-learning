import pandas as pd
import numpy as np


class TreeGenerate:
    # https://www.kanzhun.com/jiaocheng/543717.html
    # https://blog.csdn.net/qdbszsj/article/details/79081239
    file_name = './db/4.3data.txt'
    data_keys = {
        '色泽': ['青绿', '乌黑', '浅白'],
        '根蒂': ['蜷缩', '硬挺', '稍蜷'],
        '敲声': ['清脆', '沉闷', '浊响'],
        '纹理': ['稍糊', '模糊', '清晰'],
        '脐部': ['凹陷', '稍凹', '平坦'],
        '触感': ['软粘', '硬滑'],
    }
    label_keys = {'好瓜': ['是', '否']}

    def load_dataset(self):
        dataset = pd.read_csv(self.file_name)
        return dataset

    @staticmethod
    def is_same_value(D, A):
        pass

    @staticmethod
    def choose_largest_example(D):
        pass

    @staticmethod
    def choose_best_attribute(D, A):
        pass

    @staticmethod
    def tree_generate(D, A):
        node = {}
        count = D[TreeGenerate.label_keys.keys()].value_counts()
        if len(count) == 1:
            return D[TreeGenerate.label_keys.keys()].values[0]

        if len(A) == 0 or TreeGenerate.is_same_value(D, A):
            return TreeGenerate.choose_largest_example(D)

        best_attr = TreeGenerate.choose_best_attribute(D, A)

        for a_v in TreeGenerate.data_keys[best_attr]:
            Dv = D.loc[D[best_attr] == a_v].copy()
            if len(Dv) == 0:
                TreeGenerate.choose_largest_example(D)
            else:
                TreeGenerate.tree_generate(Dv, A)
        return node

    def run(self):
        pass


if __name__ == '__main__':
    TreeGenerate().run()
