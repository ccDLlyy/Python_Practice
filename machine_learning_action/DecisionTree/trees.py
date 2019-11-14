import math
import operator
import pickle
import json


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for value in label_counts.values():
        prob = float(value) / num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_value = set(feat_list)
        new_entropy = 0.0
        for value in unique_value:
            sub_data_set = split_data_set(data_set, i, value)
            prob = float(len(sub_data_set)) / len(data_set)
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:best_feat] + labels[best_feat + 1:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    feat_index = feat_labels.index(first_str)
    second_dict = input_tree[first_str]
    for key in second_dict.keys():
        if key == test_vec[feat_index]:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, file_name):
    with open('./' + file_name, 'w') as file:
        json.dump(input_tree, file)


def grad_tree(file_name):
    with open('./' + file_name, 'r') as file:
        get_tree = json.load(file)
        return get_tree


# 使用决策树预测隐形眼镜模型
def lenses_recommendation():
    with open('./lenses.txt', 'r') as file:
        # 准备数据
        lenses = [inst.strip().split('\t') for inst in file.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']

        # 分析数据

        # 训练算法
        lenses_tree = create_tree(lenses, lenses_labels)
        store_tree(lenses_tree, 'classifierStorage.json')

        # 分析数据 决策树的树形图
        import tree_plotter
        if __name__ == '__main__':
            tree_plotter.create_plot(lenses_tree)

    # 测试算法
    print('True result is "hard" and the prediction is: ' +
          classify(lenses_tree, lenses_labels, ['young', 'hyper', 'yes', 'normal']))

    # 使用算法
    lenses_tree = grad_tree('classifierStorage.json')
    print('The prediction is: ' +
          classify(lenses_tree, lenses_labels, ['presbyopic', 'myope', 'no', 'normal']))


lenses_recommendation()
