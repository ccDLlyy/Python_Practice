import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import operator
import os


def file_matrix(file_name):
    with open('./' + file_name) as file:
        array_lines = file.readlines()
    number_lines = len(array_lines)
    return_mat = np.zeros((number_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index][:] = [float(data) for data in list_from_line[0:3]]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def img_vector(file_name):
    return_mat = np.zeros((1, 1024))
    with open('./' + file_name) as file:
        for i in range(32):
            line_str = file.readline()
            line_str.strip()
            for j in range(32):
                return_mat[0][i * 32 + j] = int(line_str[j])
    return return_mat


def auto_norm(data_set):
    average = np.mean(data_set, axis=0)
    std_div = np.std(data_set, axis=0)
    norm_data_set = (data_set - average) / std_div
    return norm_data_set, average, std_div


def classify(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = np.sum(sq_diff_mat, axis=1)
    distances = sq_distances ** 0.5
    sorted_distance_index = np.argsort(distances)
    class_count = {}
    for i in range(k):
        label = labels[sorted_distance_index[i]]
        class_count[label] = class_count.get(label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def dating_class_test():
    return_mat, class_label_vector = file_matrix('datingTestSet2.txt')
    norm_data_set, average, std_div = auto_norm(return_mat)
    k = 3
    test_ratio = 0.5
    data_set_size = norm_data_set.shape[0]
    total_test = int(test_ratio * data_set_size)
    error_count = 0
    for i in range(total_test):
        class_result = classify(norm_data_set[i, :], norm_data_set[total_test:, :],
                                class_label_vector[total_test:], k)
        if class_result != class_label_vector[i]:
            error_count += 1
    return error_count / total_test


# kNN应用一：约会网站配对
def make_pair():
    # 准备数据
    data_mat, class_label = file_matrix('datingTestSet2.txt')

    # 分析数据 Matplotlib不支持中文，需要引入中文字体
    font = fm.FontProperties(fname="SimHei.ttf")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0], data_mat[:, 1])
    plt.xlabel('每年获得的飞行常客里程数', fontsize=15, fontproperties=font)
    plt.ylabel('玩游戏所耗时间百分比', fontsize=15, fontproperties=font)
    plt.show()

    # 准备数据 归一化
    norm_mat, aver, std = auto_norm(data_mat)

    # 训练算法 kNN无此步骤

    # 测试算法
    accuracy = dating_class_test()
    print(accuracy)

    # 使用算法
    get_in = input('请输入：')
    get_in.strip()
    miles, percent_games, ice_creams = get_in.split(' ')
    miles = float(miles)
    percent_games = float(percent_games)
    ice_creams = float(ice_creams)
    in_arr = np.array([miles, percent_games, ice_creams])
    in_arr = (in_arr - aver) / std
    classify_result = classify(in_arr, norm_mat, class_label, 3)
    result = ['not at all', 'in small doses', 'in large doses'][classify_result - 1]
    print(result)


# kNN应用二：手写识别系统
def hand_writing():
    # 准备数据 不用归一化
    class_label = []
    train_file_list = os.listdir('./trainingDigits')
    m = len(train_file_list)
    train_data = np.zeros((m, 1024))
    for i in range(m):
        file_name = train_file_list[i]
        file_str = file_name.split('.')[0]
        class_str = int(file_str.split('_')[0])
        class_label.append(class_str)
        train_data[i][:] = img_vector('trainingDigits/%s' % file_name)

    # 训练算法 kNN无此步骤

    # 测试算法
    test_file_list = os.listdir('./testDigits')
    error_count = 0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name = test_file_list[i]
        file_str = file_name.split('.')[0]
        class_str = int(file_str.split('_')[0])
        test_data = img_vector('testDigits/%s' % file_name)
        classify_result = classify(test_data, train_data, class_label, 3)
        if classify_result != class_str:
            error_count += 1
    print('error_count:' + str(error_count))
    print('error_rate:' + str(error_count / m_test * 100.0) + '%')


# make_pair()
# hand_writing()

