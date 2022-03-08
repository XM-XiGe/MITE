# @Time : 2022/3/4 16:34
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : MILTool.py
import os
import sys
import time

import numpy as np
import scipy.io as scio
from scipy.spatial.distance import cdist, euclidean
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MILTool:
    def __init__(self, para_file_name, bags=None):
        self.file_name = para_file_name
        if bags is not None:
            self.bags = bags
        else:
            self.bags = self.__load_file(self.file_name)
        self.bags_label = self.__bag_label(self.bags)
        self.instance_space = ins_space(self.bags)

    def get_dis_data(self, para_file_name, para_data):
        path = 'D:/data/distance/' + para_file_name[:-4] + '_distance.csv'
        if os.path.isfile(path) != 1:
            dis = cdist(para_data, para_data)
            np.savetxt(path, dis, delimiter=',')
        else:
            dis = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
        return dis

    def __load_file(self, para_path):
        """
        Load file.
        @param:
        ------------
            para_file_name: the path of the given file.
        ------------
        @return:
        ------------
            The data.
        ------------
        """
        temp_type = para_path.split('.')[-1]

        if temp_type == 'mat':
            ret_data = scio.loadmat(para_path)
            return ret_data['data']
        else:
            with open(para_path) as temp_fd:
                ret_data = temp_fd.readlines()

            return ret_data

    def get_classifier(self, classifier_model):
        '''
        Get the classifier model you want.
        :param classifier_model: The classifier Model.
        :return:
        '''
        if classifier_model == 'knn':
            return KNeighborsClassifier(n_neighbors=3)
        elif classifier_model == 'DTree':
            return DecisionTreeClassifier()
        elif classifier_model == 'poly':
            return SVC(kernel='poly')
        elif classifier_model == 'rbf_svm':
            return SVC(kernel='rbf')
        elif classifier_model == 'bys':
            return GaussianNB()
        else:
            return None

    def __bag_label(self, bags):
        """
        Get the bags label.
        :param para_bags:
        :return:
        """
        temp_bag_lab = bags[:, -1]
        return np.array([list(val)[0][0] for val in temp_bag_lab])


def get_pos_neg_instance(para_bags):
    pos_ins = []
    neg_ins = []
    for i in range(para_bags.shape[0]):
        for ins in para_bags[i, 0][:, :-1]:
            if para_bags[i, -1] == 1:
                pos_ins.append(ins)
            else:
                neg_ins.append(ins)
    pos_label = np.ones((len(pos_ins), 1)).astype("int32")
    neg_label = np.zeros((len(neg_ins), 1)).astype("int32")
    # pos_ins = np.column_stack((pos_ins, pos_label))
    # neg_ins = np.column_stack((neg_ins, neg_label))
    return np.array(pos_ins), pos_label, np.array(neg_ins), neg_label


def ins_space(para_bags):
    """
    Get the original instance space.
    :param para_bags:the original bags.
    :return: The instance space.
    """
    ins_space = []
    ins_label = []
    if para_bags.shape[0] == 1:
        for ins in para_bags[0][:, :-1]:
            ins_space.append(ins)
    else:
        for i in range(para_bags.shape[0]):
            for ins in para_bags[i, 0][:, :-1]:
                ins_space.append(ins)
                if para_bags[i, -1] == 1:
                    ins_label.append(1)
                else:
                    ins_label.append(0)

    return np.array(ins_space), np.array(ins_label)


def single_ins_space(para_bags, p_o_n=1):
    """
    Get the original positive/negative instance space.
    :param para_bags: All bags.
    :param p_o_n: positive or negative.
    :return: The instance space.
    """
    pos_instance_space = []
    neg_instance_space = []
    for i in range(para_bags.shape[0]):
        if para_bags[i, -1] == 1:
            for ins in para_bags[i, 0][:, :-1]:
                pos_instance_space.append(ins)
        else:
            for ins in para_bags[i, 0][:, :-1]:
                neg_instance_space.append(ins)
    if p_o_n == 1:
        return np.array(pos_instance_space)
    else:
        return np.array(neg_instance_space)


def get_bar(i, j):
    k = i * 10 + j + 1
    str = '>' * ((j + 10 * i) // 2) + ' ' * ((100 - k) // 2)
    sys.stdout.write('\r' + str + '[%s%%]' % (i * 10 + j + 1))
    sys.stdout.flush()
    time.sleep(0.0001)


def get_ten_fold_index(bags):
    """
    Get the training set index and test set index.
    @param
        para_k:
            The number of k-th fold.
    :return
        ret_tr_idx:
            The training set index, and its type is dict.
        ret_te_idx:
            The test set index, and its type is dict.
    """
    num_bags = bags.shape[0]
    temp_rand_idx = np.random.permutation(num_bags)
    temp_fold = int(num_bags / 10)
    ret_tr_idx = {}
    ret_te_idx = {}
    for i in range(10):
        temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx[i] = temp_tr_idx
        ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
    return ret_tr_idx, ret_te_idx


def dis_euclidean(ins1, ins2):
    """
    Calculate the distance between two instances
    :param ins1: the first instance
    :param ins2: the second instance
    :return: the distance between two instances
    """
    dis_instances = np.sqrt(np.sum((ins1 - ins2) ** 2))
    return dis_instances


def cosine_similarity(instance1, instance2):
    """
    Cosine similarity with two instances.
    :param instance1: The first instance.
    :param instance2: The two instance.
    :return:
    """
    num = instance1.dot(instance2.T)
    de_nom = np.linalg.norm(instance1) * np.linalg.norm(instance2)
    if de_nom == 0:
        return 1
    return num / de_nom


def get_cosine(ins):
    ins_long = ins.shape[0]
    cosine = np.zeros((ins_long, ins_long)).astype('float64')
    for ins_i in range(ins_long):
        for ins_j in range(ins_long):
            cosine[ins_i][ins_j] = cosine_similarity(ins[ins_i], ins[ins_j])
    return cosine


def normalize_vector(temp_vector):
    """

    :param temp_vector:
    :return:
    """
    temp_vector = np.sign(temp_vector) * np.sqrt(np.abs(temp_vector))
    temp_norm = np.linalg.norm(temp_vector)
    if temp_norm == 0:
        return temp_vector
    temp_vector = temp_vector / temp_norm
    return temp_vector


def angle_ins(ins1, ins2):
    cos_angle = cosine_similarity(ins1, ins2)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2


def classifier(classifier_model):
    '''
    Get the classifier model you want.
    :param classifier_model: The classifier Model.
    :return:
    '''
    if classifier_model == 'KNN':
        return KNeighborsClassifier(n_neighbors=5)
    elif classifier_model == 'DTree':
        return DecisionTreeClassifier()
    elif classifier_model == 'linear_SVM':
        return SVC(kernel='linear', probability=True)
    elif classifier_model == 'rbf_SVM':
        return SVC(kernel='rbf', probability=True)
    elif classifier_model == 'poly_SVM':
        return SVC(kernel='poly', probability=True)
    elif classifier_model == 'bys':
        return GaussianNB()
    else:
        return None


def discernibility_comparsion(data1, data2):
    '''

    :param data1:
    :param data2:
    :return:
    '''
    pos_mean = data1.mean(axis=0)
    neg_mean = data2.mean(axis=0)
    pos_dis_mean = cdist(data1, data1).mean()
    neg_dis_mean = cdist(data2, data2).mean()
    pos_neg_mean = euclidean(pos_mean, neg_mean)
    return pos_neg_mean / (pos_dis_mean + neg_dis_mean)


def distinguishability_comparison(data1, data2):
    fenzi = cdist(data1, data2).mean()
    pos_mean = data1.mean(axis=0)
    neg_mean = data2.mean(axis=0)
    fenmu = cdist(np.array([pos_mean]), data1).mean() + cdist(np.array([neg_mean]), data2).mean()
    return fenzi / fenmu


def comparative_distinction(bags, bag_label):
    pos = []
    neg = []
    for i in range(bags.shape[0]):
        if bag_label[i] == 1:
            pos.append(bags[i])
        else:
            neg.append(bags[i])
    return distinguishability_comparison(np.array(pos), np.array(neg))
