# @Time : 2022/3/4 16:41
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : BagEmbedding.py

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from MILFrame.MILTool import MILTool, normalize_vector
from MainCode.RepInsSe import RepInsSe


class BagEmbedding:

    def __init__(self, train_bag_set, test_bag_set, prop_can_ins=0.2, isAblation=False):
        self.test_bag_set = test_bag_set
        self.train_bag_set = train_bag_set
        self.num_tr_bag_set = self.train_bag_set.shape[0]
        self.num_te_bag_set = self.test_bag_set.shape[0]
        self.prop_can_ins = prop_can_ins
        self.isAblation = isAblation
        # print(self.isAblation)
        # Get the representative instance set.
        self.pos_rep_ins_set, self.pos_rep_ins_label, self.neg_rep_ins_set, self.neg_rep_ins_label = RepInsSe(
            self.train_bag_set, self.prop_can_ins).get_rep_ins()
        self.train_weight_vector = SVC(kernel='linear', max_iter=1000).fit(
            np.concatenate((self.pos_rep_ins_set, self.neg_rep_ins_set)),
            np.concatenate((self.pos_rep_ins_label, self.neg_rep_ins_label))).coef_[0]
        self.pos_mean,self.neg_mean=self.__get_embed_vectors(self.train_bag_set)

    def get_train_embed_vectors(self):
        """
        Get the embedding vectors of the training bags.
        :return: The train embedding vectors.
        """
        # Step 1: Get the most positive (negative) representative instance corresponding to the train set.
        pos_mean, neg_mean = self.__get_mean_vector(self.train_bag_set)
        train_embed_vectors = []
        # Step 2: Iterate over each bag to get its embedding vector
        for i in range(self.num_tr_bag_set):
            train_embed_vectors.append(
                self.__get_embed_vectors(self.train_bag_set[i, 0][:, :-1], pos_mean, neg_mean))

        if self.isAblation:
            tr_p, tr_n, tr_b, tr_pn, tr_pb, tr_nb, tr_pnb = [], [], [], [], [], [], []
            for i in range(self.num_tr_bag_set):
                p, n, b, pn, pb, nb, pnb = self.__get_embed_vectors(self.train_bag_set[i, 0][:, :-1], pos_mean,
                                                                    neg_mean)
                tr_p.append(p)
                tr_n.append(n)
                tr_b.append(b)
                tr_pn.append(pn)
                tr_pb.append(pb)
                tr_nb.append(nb)
                tr_pnb.append(pnb)
            return np.array(tr_p), np.array(tr_n), np.array(tr_b), \
                   np.array(tr_pn), np.array(tr_pb), np.array(tr_nb), np.array(tr_pnb)

        return np.array(train_embed_vectors)

    def get_test_embed_vectors(self):
        """
        Get the embedding vectors of the test bags.
        :return: The test embedding vectors.
        """
        # Step 1: Get the most positive (negative) representative instance corresponding to the test set.
        pos_mean, neg_mean = self.__get_mean_vector(self.train_bag_set)
        test_embed_vectors = []
        # Step 2: Iterate over each bag to get its embedding vector
        for i in range(self.num_te_bag_set):
            test_embed_vectors.append(
                self.__get_embed_vectors(self.test_bag_set[i, 0][:, :-1], pos_mean, neg_mean))
        if self.isAblation:
            te_p, te_n, te_b, te_pn, te_pb, te_nb, te_pnb = [], [], [], [], [], [], []

            for i in range(self.num_te_bag_set):
                p, n, b, pn, pb, nb, pnb = self.__get_embed_vectors(self.test_bag_set[i, 0][:, :-1], pos_mean,
                                                                    neg_mean)
                te_p.append(p)
                te_n.append(n)
                te_b.append(b)
                te_pn.append(pn)
                te_pb.append(pb)
                te_nb.append(nb)
                te_pnb.append(pnb)
            return np.array(te_p), np.array(te_n), np.array(te_b), \
                   np.array(te_pn), np.array(te_pb), np.array(te_nb), np.array(te_pnb)

        return np.array(test_embed_vectors)

    def __get_embed_vectors(self, bag, pos_mean, neg_mean):
        """
        Get the embedding vector for each bag
        :param bag: A single bag.
        :param pos_mean: The most positive representative instance.
        :param neg_mean: The most negative representative instance.
        :return: The bag embed vector.
        """
        # Step 1: Initialize the positive (negative) perspective vector.
        pos_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        neg_pers_vector = np.zeros(bag.shape[1]).astype('float32')

        # Step 2: Traverse the instances in each bag, divide it into two sub-bags,
        # and get the corresponding positive and negative perspective vectors.
        pos_bag_dis = cdist(bag, self.pos_rep_ins_set)
        neg_bag_dis = cdist(bag, self.neg_rep_ins_set)

        for i in range(bag.shape[0]):

            pos_ins_dis_max = pos_bag_dis[i].max()
            neg_ins_dis_max = neg_bag_dis[i].max()
            pos_ins_dis_min = pos_bag_dis[i].min()
            neg_ins_dis_min = neg_bag_dis[i].min()

            min_fun = abs(pos_ins_dis_min - neg_ins_dis_min)
            max_fun = abs(pos_ins_dis_max - neg_ins_dis_max)

            if min_fun < max_fun:
                if pos_ins_dis_max > neg_ins_dis_max:
                    neg_pers_vector += bag[i] - neg_mean
                else:
                    pos_pers_vector += bag[i] - pos_mean
            else:
                if pos_ins_dis_min > neg_ins_dis_min:
                    neg_pers_vector += bag[i] - neg_mean
                else:
                    pos_pers_vector += bag[i] - pos_mean

        # Step 3: Get bag perspective vector.
        ins_score = []
        for i in range(bag.shape[0]):
            ins_score.append(np.dot(self.train_weight_vector, bag[i]))
        bag_pers_vector = bag[ins_score.index(max(ins_score))]

        # Whether to perform ablation experiments.
        if not self.isAblation:
            return normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector, bag_pers_vector)))
        else:
            return normalize_vector(pos_pers_vector), normalize_vector(neg_pers_vector), normalize_vector(
                bag_pers_vector), normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector))), \
                   normalize_vector(np.concatenate((pos_pers_vector, bag_pers_vector))), \
                   normalize_vector(np.concatenate((neg_pers_vector, bag_pers_vector))), \
                   normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector, bag_pers_vector)))

    def __get_mean_vector(self, bag_set):
        """
        Get the most positive (negative) representative instance corresponding to the bag set.
        :param bag_set: The bag set.
        :return: The most positive (negative) representative instance.
        """
        index_ins_pos = []
        index_ins_neg = []
        # Step 1:Record the representative instance index that resulted in the distance (dis_pos or dis_neg) generation.
        for i in range(bag_set.shape[0]):
            # Step 1.1: Calculate the distance of the bag to the positive and negative representative sets.
            bag_pos_dis = cdist(bag_set[i, 0][:, :-1], self.pos_rep_ins_set)
            bag_neg_dis = cdist(bag_set[i, 0][:, :-1], self.neg_rep_ins_set)

            # Step 1.2: If it is a positive bag, find the closest distance value from the positive representative
            # and the farthest distance value from the negative representative.
            # else, find the farthest distance value from the positive representative
            # and the closest distance value from the negative representative.

            dis_pos = bag_pos_dis.min()
            dis_neg = bag_neg_dis.min()

            # # Step 1.3: Record the representative instance index that resulted in the distance (dis_pos) generation.
            for k in range(bag_pos_dis.shape[1]):
                for j in range(bag_pos_dis.shape[0]):
                    if bag_pos_dis[j][k] == dis_pos:
                        index_ins_pos.append(k)
                        break
            # # Step 1.4: Record the representative instance index that resulted in the distance (dis_neg) generation.
            for k in range(bag_neg_dis.shape[1]):
                for j in range(bag_neg_dis.shape[0]):
                    if bag_neg_dis[j][k] == dis_neg:
                        index_ins_neg.append(k)
                        break

        # Step 2: Find the representative instance within the index value
        # and average it to get the final most representative instance.
        pos_mean = self.pos_rep_ins_set[index_ins_pos].mean(axis=0)
        neg_mean = self.neg_rep_ins_set[index_ins_neg].mean(axis=0)

        return pos_mean, neg_mean


if __name__ == '__main__':
    file_path = "D:/Data/data_zero/benchmark/elephant.mat"
    mil = MILTool(file_path)
    bags = mil.bags

    Demo = BagEmbedding(bags, bags, 0.2)
    print(Demo.get_train_embed_vectors().shape)
