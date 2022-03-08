# @Time : 2022/3/5 20:37
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : MainProgramAblation.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from MILFrame.MILTool import MILTool, classifier, get_ten_fold_index
from MainCode.BagEmbedding import BagEmbedding


class MainProgram:
    def __init__(self, file_path, prop_can_ins=0.2):
        self.file_path = file_path
        self.prop_can_ins = prop_can_ins
        self.times_cv = 5

    def CV(self):
        MIL_model = MILTool(para_file_name=self.file_path)
        # bag_set = MIL_model.bags
        # bag_set_label = MIL_model.bags_label
        acc = []
        f = []
        for times in range(self.times_cv):
            temp_acc, temp_f = self.__one_10cv(MIL_model.bags, MIL_model.bags_label)
            acc.append(temp_acc)
            f.append(temp_f)
        acc = np.array(acc)
        f = np.array(f)
        seven_res_acc = np.zeros((7, 3)).astype('float32')
        for i in range(acc.shape[0]):  # 5次10cv
            for j in range(acc.shape[1]):  # 7个算法
                seven_res_acc[j] += acc[i][j]

        seven_res_f = np.zeros((7, 3)).astype('float32')
        for i in range(f.shape[0]):  # 5次10cv
            for j in range(f.shape[1]):  # 7个算法
                seven_res_f[j] += f[i][j]
        print(seven_res_acc / self.times_cv)
        print(seven_res_f / self.times_cv)
        print('\n\n')
        # print(f)

    def __one_10cv(self, bag_set, bag_set_label):
        """

        :param bag_set:
        :param bag_set_label:
        :return:
        """
        knn_estimator = classifier('KNN')
        DTree_estimator = classifier('DTree')
        SVM_estimator = classifier('poly_SVM')

        acc_score = np.zeros((7, 3)).astype('float32')
        f_score = np.zeros((7, 3)).astype('float32')

        train_index, test_index = get_ten_fold_index(bag_set)
        tr_data = []
        for i in range(10):
            # sleep(0.0001)
            train_bag_set = bag_set[train_index[i]]
            test_bag_set = bag_set[test_index[i]]
            model_learning = BagEmbedding(train_bag_set, test_bag_set, self.prop_can_ins, isAblation=True)

            tr_p, tr_n, tr_b, tr_pn, tr_pb, tr_nb, tr_pnb = model_learning.get_train_embed_vectors()
            # print(tr_p.shape,tr_n., tr_b, tr_pn, tr_pb, tr_nb, tr_pnb)
            te_p, te_n, te_b, te_pn, te_pb, te_nb, te_pnb = model_learning.get_test_embed_vectors()

            com_tr_data = np.array([tr_p.T.tolist(), tr_n.T.tolist(), tr_b.T.tolist(), tr_pn.T.tolist(),
                                    tr_pb.T.tolist(), tr_nb.T.tolist(), tr_pnb.T.tolist()])
            com_te_data = np.array((te_p.T.tolist(), te_n.T.tolist(), te_b.T.tolist(), te_pn.T.tolist(),
                                    te_pb.T.tolist(), te_nb.T.tolist(), te_pnb.T.tolist()))

            train_vector_label = bag_set_label[train_index[i]]
            test_vector_label = bag_set_label[test_index[i]]

            for j in range(7):
                temp_a1, temp_a2, temp_a3, \
                temp_f1, temp_f2, temp_f3 = \
                    self.acc_and_f(knn_estimator, DTree_estimator, SVM_estimator,
                                   np.array(com_tr_data[j]).T, np.array(com_te_data[j]).T, train_vector_label,
                                   test_vector_label)
                acc_score[j][0] += temp_a1
                acc_score[j][1] += temp_a2
                acc_score[j][2] += temp_a3
                f_score[j][0] += temp_f1
                f_score[j][1] += temp_f2
                f_score[j][2] += temp_f3

        acc_score = acc_score / 10
        f_score = f_score / 10
        # print(acc_score, f_score)
        return acc_score, f_score

    def acc_and_f(self, knn_estimator, DTree_estimator, SVM_estimator, train_data, test_data, train_label, test_label):

        # print(train_data.shape,train_label.shape)
        KNN_model = knn_estimator.fit(train_data, train_label)
        predict_knn_label = KNN_model.predict(test_data)

        DTree_model = DTree_estimator.fit(train_data, train_label)
        predict_DTree_label = DTree_model.predict(test_data)
        SVM_model = SVM_estimator.fit(train_data, train_label)
        predict_SVM_label = SVM_model.predict(test_data)

        return accuracy_score(test_label, predict_knn_label), \
               accuracy_score(test_label, predict_DTree_label), \
               accuracy_score(test_label, predict_SVM_label), \
               f1_score(test_label, predict_knn_label), \
               f1_score(test_label, predict_DTree_label), \
               f1_score(test_label, predict_SVM_label)


if __name__ == '__main__':
    file_name = "D:/Data/data_zero/benchmark/tiger.mat"
    print(file_name)
    for ratio in np.arange(0.1, 1.1, 0.1):
        MainProgram(file_name, ratio).CV()
