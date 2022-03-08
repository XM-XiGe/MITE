# @Time : 2022/3/5 8:56
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : MainProgram.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from MILFrame.MILTool import MILTool, classifier, get_ten_fold_index, comparative_distinction
from MainCode.BagEmbedding import BagEmbedding


class MainProgram:
    def __init__(self, file_path, prop_can_ins=0.2):
        self.file_path = file_path
        self.prop_can_ins = prop_can_ins
        self.times_cv = 5

    def CV(self):
        MIL_model = MILTool(para_file_name=self.file_path)
        acc = []
        f = []
        for times in range(self.times_cv):
            temp_acc, temp_f = self.__one_10cv(MIL_model.bags, MIL_model.bags_label)
            acc.append(temp_acc)
            f.append(temp_f)
        acc = np.array(acc)
        f = np.array(f)

        knn_acc = acc[:, 0].mean()
        knn_acc_std = np.std(acc[:, 0])
        DTree_acc = acc[:, 1].mean()
        DTree_acc_std = np.std(acc[:, 1])
        SVM_acc = acc[:, 2].mean()
        SVM_acc_std = np.std(acc[:, 2])

        knn_f = f[:, 0].mean()
        knn_f_std = np.std(f[:, 0])
        DTree_f = f[:, 1].mean()
        DTree_f_std = np.std(f[:, 1])
        SVM_f = f[:, 2].mean()
        SVM_f_std = np.std(f[:, 2])

        acc_res = "&$%.3f" % (knn_acc) + "" + "_{\pm%.3f}" % (knn_acc_std) + \
                  "\t\t\t&$%.3f" % (DTree_acc) + "_{\pm%.3f}" % (DTree_acc_std) + \
                  "\t\t\t&$%.3f" % (SVM_acc) + "_{\pm%.3f}" % (SVM_acc_std)
        f_res = "&$%.3f" % (knn_f) + "" + "_{\pm%.3f}" % (knn_f_std) + \
                "\t\t\t&$%.3f" % (DTree_f) + "_{\pm%.3f}" % (DTree_f_std) + \
                "\t\t\t&$%.3f" % (SVM_f) + "_{\pm%.3f}" % (SVM_f_std)

        print('acc:' + acc_res + '\t\t\t' + 'f_1:' + f_res)

    def __one_10cv(self, bag_set, bag_set_label):
        """

        :param bag_set:
        :param bag_set_label:
        :return:
        """
        knn_estimator = classifier('KNN')
        DTree_estimator = classifier('DTree')
        SVM_estimator = classifier('poly_SVM')

        acc_score = np.zeros(3).astype('float32')
        f_score = np.zeros(3).astype('float32')
        discrim_score = 0

        train_index, test_index = get_ten_fold_index(bag_set)

        for i in range(10):
            # sleep(0.0001)
            train_bag_set = bag_set[train_index[i]]
            test_bag_set = bag_set[test_index[i]]
            model_learning = BagEmbedding(train_bag_set, test_bag_set, self.prop_can_ins)

            train_vector = model_learning.get_train_embed_vectors()
            test_vector = model_learning.get_test_embed_vectors()
            train_vector_label = bag_set_label[train_index[i]]
            test_vector_label = bag_set_label[test_index[i]]

            KNN_model = knn_estimator.fit(train_vector, train_vector_label)
            predict_knn_label = KNN_model.predict(test_vector)

            acc_score[0] += accuracy_score(test_vector_label, predict_knn_label)
            f_score[0] += f1_score(test_vector_label, predict_knn_label)

            DTree_model = DTree_estimator.fit(train_vector, train_vector_label)
            predict_DTree_label = DTree_model.predict(test_vector)

            acc_score[1] += accuracy_score(test_vector_label, predict_DTree_label)
            f_score[1] += f1_score(test_vector_label, predict_DTree_label)

            SVM_model = SVM_estimator.fit(train_vector, train_vector_label)
            predict_SVM_label = SVM_model.predict(test_vector)

            acc_score[2] += accuracy_score(test_vector_label, predict_SVM_label)
            f_score[2] += f1_score(test_vector_label, predict_SVM_label)

        acc_score = acc_score / 10
        f_score = f_score / 10

        return acc_score, f_score


if __name__ == '__main__':
    file_name = "D:/Data/data_zero/benchmark/musk1.mat"
    print(file_name)
    for ratio in np.arange(0.1, 1.1, 0.1):
        MainProgram(file_name, ratio).CV()
