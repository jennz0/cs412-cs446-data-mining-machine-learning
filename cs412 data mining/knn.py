import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
from dataloader import *
import numpy as np
import matplotlib.pyplot as plt


dataloader = DataLoader(Xtrainpath='data11/train-images-idx3-ubyte.gz',
                            Ytrainpath='data11/train-labels-idx1-ubyte.gz',
                            Xtestpath='data11/t10k-images-idx3-ubyte.gz',
                            Ytestpath='data11/t10k-labels-idx1-ubyte.gz')
Xtrain, Ytrain, Xtest, Ytest = dataloader.load_data()
Xtrain = np.reshape(Xtrain, (60000,784))
Xtest = np.reshape(Xtest, (10000,784))
Xtrain = Xtrain/255.0
Xtest = Xtest/255.0

train_sample_size = 60000
test_sample_size = 10000
true_count_euc,true_count_man= 0,0
label_e,maximum_e,label_m,maximum_m = -1, 0, -1, 0
poll_e, poll_m = {}, {}
acc_euc, acc_man = [],[]
tree = spatial.KDTree(Xtrain)
idx_e, idx_m = np.array([]),np.array([])

for k in [1,3,5,7,9]:
    for j in range(len(Xtest)):
        target = Xtest[j]
        knn_euc = tree.query(target, k=k, p = 2)[1]    #euclidean
        knn_man = tree.query(target, k=k, p = 1)[1]
        if k == 1:
            knn_euc = np.array([knn_euc])
            knn_man = np.array([knn_man])

        for i in range(len(knn_euc)):
            idx_e = knn_euc[i]
            train_label_e = Ytrain[idx_e]
            if train_label_e in poll_e:
                poll_e[train_label_e] += 1
            else:
                poll_e[train_label_e] = 1

            idx_m = knn_man[i]
            train_label_m = Ytrain[idx_m]
            if train_label_m in poll_m:
                poll_m[train_label_m] += 1
            else:
                poll_m[train_label_m] = 1

        for ii in poll_e:
            if poll_e[ii] >= maximum_e:
                maximum_e = poll_e[ii]
                label_e = ii

        for iii in poll_m:
            if poll_m[iii] >= maximum_m:
                maximum_m = poll_m[iii]
                label_m = iii

        if label_e == Ytest[j]:
            true_count_euc += 1
        if label_m == Ytest[j]:
            true_count_man += 1
        poll_e,poll_m = {}, {}
    acc_euc.append(true_count_euc/10000)
    acc_man.append(true_count_man/10000)
    true_count_man, true_count_euc = 0,0
print(acc_euc,acc_man)

plt.plot([1,3,5,7,9],acc_euc)
plt.xlabel('kval')
plt.ylabel('accuracy')
plt.show()
plt.plot([1,3,5,7,9],acc_man)
plt.xlabel('kval')
plt.ylabel('accuracy')
plt.show()




