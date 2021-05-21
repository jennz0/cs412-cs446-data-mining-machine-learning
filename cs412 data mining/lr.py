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
Xtrain_norm = Xtrain/255.0
Xtest_norm = Xtest/255.0
Ytrain_mod = np.where(Ytrain == 0,1,0)
Ytest_mod = np.where(Ytest == 0,1,0)
loglikeli, accuracy = [],[]
'''for i in range(60000):
    for j in range(784):
        Xtrain[i][j] = Xtrain[i][j]/255.0
        Xtest[i][j] = Xtest[i][j]/255.0'''

num_iter = 100
learning_rate = 0.1/60000
w_ = np.zeros(784)

for i in range(num_iter):
    step = np.zeros(784)
    for x,y in zip(Xtrain_norm,Ytrain_mod):
         step += x*(y - np.exp(np.dot(w_,x))/ (1+np.exp(np.dot(w_,x))))
    w_ += step*learning_rate
    likeli = 0.0
    acc_count = 0
    for x,y in zip(Xtrain_norm,Ytrain_mod):
        likeli += y*np.dot(x,w_) - np.log(1+np.exp(np.dot(w_,x)))
    loglikeli.append(likeli)
    for x,y in zip(Xtest_norm, Ytest_mod):
        pre = np.exp(np.dot(w_,x))/(1+np.exp(np.dot(w_,x)))
        if (((pre>=0.5)and (y==1)) or ((pre<0.5)and (y == 0))):
            acc_count += 1
    accuracy.append(acc_count/10000)

plt.plot([x for x in range(1,101)], loglikeli)
plt.xlabel('num iter')
plt.ylabel('log likelihood')
plt.show()
plt.plot([x for x in range(1,101)], accuracy)
plt.xlabel('num iter')
plt.ylabel('accuracy')
plt.show()
