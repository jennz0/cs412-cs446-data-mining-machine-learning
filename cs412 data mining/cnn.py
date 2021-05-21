import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloader import *
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(5,5),groups=10)
        #self.pool = nn.MaxPool2d(2, stride=2)
        #self.conv2 = nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(5,5),groups=20)
        #self.pool = nn.MaxPool2d(2, stride=2)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16,kernel_size=3,stride =1, padding= 0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride =1, padding= 0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32*5*5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        '''x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)'''
        return out

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
n_epochs = 40
ninput = 784
nhidden = 32
nclass = 10
train_batch_size = 256
test_batch_size = 1000
dataloader = DataLoader(Xtrainpath='data11/train-images-idx3-ubyte.gz',
                        Ytrainpath='data11/train-labels-idx1-ubyte.gz',
                        Xtestpath='data11/t10k-images-idx3-ubyte.gz',
                        Ytestpath='data11/t10k-labels-idx1-ubyte.gz')
Xtrain, Ytrain, Xtest, Ytest = dataloader.load_data()
Xtrain = torch.Tensor(Xtrain/255.0)
Ytrain = torch.Tensor(Ytrain)
Xtest = torch.Tensor(Xtest/255.0)
Ytest = torch.Tensor(Ytest)
Xtrain = Xtrain.reshape(60000,1,28,28)
Xtest = Xtest.reshape(10000,1,28,28)


train_dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
test_dataset = torch.utils.data.TensorDataset(Xtest, Ytest)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
trainloss_2d,testloss_2d =[],[]
trainacc_2d, testacc_2d = [],[]

for i in range(3):
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    vec_loss_train,vec_loss_test = [],[]
    trainacc, testacc = [],[]
    for epoch in range(n_epochs):
        train_loss = 0
        test_loss = 0
        correct = 0
        total = 0
        acc = 0
        model.train()
        for features,label in train_loader:
            features = features.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output,label.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predict = torch.max(output,1)
            #correct = (predict == label).squeeze()
            acc += (predict == label).sum().item()
            total += label.size(0)
        trainacc.append(acc/len(train_loader.sampler))
        acc = 0
        total = 0

        model.eval()
        for features,label in test_loader:
            output = model(features)
        # calculate the loss
            loss = criterion(output, label.long())
        # update test loss 
            test_loss += loss.item()
            _,predict = torch.max(output,1)
            correct = (predict == label).squeeze()
            acc += correct.sum().item()
            total += label.size(0)

        testacc.append(acc/len(test_loader.sampler))
        vec_loss_train.append(train_loss)
        vec_loss_test.append(test_loss)
    trainloss_2d.append(vec_loss_train)
    testloss_2d.append(vec_loss_test)
    trainacc_2d.append(trainacc)
    testacc_2d.append(testacc)
trainloss_mean = np.mean(trainloss_2d, axis = 0)
testloss_mean = np.mean(testloss_2d, axis = 0)
trainloss_std = np.std(trainloss_2d, axis = 0)
testloss_std = np.std(testloss_2d, axis = 0)
trainacc_mean = np.mean(trainacc_2d, axis = 0)
testacc_mean = np.mean(testacc_2d, axis = 0)
trainacc_std = np.std(trainacc_2d, axis = 0)
testacc_std = np.std(testacc_2d, axis = 0)

plt.xlabel('epoch')
plt.ylabel('loss')
plt.errorbar([i for i in range(40)], np.array(trainloss_mean),yerr = trainloss_std)
plt.errorbar([i for i in range(40)], np.array(testloss_mean), yerr = testloss_std)
plt.show()

plt.errorbar([i for i in range(40)], np.array(trainacc_mean),yerr = trainacc_std)
plt.errorbar([i for i in range(40)], np.array(testacc_mean),yerr = testacc_std)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
