import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloader import *
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        '''self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        '''
        self.fc1 = nn.Linear(28*28, 32)
        #self.fc3 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

        pass
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc2(x)
        return x
        '''
        x = x.view(-1,28*28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc(x))
        # add dropout layer
        x = self.fc2(x)
        return x'''

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

train_dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
test_dataset = torch.utils.data.TensorDataset(Xtest, Ytest)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
trainloss_2d,testloss_2d =[],[]
trainacc_2d, testacc_2d = [],[]

for i in range(3):
    model = MLP()
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
print(trainloss_std)


plt.xlabel('epoch')
plt.ylabel('loss')
plt.errorbar([i for i in range(40)], np.array(trainloss_mean),yerr = trainloss_std)
plt.errorbar([i for i in range(40)], np.array(testloss_mean), yerr = testloss_std)

plt.show()


plt.xlabel('epoch')
plt.ylabel('acc')
plt.errorbar([i for i in range(40)], np.array(trainacc_mean),yerr = trainacc_std)
plt.errorbar([i for i in range(40)], np.array(testacc_mean),yerr = testacc_std)

plt.show()
