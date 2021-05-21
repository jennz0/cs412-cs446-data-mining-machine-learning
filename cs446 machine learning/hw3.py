import hw3_utils as utils
#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.batch_size = 5
        self.numofC = num_channels
        self.width = 8
        self.height = 8
        self.cnn1 = nn.Conv2d(num_channels,num_channels, 3, stride = 1, padding = 1,bias=False)
        self.cnn2 = nn.Conv2d(num_channels,num_channels, 3, stride = 1, padding = 1,bias=False)
        self.relu1 = nn.ReLU()
        self.batch2d = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out = self.cnn1(x)
        out = self.batch2d(out)
        out = self.relu1(out)
        out = self.cnn2(out)
        out = self.batch2d(out)
        m = nn.ReLU()
        print(type(out))
        result = m(out+x)
        return result
#obj = Block(nn.Module)
#obj.forward(x)

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.cnn1 = nn.Conv2d(1,num_channels, 3, stride = 2, padding = 1,bias=False)
        self.batch2d = nn.BatchNorm2d(num_channels)        
        self.relu1 = nn.ReLU()
        self.maxp = nn.MaxPool2d(2)
        self.blk = Block(num_channels)
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Linear(num_channels,10,bias = True)


    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        print(x.shape)
        out = self.cnn1(x)
        out = self.batch2d(out)
        out = self.relu1(out)
        out = self.maxp(out)
        out = self.blk(out)

        out = self.adp(out)
        print(out.shape)
        out = out.reshape((10,5))
        out = self.lin(out)
        print(out.shape)
        #out = out.view(out.size[0], -1)
        m = nn.ReLU()
        #out = m(out)
        return out


##### nearest neighbor problem ###
        
        
def one_nearest_neighbor(X,Y,X_test):
    
    # return labels for X_test as torch tensor
    idx_of_nearest = 0
    distance = 0
    mindistance = 100
    maxpoll = -1
    ds = []
    poll = {}
    label = -1
    for i in range(len(X_test)):
        maxpoll = 0
        poll ={}
        mindistance = 100
        for j in range(len(X)):
            distance = ((X_test[i][0]-X[j][0])**2+(X_test[i][1]-X[j][1])**2)**0.5
            if distance <= mindistance:
                idx_of_nearest = j

                if distance < mindistance:
                    poll = {}
                    poll[Y[idx_of_nearest]] = 1
                if distance == mindistance:
                    if Y[j] in poll:
                        poll[Y[idx_of_nearest]] += 1
                    else:
                        poll[Y[idx_of_nearest]] = 1
                mindistance = distance
            #print(poll)

        for ii in poll:
            #label = ii
            if poll[ii] > maxpoll:
                maxpoll = poll[ii]
                label = ii
        

        ds.append(label)
        #print(idx_of_nearest)
    #print(Y)
    return torch.tensor(ds)
    '''print(X.shape)
    print(Y.shape)
    print(X_test.shape)

    return []'''

'''X = torch.tensor([[1,2],[2,3],[3,2],[1,1],[5,5]])
Y = torch.tensor([0,0,1,0,1])
Xt = torch.tensor([[0,1],[3,2],[4,5],[2,2]])
print(one_nearest_neighbor(X,Y,Xt))'''
