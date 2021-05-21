import torch
import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 4 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 3
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    #print(X.shape)
    #print(Y.shape)
    n = len(X)
    d = len(X[0])
    w = torch.zeros(d+1)
    w = torch.reshape(w,(d+1,1))

    X0 = torch.ones(n)
    X1 = torch.reshape(X0,(n,1))
    XX = torch.cat((X1,X),1)
    for i in range(num_iter):
        w -= (lrate/n)*(torch.t(XX)@(XX@w- Y))
    #print(w.shape)
    return w

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    
    n = len(X)
    d = len(X[0])
    w = torch.zeros(d+1)
    w = torch.reshape(w,(d+1,1))

    X0 = torch.ones(n)
    X1 = torch.reshape(X0,(n,1))
    XX = torch.cat((X1,X),1)
    return torch.pinverse(XX)@Y

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    dataX,dataY = utils.load_reg_data()
    plt.scatter(dataX,dataY,color = "orange")
    X0 = torch.ones(100)
    X1 = torch.reshape(X0,(100,1))
    XX = torch.cat((X1,dataX),1)
    plt.plot(dataX,torch.t(torch.t(linear_normal(dataX,dataY))@torch.t(XX)))
    plt.title("linear normal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return plt.gcf()

# Problem 4
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n = len(X)
    d = len(X[0])

    w = torch.zeros(int(1 + d + d * (d + 1) / 2))
    w = torch.reshape(w,(int(1 + d + d * (d + 1) / 2),1))

    X0 = torch.ones(n)
    X0t = torch.reshape(X0,(n,1))
    X = torch.cat((X0t,X),1)
    for i in range(1,d+1):
        for j in range(i,d+1):
            tmp = X[:,i]*X[:,j]
            tmp = torch.reshape(tmp, (n,1))
            X = torch.cat((X,tmp),1)
    #print("111111111")
    for i in range(num_iter):
        w -= (lrate/n)*(torch.t(X)@(X@w- Y))
    return w

#a = torch.tensor([[2,3,4],[5,6,7]])
#b = torch.tensor([2,2])
#poly_gd(a, b, lrate=0.01, num_iter=1000)

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n = len(X)
    d = len(X[0])

    w = torch.zeros(int(1 + d + d * (d + 1) / 2))
    w = torch.reshape(w,(int(1 + d + d * (d + 1) / 2),1))

    X0 = torch.ones(n)
    X0t = torch.reshape(X0,(n,1))
    X = torch.cat((X0t,X),1)
    for i in range(1,d+1):
        for j in range(i,d+1):
            tmp = X[:,i]*X[:,j]
            tmp = torch.reshape(tmp, (n,1))
            X = torch.cat((X,tmp),1)
    return torch.pinverse(X)@Y

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    dataX,dataY = utils.load_reg_data()
    plt.scatter(dataX,dataY,color = "pink")
    
    n = len(dataX)
    d = len(dataX[0])
    X0 = torch.ones(n)
    X0t = torch.reshape(X0,(n,1))
    X = torch.cat((X0t,dataX),1)
    for i in range(1,d+1):
        for j in range(i,d+1):
            tmp = X[:,i]*X[:,j]
            tmp = torch.reshape(tmp, (n,1))
            X = torch.cat((X,tmp),1)
    plt.plot(dataX,torch.t(torch.t(poly_normal(dataX,dataY))@torch.t(X)))
    plt.title("poly normal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return plt.gcf()

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    pass

# Problem 5
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n = len(X)
    d = len(X[0])
    w = torch.zeros(d+1)
    w = torch.reshape(w,(d+1,1))

    X0 = torch.ones(n)
    X1 = torch.reshape(X0,(n,1))
    XX = torch.cat((X1,X),1)
    for i in range(num_iter):
        grad = 0
        w -= (lrate /n)* torch.reshape(torch.t((-1/(torch.exp(Y*(XX@w))+1)*Y))@XX, (d+1, 1))        
    return w
def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass
