import torch
import hw4_utils
import numpy as np

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = hw4_utils.load_data()
    init_c = torch.transpose(init_c,0,1)
    c1 = init_c[0]
    c2 = init_c[1]
    t1 = torch.tensor([])
    t2 = torch.tensor([])
    X = torch.transpose(X,0,1)
    for j in range(n_iters):
        t1 = torch.tensor([])
        t2 = torch.tensor([])
        for i in range(len(X)):
            d1 = ((X[i][0]-c1[0])**2 +(X[i][1]-c1[1])**2)**0.5
            d2 = ((X[i][0]-c2[0])**2 +(X[i][1]-c2[1])**2)**0.5
            if d1 <= d2: #belongs to c1 cluster
                t1 = torch.cat((t1,X[i]))
            else:
                t2 = torch.cat((t2,X[i]))
        l1= int(len(t1)/2)
        t1 = t1.reshape(l1,2)
        l2= int(len(t2)/2)
        t2 = t2.reshape(l2,2)
        
        #c1[0]
        c1[0] = torch.mean(t1,0)[0]
        c1[1] = torch.mean(t1,0)[1]
        c2[0] = torch.mean(t2,0)[0]
        c2[1] = torch.mean(t2,0)[1]
    init_c = torch.Tensor([[c1[0],c2[0]],[c1[1],c2[1]]])    
    return init_c
    
