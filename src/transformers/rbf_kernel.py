'''
Kernel could be a separate componenet on its own
With different kernel functions, we aim to lock ito certian exposure about structure of dataset 

1. Radial basis Kernel Pca unit
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import pdist, squareform

class RBK:
    '''
    Radia Basis Kernel PCA

    exponents based function -> Mapping to infinnity
    '''
    def __init__(self, X, gamma=0.00001, n_component=2):
        self.gamma = gamma
        self.X = X
        self.n = n_component

    def load(self):
        '''
        we load dataset after pair wise squaring
        then load it with kernel function of rbf 
        X_kps : Projected PCA component of higher dimensional space

        '''
        X = self.X
        gamma = self.gamma
        n_component = self.n 


        sq_dists = pdist(X, 'sqeuclidean')
        mat_sq_dist = squareform(sq_dists)

        # RBF KERNEL MATRIX
        K = np.exp(-gamma * mat_sq_dist)
        
        # Center the Kernel matrix | How this works ? 
        N = K.shape[0]
        one_n = np.ones((N,N)) / N

        # column average with global average addition
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        eigvals, eigvect = np.linalg.eigh(K_centered)

        # sorting eigen values with respect to eigen vectors
        indices = np.argsort(eigvals)[::-1] # descending order
        X_kps = eigvect[:, indices[:n_component]] # top k vectors

        return X_kps
