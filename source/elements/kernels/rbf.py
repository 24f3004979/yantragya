'''
Kernelized pathway

Original dataset -> Infinite dimensional projection
PCA based top n component extraction
'''

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import pdist, squareform, cdist

class RBK:
    def __init__(self, X, gamma=0.00001, n_component=2):
        self.gamma = gamma
        self.X = X
        self.n = n_component
        
        # New attributes to store training state for future predictions
        self.K_centered = None
        self.eigvals = None
        self.eigvect = None
        self.X_kps = None
        self.one_n = None

    def train(self):
        X = self.X
        gamma = self.gamma
        n_component = self.n 

        sq_dists = pdist(X, 'sqeuclidean')
        mat_sq_dist = squareform(sq_dists)

        K = np.exp(-gamma * mat_sq_dist)
        
        N = K.shape[0]
        self.one_n = np.ones((N, N)) / N

        # Center the Kernel matrix
        self.K_centered = K - self.one_n.dot(K) - K.dot(self.one_n) + self.one_n.dot(K).dot(self.one_n)
        eigvals, eigvect = np.linalg.eigh(self.K_centered)

        # Sort eigenvalues and eigenvectors in descending order
        indices = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[indices[:n_component]]
        self.eigvect = eigvect[:, indices[:n_component]]
        
        # Scale eigenvectors by the square root of eigenvalues for correct PCA scaling
        self.X_kps = self.eigvect * np.sqrt(self.eigvals)

        return self.X_kps

    def load(self, X_new):
        '''
        Projects brand new test data into the existing Kernel PCA space.
        '''
        # 1. Compute RBF distance between new data and original training data
        # Shape: (n_samples_new, n_samples_training)
        mat_sq_dist_new = cdist(X_new, self.X, 'sqeuclidean')
        K_new = np.exp(-self.gamma * mat_sq_dist_new)
        
        # 2. Center the new kernel matrix using training column averages
        N_train = self.X.shape[0]
        one_new = np.ones((X_new.shape[0], N_train)) / N_train
        
        K_new_centered = (K_new - one_new.dot(self.K_centered) - 
                          K_new.dot(self.one_n) + 
                          one_new.dot(self.K_centered).dot(self.one_n))
        
        # 3. Project onto the saved training eigenvectors | normalizing with eigen values
        X_new_kps = K_new_centered.dot(self.eigvect) / np.sqrt(self.eigvals)
        
        return X_new_kps

