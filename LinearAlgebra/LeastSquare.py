"""
Least square method
    Find best curve fit for given dataset
"""

import numpy as np


def least_square(X, Y):
    """input : X,Y
    Output : weights [ slope, y_intercept ]"""
    ones = np.ones(len(X))

    A = np.column_stack((X, ones))
    transpose_product = np.matmul(A.T, A)
    inv = np.linalg.inv(transpose_product)

    label_x = np.matmul(A.T, Y)
    final_weight = np.dot(inv, label_x)

    return final_weight
