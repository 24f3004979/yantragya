"""
Least square method
    Find best curve fit for given dataset
"""

import numpy as np

'''
With least square being the assumption that dataset could be made with ease of finding its inverse but its not always the case to find the inverse of the given matrix

thus we go with finding with pseudo inverse
'''


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


