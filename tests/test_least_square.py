# testing Least square unit
from LinearAlgebra import LeastSquare
import numpy as np


def test_1():
    X = np.array([1, 2, 3])
    Y = np.array([1, 2, 3])
    

    # Making simple X and Y into numpy array
    weights = LeastSquare.least_square(X, Y)
    print(weights)
    assert len(weights) == 2
