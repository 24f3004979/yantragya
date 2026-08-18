'''
Simple Data generator module
Making high dimensional dataset with loging information about dataset
1. controlled randomness
2. Known internal parameter for validation of components
'''
import numpy as np

def data_generator(n_samples=50, noice_level=0.1, seed=42, dimension=2):
    np.random.seed(seed)

    u1 = np.linspace(-3, 3, n_samples)
    u2 = np.linspace(-3, 3, n_samples)
    np.random.shuffle(u2)  # Breaking linear alignment

    # Contructing features
    x1 = u1 + np.random.normal(0, noice_level, n_samples)
    x2 = u2 + np.random.normal(0, noice_level, n_samples)

    # One dependent feature | Tweak dependency
    if dimension == 2:
        x3 = x1 * 0.3 + x2 * 2 + np.random.normal(0, noice_level, n_samples)
    else:
        x3 = u1 + np.random.normal(0, noice_level, n_samples)

    # Final Data point with (n,3) | dependency based
    X = np.column_stack((x1,x2,x3))
    y = (0.2 * x1 + 0.8 * x2 + 0.3 * x3) + np.random.normal(0, noice_level, n_samples)
    return X,y  # Dataset with target vector
