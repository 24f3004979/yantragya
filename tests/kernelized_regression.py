'''
tesing component : KernelPCA 
Trying to break down high dimensional dataset into simpler components

decomposing swissroll dataset :)
'''
from sklearn.datasets import make_swiss_roll
from src.models.LinearRegression.KPCA import RBK
import matplotlib.pyplot as plt

X, t = make_swiss_roll(
        n_samples=500, 
        noise=0.1,
        random_state=40
        )


# Loading RBK into with swiss roll dataset 
print(f"Loadng Kernel ")
kernel = RBK(X)

maped_data = kernel.train()  # training core module
print(f"Maping output : \n {maped_data}")

map_x = maped_data[:,0]
map_y = maped_data[:,1]


x_1 = X[:,0]
x_2 = X[:,1]
x_3 = X[:,2]
'''
fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(
        x_1,
        x_2,
        x_3,
        c=t
        )
ax.set_title('Swiss ROll dataset')
'''
# Maped dataset visual 
plt.scatter(map_x, map_y, color='blue', alpha=0.3) # sub plot


plt.show()
