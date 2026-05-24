"""
Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""

import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5]
Y = [1, 3, 2, 5, 4]

# plot creation
plt.plot(X, Y)

plt.xlabel("data points")
plt.ylabel("target labels")
plt.title("Least square into action")

plt.show()
