import matplotlib.pyplot as plt
import numpy as np

# 1. Generate dummy data
np.random.seed(42)
X = np.linspace(0, 10, 100)
# True data with some noise
y_dataset = 2.5 * X + 5 + np.random.normal(0, 2, 100) 

# 2. Define prediction equations (e.g., y = mx + c)
y_model1 = 2.3 * X + 6      # First model prediction
y_model2 = 2.7 * X + 4.5    # Second model prediction

# 3. Layer the plots sequentially
plt.figure(figsize=(8, 5))

# Layer 1: Base dataset (Scatter plot)
plt.scatter(X, y_dataset, color='gray', alpha=0.6, label='Actual Dataset')

# Layer 2: Model 1 prediction (Line plot on top of scatter)
plt.plot(X, y_model1, color='blue', linewidth=2, label='Model 1: $y = 2.3x + 6$')

# Layer 3: Model 2 prediction (Line plot on top of Model 1)
plt.plot(X, y_model2, color='red', linewidth=2, linestyle='--', label='Model 2: $y = 2.7x + 4.5$')

# 4. Add chart details
plt.title('Layered Model Predictions vs Actual Data')
plt.xlabel('X (Input Feature)')
plt.ylabel('y (Target Value)')
plt.legend() # Displays the labels in order
plt.grid(True, alpha=0.3)

# Display the final composite graph
plt.show()

