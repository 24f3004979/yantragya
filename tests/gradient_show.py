import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.models.LinearRegression.GradientDescent import GradientDescent

# 1. Load and Split Dataset
df = pd.read_csv('/home/madhav/workspace/projects/yantragya/cleaned_clean_data.csv')
cutoff = int(len(df) * 0.8)

X_train = np.array(df['X'].iloc[:cutoff])
y_train = np.array(df['y'].iloc[:cutoff]).reshape(-1, 1)

X_test_raw = np.array(df['X'].iloc[cutoff:])
y_test = np.array(df['y'].iloc[cutoff:]).reshape(-1, 1)

# Add intercept column to training data for manual cost mapping
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train.reshape(-1, 1)])

# 2. Extract Training Path History 
# Note: If your model.train doesn't return history, modify it to yield or return w_history
model = GradientDescent(X_train, y_train)
w_initial = np.array([3.0, 3.0]).reshape(-1, 1)

# Simulated/Extracted optimization path for visualization tracking
# Replace this mock loop with your internal model tracking if available
w_history = [w_initial.flatten()]
w_current = w_initial.copy()

# Simulating 50 steps of weight updates for the plot path
lr = 0.01 
for _ in range(50):
    gradients = (2 / len(X_train_bias)) * X_train_bias.T @ (X_train_bias @ w_current - y_train)
    w_current -= lr * gradients
    w_history.append(w_current.flatten())
w_history = np.array(w_history)
w = w_current  # Final optimized weight

# 3. Predict on Test Set
X_test = X_test_raw.reshape(-1, 1)
ones_column = np.ones((X_test.shape[0], 1))
X_test_bias = np.hstack([ones_column, X_test])
predictions = X_test_bias @ w

# 4. Generate the Comprehensive Visual Plot
fig = plt.figure(figsize=(14, 6))

# --- LEFT PLOT: Loss Contour & Optimization Path ---
ax1 = fig.add_subplot(121)

# Generate a grid of weights around the final solution to draw the cost landscape
w0_vals = np.linspace(w[0, 0] - 4, w[0, 0] + 4, 100)
w1_vals = np.linspace(w[1, 0] - 4, w[1, 0] + 4, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
SSE_grid = np.zeros(W0.shape)

# Compute Sum of Squared Errors across the grid space
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w_grid = np.array([W0[i, j], W1[i, j]]).reshape(-1, 1)
        errors = (X_train_bias @ w_grid) - y_train
        SSE_grid[i, j] = np.sum(errors ** 2)

# Draw contours
contour = ax1.contourf(W0, W1, SSE_grid, levels=20, cmap='viridis_r')
fig.colorbar(contour, ax=ax1, label='Sum of Squared Error (Cost)')

# Overlap the Gradient Descent path
ax1.plot(w_history[:, 0], w_history[:, 1], color='red', marker='o', markersize=4, label='Descent Path')
ax1.scatter(w[0, 0], w[1, 0], color='gold', marker='*', s=150, zorder=5, label='Optimized Minima')
ax1.set_title('Gradient Descent: Minimizing Error')
ax1.set_xlabel('Intercept (w0)')
ax1.set_ylabel('Slope (w1)')
ax1.legend()

# --- RIGHT PLOT: Data Fit (Your Original View Extended) ---
ax2 = fig.add_subplot(122)
ax2.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train Data')
ax2.scatter(X_test_raw, y_test, color='orange', alpha=0.7, label='Test Data Target')
ax2.plot(X_test_raw, predictions, color='red', linewidth=2, label='Model Prediction Line')
ax2.set_title('Final Fit Regression Line')
ax2.set_xlabel('X Value')
ax2.set_ylabel('y Value')
ax2.legend()

plt.tight_layout()
plt.show()
