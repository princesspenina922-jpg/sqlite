import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(x):
    return x**2 + 2*x + 1

def grad_f(x):
    return 2*x + 2

# Gradient descent parameters
x = 4.0             # Initial guess
alpha = 0.1         # Learning rate
tolerance = 1e-6
max_iter = 100

# To store the path of x values
x_values = [x]

# Gradient Descent Loop
for _ in range(max_iter):
    grad = grad_f(x)
    if abs(grad) < tolerance:
        break
    x = x - alpha * grad
    x_values.append(x)

# Create points for plotting the function
x_plot = np.linspace(min(x_values) - 1, max(x_values) + 1, 100)
y_plot = f(x_plot)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = x² + 2x + 1')
plt.plot(x_values, [f(xi) for xi in x_values], 'ro-', label='Gradient Descent Steps')
plt.axvline(x=-1, color='gray', linestyle='--', label='True Minimum x = -1')
plt.title('Gradient Descent on f(x) = x² + 2x + 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
