import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

m = x_train.shape[0] # number of training examples
print(f"m: {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]

plt.scatter(x_train, y_train,marker='x', color='blue')
plt.title('Housing Prices')
plt.xlabel('Size (Square Feet)')
plt.ylabel('Price (USD)')
plt.show()