import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

m = x_train.shape[0] # number of training examples
print(f"m: {m}")

i = 0

plt.scatter(x_train, y_train,marker='x', color='blue')
plt.title('Housing Prices')
plt.xlabel('Size (Square Feet)')
plt.ylabel('Price (USD)')
plt.show()

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wp = np.zeros(m)
    for i in range(m):
        f_wp[i] = w*x[i] + b
    return f_wp

temp_f_wp = compute_model_output(x_train, 100, 100)
plt.plot(x_train, temp_f_wp, color='red',label='our prediction')
plt.scatter(x_train, y_train,marker='x', color='blue',label='actual visualization')
plt.title('Housing Prices')
plt.xlabel('Size (Square Feet)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

w=200
b=100
x_i = 1.2
cost_1200ft = w*x_i + b

print(f"Cost of 1200 sq ft house: {cost_1200ft}")
