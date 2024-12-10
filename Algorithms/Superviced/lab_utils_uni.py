import numpy as np
import matplotlib.pyplot as plt

def plt_intuition(x_train, y_train):
    plt.scatter(x_train, y_train,marker='x', color='blue')
    plt.title('Housing Prices')
    plt.xlabel('Size (Square Feet)')
    plt.ylabel('Price (USD)')
    plt.show()

def plt_stationary(x_train, y_train, w, b):
    temp_f_wp = compute_model_output(x_train, w, b)
    plt.plot(x_train, temp_f_wp, color='red',label='our prediction')
    plt.scatter(x_train, y_train,marker='x', color='blue',label='actual visualization')
    plt.title('Housing Prices')
    plt.xlabel('Size (Square Feet)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def plt_update_onclick(x_train, y_train, w, b):
    plt_intuition(x_train, y_train)
    plt_stationary(x_train, y_train, w, b)

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wp = np.zeros(m)
    for i in range(m):
        f_wp[i] = w*x[i] + b
    return f_wp

def soup_bowl(x_train, y_train, w, b):
    plt_update_onclick(x_train, y_train, w, b)
    w=200
    b=100
    x_i = 1.2
    cost_1200ft = w*x_i + b
    print(f"Cost of 1200 sq ft house: {cost_1200ft}")