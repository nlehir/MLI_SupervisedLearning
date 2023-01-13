import os

import numpy as np
import matplotlib.pyplot as plt


def func(x: float) -> float:
    return (x-1)**2 +3.5

def func_derivative(x: float) -> float:
    return 2 * (x-1)

n_iter = int(1e3)
x_start = 6
x = x_start
gamma = 1
x_list = list()
y_list = list()
x_list.append(x)
y_list.append(func(x))
for it in range(n_iter):
    print(f"iter: {it}")
    # gradient update
    x = x - gamma * func_derivative(x)
    x_list.append(x)
    y_list.append(func(x))
    print(f"x: {x}")
    print(f"f(x): {func(x)}")

xx = np.linspace(-10, 10)
yy = func(xx)
plt.plot(xx, yy)
plt.plot(x_list, y_list, "o", label="GD iterates", alpha=0.8, markersize=5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
title=f"gradient descent\ngamma={gamma}"
plt.title(title)
figpath = os.path.join("1d_function", f"function_gd_gamma_{gamma}.pdf")
plt.savefig(figpath)
plt.close()
