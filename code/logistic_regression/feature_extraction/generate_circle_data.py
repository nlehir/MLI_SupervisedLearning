import math
import numpy as np
import matplotlib.pyplot as plt

threshold = 1
overlap = 0.1

def inside(threshold):
    """
    Generate data closer to the origin
    """
    radius = np.random.uniform(0, (1 + overlap) * threshold, 100)
    theta = np.random.uniform(0, 2*math.pi, 100)
    xpos = radius * np.cos(theta)
    ypos = radius * np.sin(theta)
    return xpos, ypos


def outside(threshold):
    """
    Generate data further from the origin
    """
    radius = np.random.uniform((1 - overlap) * threshold, 3*threshold, 200)
    theta = np.random.uniform(0, 2*math.pi, 200)
    xpos = radius * np.cos(theta)
    ypos = radius * np.sin(theta)
    return xpos, ypos

xposin, yposin = inside(1)
plt.plot(xposin, yposin, "o", alpha = 0.5, color="skyblue", label = "class 0")

xposout, yposout = outside(1.1)
plt.plot(xposout, yposout, "o", alpha = 0.5, color = "mediumblue", label = "class 1")

plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xticks([-3, 0, 3])
plt.yticks([-3, 0, 3])
plt.legend(loc="best")
plt.title("classification problem")
plt.savefig("circle.pdf")


x_in = np.column_stack((xposin, yposin))
y_in = np.zeros((len(x_in), 1))
x_out = np.column_stack((xposout, yposout))
y_out = np.ones((len(x_out), 1))
x = np.vstack((x_in, x_out))
y = np.vstack((y_in, y_out))

np.save("X", x)
np.save("y", y)
