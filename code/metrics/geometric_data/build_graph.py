"""
Study different distances for geometric data
"""


import math
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    data = np.load("./data.npy")

    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, "bo")

    def connectpoints(x, y, p1, p2):
        x1, x2 = x[p1], x[p2]
        y1, y2 = y[p1], y[p2]
        plt.plot([x1, x2], [y1, y2], "k-")

    threshold = 1
    # choice of the distance type
    distance_type = "infinie"
    # distance_type = "manhattan"
    # distance_type = "custom"
    distance_type = "euclidean"

    for (i, j) in [(i, j) for i in range(0, x.shape[0]) for j in range(0, x.shape[0])]:
        x_i = x[i]
        y_i = y[i]
        x_j = x[j]
        y_j = y[j]

        if distance_type == "manhattan":
            distance = abs(x_i - x_j) + abs(y_i - y_j)
        elif distance_type == "infinie":
            distance = max(abs(x_i - x_j), abs(y_i - y_j))
        elif distance_type == "custom":
            distance = 0.1 * abs(x_i - x_j) + 0.9 * abs(y_i - y_j)
        elif distance_type == "euclidean":
            distance = math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)

        if distance <= threshold:
            if i is not j:
                connectpoints(x, y, i, j)

    if distance_type == "infinie":
        plt.title(f"threshold {threshold}\ndistance " + r"$L_{\infty}$")
    else:
        plt.title(f"threshold {threshold}\ndistance {distance_type}")
    fig_name = f"thres_{threshold}_dist_{distance_type}.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    main()
