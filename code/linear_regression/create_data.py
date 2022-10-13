import numpy as np
import matplotlib.pyplot as plt

mean_noise = 0
std_noise = 4


def oracle(x):
    """Oracle

    :x: input variable
    :returns: oracle prediction y

    """
    return 3 * x + 2


inputs = np.random.uniform(-2, 13, 50)
outputs = [oracle(x) + np.random.normal(mean_noise, std_noise) for x in inputs]
samples = np.column_stack((inputs, outputs))

plt.plot(inputs, outputs, "o")
plt.xlabel("inputs")
plt.ylabel("outputs")
plt.title("dataset")
plt.savefig("dataset.pdf")


np.save("samples", samples)
