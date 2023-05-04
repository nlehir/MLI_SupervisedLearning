import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    mean_noise = 0
    std_noise = 5

    def bayes_predictor(x):
        return -3 * x + 150

    # temperature in degree
    temperature = np.random.uniform(-5, 35, 300)

    # power consumption in MW
    power_consumption = [
        bayes_predictor(x) + np.random.normal(mean_noise, std_noise)
        for x in temperature
    ]
    samples = np.column_stack((temperature, power_consumption))

    # plot dataset
    plt.plot(temperature, power_consumption, "o", alpha=0.7)
    plt.xlabel("temperature (°C)")
    plt.ylabel("power_consumption (MW)")
    plt.title("dataset")
    plt.savefig("dataset.pdf")

    # save dataset
    data_path = os.path.join("data", "samples")
    np.save(data_path, samples)


if __name__ == "__main__":
    main()
