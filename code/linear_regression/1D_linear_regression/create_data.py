import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    mean_noise = 0
    std_noise = 10

    rng = np.random.default_rng()

    def bayes_predictor(x):
        return -3*x - (x/2)**2 + 150

    n_samples = 300

    # temperature in degree
    temperature = np.random.uniform(-5, 35, n_samples)

    # power consumption in MW
    power_consumption = bayes_predictor(temperature) + rng.normal(
            loc=mean_noise,
            scale=std_noise,
            size=n_samples,
            )
    samples = np.column_stack((temperature, power_consumption))

    # plot dataset
    plt.plot(temperature, power_consumption, "o", alpha=0.7)
    plt.xlabel("temperature (°C)")
    plt.ylabel("power_consumption (MW)")
    plt.title("Temperature (°C) vs power consumption (MW)")
    plt.savefig("dataset.pdf")

    # save dataset
    data_path = os.path.join("data", "samples")
    np.save(data_path, samples)


if __name__ == "__main__":
    main()
