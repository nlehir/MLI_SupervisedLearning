"""
    File illustrating the analysis of a time series.
    The time series stores measurements of the level
    of tide as a function of time.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.optimize

# open file
file_name = 'data.csv'

times = list()
tide_level = list()

# load the data
with open(file_name, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        times.append(float(row[0]))
        tide_level.append(float(row[1]))

times = np.asarray(times)
tide_level = np.asarray(tide_level)

"""
    Part A : we want to plot the line graph
    of the tide lebel as a function of time.
"""
# plot the line graph of the data
plt.plot(times, tide_level)
plt.title("tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.savefig("tide_level.pdf")
plt.close()

plt.plot(times, tide_level, "o")
plt.title("tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.savefig("tide_level_o.pdf")
plt.close()

plt.plot(times, tide_level, "x")
plt.title("tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.savefig("tide_level_x.pdf")
plt.close()

"""
    Part B : we want to select a subset
    of the dataset that could be useful in order
    to find structure in it.
"""
selected_id = np.arange(20, 1200)
selected_time_index = times[selected_id]
selected_level = tide_level[selected_id]

plt.plot(selected_time_index, selected_level, "o")
plt.title("selected tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.savefig("selected_tide_level.pdf")

"""
    Part C : manually selected plot limits.
"""
plt.plot(times, tide_level, "o")
plt.title("tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.xlim([2,8])
plt.savefig("tide_level_limits.pdf")
plt.close()

"""
    Part D : optimizing a function
"""


def fit_sinus(times, tide_level):
    """
        function used to fit a sinusoidal function to the data.
        :param times: array of time steps
        :param tide level: array of tide levels
    """
    # guess initial values for the parameters
    # using spectral analysis
    # Fourier transform
    ff = np.fft.fftfreq(len(times), (times[1]-times[0]))
    Ftide_level = abs(np.fft.fft(tide_level))
    guess_freq = abs(ff[np.argmax(Ftide_level[1:])+1])
    guess_amp = np.std(tide_level) * 2.**0.5
    guess_offset = np.mean(tide_level)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    # define the function to optimize
    def sinfunc(t, A, w, phi, offset):
        return A * np.sin(w*t + phi) + offset

    popt, _ = scipy.optimize.curve_fit(sinfunc, times, tide_level, p0=guess)

    A, w, phi, offset = popt
    f = w/(2.*np.pi)
    def fitted_function(new_time):
        return A * np.sin(w*new_time + phi) + offset

    print(f"amplitude : {A}")
    print(f"period : {1./f}")
    print(f"offset : {offset}")

    return fitted_function

fitted_function = fit_sinus(times, tide_level)

"""
    Part E : visually assess our optimized function
"""
predicted_tide_level=fitted_function(times)

plt.plot(times, tide_level, "o", label="measured data")
plt.plot(times, predicted_tide_level, label="model")
plt.legend(loc="best")
plt.title("tide level as a function of time")
plt.xlabel("time (hours)")
plt.ylabel("tide level (meters)")
plt.savefig("prediction.pdf")
plt.close()
