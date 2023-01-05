import numpy as np
import math
import matplotlib.pyplot as plt

nb_points = 10000

# noise standard deviation
noise_std = 0.5

# amplitude in meters
amplitude = 15

# beginning and end of experiment in hours
start_time = 0
end_time = 96
step = (end_time-start_time)/nb_points

# measurement times in hours
times = np.arange(start_time, end_time, step)

period = 7
frequence = 1/period
pulsation = 2*math.pi*frequence

sine_waveform = np.sin(times*pulsation)

noise = np.random.normal(0, noise_std, sine_waveform.shape)

offset = 5

# noisy measurements
tide_level = sine_waveform + noise + offset

data = np.column_stack((times, tide_level))

np.savetxt("data.csv", data, delimiter=",")
