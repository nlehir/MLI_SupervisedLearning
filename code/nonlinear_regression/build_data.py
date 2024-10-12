import numpy as np

from constants import (
       NOISE_STD,
       SAMPLING_RATE,
       FREQUENCY,
       NB_PERIODS,
       SIGNAL_PERIOD,
       AMPLITUDE,
       OFFSET,
       INITIAL_PHASE,
        )

def real_function(
        time,
        noise_std=0.0,
        ):
    pulsation = 2 * np.pi * FREQUENCY
    sine_waveform = AMPLITUDE * np.sin(INITIAL_PHASE + time * pulsation)
    sine_waveform += AMPLITUDE / 2 * np.sin(INITIAL_PHASE + time * pulsation / 3)
    noise = np.random.normal(0, noise_std, sine_waveform.shape)
    tide_level = sine_waveform + noise + OFFSET
    return tide_level


def make_data():
    # measurement time in hours
    step = 1/SAMPLING_RATE
    time = np.arange(
            start=0,
            stop=NB_PERIODS * SIGNAL_PERIOD+step,
            step=step,
            )

    tide_level = real_function(
            time=time,
            noise_std=NOISE_STD,
            )

    print(f"{FREQUENCY=} Hz")
    print(f"{SAMPLING_RATE=} Hz")
    return time, tide_level
