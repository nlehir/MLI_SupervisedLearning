"""
    File illustrating the analysis of a time series.
    The time series stores measurements of the level
    of tide as a function of time.
"""

import math

from build_data import real_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from constants import (
        SIGNAL_PERIOD,
        SAMPLING_RATE_FACTOR,
        FREQUENCY,
        OFFSET,
        AMPLITUDE,
        INITIAL_PHASE,
        NB_PERIODS,
        NOISE_STD,
        DUPPLICATION_FACTOR,
        )
from build_data import make_data

FONTSIZE = 7
FIG_SIZE = (12, 9)


def clean_filename(text):
    text = text.replace(".", "_")
    return text


def analyze_signal(
        time,
        tide_level,
        fig,
        ax1,
        ax2,
        ax3,
        ) -> np.ndarray:
    time_spacing = time[1] - time[0] # hours
    sampling_rate = 1 / time_spacing
    nb_samples = len(time)

    """
    How can the reconstruction be the same with different sample frequencies ??
    """
    # frequency bins in cycles per hour
    sample_frequencies = np.fft.fftfreq(
        n=nb_samples,
        d=time_spacing,
    )
    frequency_spacing = sample_frequencies[1] - sample_frequencies[0]
    max_freq = np.max(sample_frequencies)
    min_freq = np.min(sample_frequencies)
    delta_freq = max_freq - min_freq
    max_time = np.max(time)
    min_time = np.min(time)
    delta_t = max_time - min_time

    fourier_transform = np.fft.fft(tide_level)
    """
        Why abs ?
    """
    guess_freq = abs(sample_frequencies[np.argmax(np.abs(fourier_transform[1:])) + 1])

    title = (
        f"sampling rate: {SAMPLING_RATE_FACTOR} frequency ({sampling_rate:.5f} samples per hour)\n"
        f"signal period: {SIGNAL_PERIOD} hours, frequency: {FREQUENCY:.5f} cycles per hour\n"
        f"number of samples: {nb_samples}, "
        f"{NB_PERIODS} periods\n"
        f"df: {frequency_spacing:.5f}, "
        # f"min freq: {min_freq:.5f}, max freq: {max_freq:.5f} cycles per hour\n"
        r"$\Delta$"
        f"f: {delta_freq:.5f} cycles per hour\n"
        f"dt: {time_spacing:.5f}, "
        r"$\Delta$"
        f"t: {delta_t:.5f} hours\n"
        r"$df=1/\Delta t$"
        ", "
        r"$\Delta f=(n-1)df=1/dt$"
        f"\noffset: {OFFSET} m, "
        f"amplitude: {AMPLITUDE} m, "
        f"initial phase: {INITIAL_PHASE * 180 / np.pi} °, "
        f"noise: {NOISE_STD}\n"
    )
    fig.suptitle(title, fontsize=FONTSIZE)

    time_linspace = np.linspace(
            start=min(time),
            stop=max(time),
            num=500,
            )
    tide_real = real_function(time=time_linspace)


    ax1.plot(
            time,
            tide_level,
            "o",
            markersize=5,
            alpha=0.8,
            label="measurements",
            )
    ax1.plot(
            time_linspace,
            tide_real,
            markersize=5,
            alpha=0.3,
            label="real function",
            color="blue"
            )
    ax1.set_ylabel("Tide level (m)", fontsize=FONTSIZE)
    ax1.set_xlabel("Time (hours)", fontsize=FONTSIZE)
    ax1.legend(loc="best", fontsize=FONTSIZE)
    ax1.set_title("Raw signal", fontsize=FONTSIZE)

    ax2.plot(
            sample_frequencies,
            # fourier_transform.real,
            np.abs(fourier_transform.real),
            "o",
            markersize=5,
            alpha=0.7,
            label="real",
            )
    ax2.plot(
            sample_frequencies,
            np.abs(fourier_transform.imag),
            "o",
            markersize=5,
            alpha=0.7,
            label="imag",
            )
    ax2.set_xlabel("Frequency (Cycles per hour)", fontsize=FONTSIZE)
    ax2.set_ylabel("Fourier transform modulus", fontsize=FONTSIZE)
    ax2.legend(loc="best", fontsize=FONTSIZE)
    ax2.set_yscale("log")
    title = (
        "Fourier transform modulus\n"
        f"Most present freq: {guess_freq:.5f} cycles per hour\n"
        f"Period: {1/guess_freq:.5f} Hours"
    )
    ax2.set_title(title, fontsize=FONTSIZE)

    reconstruction = np.fft.ifft(
            a=fourier_transform,
            n=nb_samples,
            )
    difference_norm = np.linalg.norm(reconstruction-tide_level)

    """
    Respect the numpy convention of the sorting
    """
    fourier_transform_padded = np.zeros(
            shape=nb_samples + (DUPPLICATION_FACTOR-1)*(nb_samples-1),
            dtype=np.complex64,
            )
    fourier_transform_padded[:nb_samples//2] = fourier_transform[:nb_samples//2]
    fourier_transform_padded[-nb_samples//2:] = fourier_transform[-nb_samples//2:]

    """
    Probably a normalization involved: TODO: double check
    """
    reconstruction_extended = DUPPLICATION_FACTOR * np.fft.ifft(
            a=fourier_transform_padded,
            )

    extended_time = np.linspace(
            start=0,
            stop=max(time),
            num=len(reconstruction_extended),
            endpoint=True,
            )

    ax3.plot(time, tide_level, "o", label="raw signal", markersize=5, alpha=0.2)

    """
    Imag parts are negligible
    """
    ax3.plot(time, reconstruction.real, "o", label="reconstruction", markersize=5, alpha=0.7)
    ax3.plot(extended_time, reconstruction_extended.real, "o", label="reconstruction extended", markersize=5, alpha=0.2)
    tide_real = real_function(time=time_linspace)
    ax3.plot(
            time_linspace,
            tide_real,
            markersize=5,
            alpha=0.3,
            label="real function",
            color="blue"
            )
    ax3.legend(loc="best", fontsize=FONTSIZE)
    ax3.set_xlabel("Time (hours)", fontsize=FONTSIZE)
    ax3.set_ylabel("Tide level (hours)", fontsize=FONTSIZE)
    title = (
        "Reconstructed signal\n"
        f"Difference: {difference_norm:.3E}"
            )
    ax3.set_title(title, fontsize=FONTSIZE)


    guess_amp = np.std(tide_level) * 2.0**0.5
    guess_offset = np.mean(tide_level)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
    return guess

def fit_sinus(
        X,
        y,
        guess,
        ) -> dict:

    # define the function to optimize
    def sinfunc(t, A, w, phi, offset):
        return A * np.sin(w * t + phi) + offset

    popt, _ = scipy.optimize.curve_fit(sinfunc, X, y, p0=guess)
    # popt, _ = scipy.optimize.curve_fit(sinfunc, X, y)

    amplitude, pulsation, initial_phase, offset = popt
    frequency = pulsation / (2.0 * math.pi)

    def fitted_function(t):
        return amplitude * np.sin(pulsation * t + initial_phase) + offset

    print(f"amplitude : {amplitude}")
    print(f"period : {1./frequency}")
    print(f"offset : {offset}")

    return dict(
            amplitude=amplitude,
            frequency=frequency,
            initial_phase=initial_phase,
            offset=offset,
            fitted_function=fitted_function,
            )


def main():
    time, tide_level = make_data()
    """
        Plot of the tide lebel as a function of time.
    """
    plt.plot(time, tide_level, "o", markersize=4, alpha=0.4)
    plt.title("tide level as a function of time")
    plt.xlabel("time (hours)", fontsize=FONTSIZE)
    plt.ylabel("tide level (meters)", fontsize=FONTSIZE)
    plt.savefig("tide_level.pdf")
    plt.close()

    """
        Fit a function to these data
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=FIG_SIZE)
    guessed_params = analyze_signal(
            time=time,
            tide_level=tide_level,
            fig=fig,
            ax1=ax1,
            ax2=ax2,
            ax3=ax3,
            )
    time_train, time_test, tide_level_train, tide_level_test = train_test_split(time, tide_level, test_size=0.33)
    # fitted_function = fit_sinus(X=time_train, y=tide_level_train, guess=guessed_params)
    returned_dict = fit_sinus(X=time, y=tide_level, guess=guessed_params)
    fitted_function = returned_dict["fitted_function"]

    """
        Visually assess our optimized function
    """
    predicted_tide_level_train = fitted_function(time_train)
    predicted_tide_level_test = fitted_function(time_test)
    time_linspace = np.linspace(start=min(time), stop=max(time), num=400)
    predicted_tide_level_linspace = fitted_function(time_linspace)

    train_score = r2_score(predicted_tide_level_train, tide_level_train)
    """
    If the sampling is loose, it might happen that all the
    values in tide_level_test are equal, leading to a bad test R2 !
    """
    test_score = r2_score(predicted_tide_level_test, tide_level_test)

    ax4.plot(
        time_train, tide_level_train, "o", label="train data", alpha=0.6, markersize=4, color="blue"
    )
    ax4.plot(
        time_test, tide_level_test, "x", label="test data", alpha=0.6, markersize=4, color="green"
    )
    ax4.plot(
        time_linspace,
        predicted_tide_level_linspace,
        label="model",
        color="orange",
        markersize=1,
        alpha=0.5,
    )
    ax4.legend(loc="best", fontsize=FONTSIZE)
    title = (
        "found parameters: "
        f"amplitude = {returned_dict['amplitude']:.3f} m, "
        f"period = {1/returned_dict['frequency']:.3f} hours, "
        f"offset = {returned_dict['offset']:.3f} m, "
        f"initial_phase = {returned_dict['initial_phase']*180/np.pi:.3f} °\n"
        f"train r2: {train_score:.3E}, "
        f"test r2: {test_score:.3E}"
            )
    ax4.set_title(
            label=title,
            fontsize=FONTSIZE,
    )
    ax4.set_xlabel("time (hours)", fontsize=FONTSIZE)
    ax4.set_ylabel("tide level (meters)", fontsize=FONTSIZE)

    plt.tight_layout()
    fig_name = f"fourier_transform_sampling_{SAMPLING_RATE_FACTOR:.1f}_freq_{NB_PERIODS}_periods_noise_std_{NOISE_STD}"
    plt.savefig(f"{clean_filename(fig_name)}.pdf")
    plt.close()


if __name__ == "__main__":
    main()
