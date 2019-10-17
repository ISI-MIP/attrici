import numpy as np


def series(t, p, modes):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, modes + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


def rescale(df, modes):
    """ This function computes a scaled (0, 1) fourier series for a given input dataset.
    An input vector of dates ("ds") must be available in a datestamp format.
    If the time vector has gaps (due to dropped NA's), the fourier series will
     also contain gaps (jumps in value).
    The output format will be of [len["ds"], 2*modes], where the first
    half of the columns contains the cos(x)-series and die latter half
    contains the sin(x)-series
    """

    # rescale the period, as t is also scaled
    p = 365.25 / (df["ds"].max() - df["ds"].min()).days
    x = series(df["t"], p, modes)
    return x

def get_fourier_valid(df, valid_index, modes):

    x_fourier = []
    for mode in modes:
        xf = rescale(df, mode)
        x_fourier.append(xf[valid_index, :])

    return x_fourier
