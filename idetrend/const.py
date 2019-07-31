import numpy as np
import settings as s

threshold = {
    "tas": (0, ),
    "tasrange": (.01, ),
    "tasskew": (.0001, .9999),
    "pr": (.0000011574, ),
    "prsnratio": (.01, .99),
    "rhs": (.01, 99.99),
    "ps": (0, ),
    "rsds": (0, ),
    "rlds": (0, ),
    "wind": (0.01, ),
}

def scale(y_to_scale, y_orig, gmt=False):
    if gmt:
        lower = 0
    if not gmt:
        if len(threshold[s.variable]) <= 2:
            lower = threshold[s.variable][0]
            print("got lower bound")
            y_to_scale[y_to_scale <= lower] = np.nan
        if len(threshold[s.variable]) == 2:
            upper = threshold[s.variable][1]
            print("got upper bound")
            y_to_scale[y_to_scale >= upper] = np.nan
    return (y_to_scale - np.nanmin(y_orig) + lower) / (np.nanmax(y_orig) - np.nanmin(y_orig))


def precip(y_to_scale, y_orig):
    """scale and transform data with lower boundary y to y_original"""
    y_to_scale[
        y_to_scale <= 0.000001157407
    ] = 0  # amounts to .1 mm per day if unit is mm per sec
    y_to_scale = np.log(y_to_scale)
    return scale(y_to_scale, y_orig)


# FIXME: double check
def rhs(y_to_scale, y_orig):
    """scale and transform data with lower boundary y to y_original"""
    y_to_scale = 2.0 * np.ma.arctanh(2.0 * y_to_scale / (100 - 0) - 1.0)
    return scale(y_to_scale, y_orig)


def wind(y_to_scale, y_orig):
    """scale and transform wind data with lower boundary y to y_original"""
    y_to_scale = np.log(y_to_scale)
    return (y_to_scale - np.nanmin(y_orig)) / (np.nanmax(y_orig) - np.nanmin(y_orig))


transform_dict = {
    "tasskew": scale,
    "tasrange": scale,
    "tas": scale,
    "pr": precip,
    "prsn_rel": precip,
    "rhs": rhs,
    "ps": None,
    "rsds": None,
    "rlds": None,
    "wind": wind,
}


def rescale(y, y_orig):
    """rescale "standard" data y to y_original"""
    return y * (np.nanmax(y_orig) - np.nanmin(y_orig))


def re_standard(y, y_orig):
    """standard" data y to y_original"""
    return rescale(y, y_orig) + np.nanmin(y_orig)


def re_precip(y, y_orig):
    """rescale and transform data with lower boundary y to y_original"""
    return np.exp(y)


# FIXME: double check
def re_rhs(y, y_orig):
    """ scaled inverse logit for input data of values in [0, 100]
    as for rhs. minval and maxval differ by purpose from these
    in dictionaries below."""
    return 100 * 0.5 * (1.0 + np.ma.tanh(0.5 * y))


# FIXME: code doubling!
def re_wind(y, y_orig):
    """rescale and transform data with lower boundary y to y_original"""
    return np.exp(y)


#  set of inverse transform functions
retransform_dict = {
    "tasskew": re_standard,
    "tasrange": re_standard,
    "tas": re_standard,
    "pr": re_precip,
    "prsn_rel": re_precip,
    "rhs": re_rhs,
    "ps": re_standard,
    "rsds": re_standard,
    "rlds": re_standard,
    "wind": re_wind,
}

def fourier_series(t, p, modes):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, modes + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


def rescale_fourier(df, modes):
    """ This function computes a scaled (0, 1) fourier series for a given input dataset.
    An input vector of dates ("ds") must be available in a datestamp format.
    If the time vector has gaps (due to dropped NA's), the fourier series will also contain gaps (jumps in value).
    The output format will be of [len["ds"], 2*modes], where the first half of the columns contains the cos(x)-series and die latter half
    contains the sin(x)-series
    """

    # rescale the period, as t is also scaled
    p = 365.25 / (df["ds"].max() - df["ds"].min()).days
    x = fourier_series(df["t"], p, modes)
    return x

######## Not needed but kept for possible later use ####
#  unit = {
#      "tasmax": "K",
#      "tas": "K",
#      "tasmin": "K",
#      "pr": "mm/s",
#      "rhs": "%",
#      "ps": "hPa",
#      "rsds": u"J/cm\u00B2",
#      "rlds": u"J/cm\u00B2",
#      "wind": "m/s",
#  }
#
