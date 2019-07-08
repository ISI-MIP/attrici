import numpy as np

def standard(y_to_scale, y_orig):
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

def precip(y_to_scale, y_orig):
    """scale and transform data with lower boundary y to y_original"""
    y_to_scale[y_to_scale<=0.000001157407] = 0   # amounts to .1 mm per day if unit is mm per sec
    y_to_scale = np.log(y_to_scale)
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

def rhs(y_to_scale, y_orig):
    """scale and transform data with lower boundary y to y_original"""
    y_to_scale = 2.0 * np.ma.arctanh(2.0 * y_to_scale / (100 - 0) - 1.0)
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

def wind(y_to_scale, y_orig):
    """scale and transform wind data with lower boundary y to y_original"""
    y_to_scale = np.log(y_to_scale)
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

transform_dict = {
    "tasmin": standard,
    "tas": standard,
    "tasmax": standard,
    "pr": precip,
    "rhs": rhs,
    "ps": standard,
    "rsds": standard,
    "rlds": standard,
    "wind": wind,
}

def rescale(y, y_orig):
    """rescale "standard" data y to y_original"""
    return y * (y_orig.max() - y_orig.min())

def re_standard(y, y_orig):
    """standard" data y to y_original"""
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()

def re_precip(y, y_orig):
    """rescale and transform data with lower boundary y to y_original"""
    y = np.exp(y)
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()

def re_rhs(y, y_orig):
    """ scaled inverse logit for input data of values in [0, 100]
    as for rhs. minval and maxval differ by purpose from these
    in dictionaries below."""
    y = 100 * 0.5 * (1.0 + np.ma.tanh(0.5 * y))
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()

def re_wind(y, y_orig):
    """rescale and transform data with lower boundary y to y_original"""
    y = np.exp(y)
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()

#  set of inverse transform functions
retransform_dict = {
    "tasmin": re_standard,
    "tas": re_standard,
    "tasmax": re_standard,
    "pr": re_precip,
    "rhs": re_rhs,
    "ps": re_standard,
    "rsds": re_standard,
    "rlds": re_standard,
    "wind": re_wind,
}

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
#  maxval = {
#      "tasmax": None,
#      "tas": None,
#      "tasmin": None,
#      "pr": None,
#      "rhs": 99.9,
#      "ps": None,
#      "rsds": 3025.0,
#      "rlds": 3025.0,
#      "wind": None,
#  }
