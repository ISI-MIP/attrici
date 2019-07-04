import numpy as np

def standard(y_to_scale, y_orig):
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

def precip(y_to_scale, y_orig):
    """scale and transform data with lower boundary y to y_original"""
    y = np.exp(y)
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

def logit(y_to_scale, y_orig):
    """scale and transform data with lower boundary y to y_original"""
    y = 2.0 * np.ma.arctanh(2.0 * data / (maxval - minval) - 1.0)
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())

def expit(data):
    """ scaled inverse logit for input data of values in [0, 100]
    as for rhs. minval and maxval differ by purpose from these
    in dictionaries below."""
    minval = 0
    maxval = 100
    return minval + (maxval - minval) * 0.5 * (1.0 + np.ma.tanh(0.5 * data))

transform_dict = {
    "tasmin": standard,
    "tas": standard,
    "tasmax": standard,
    "pr": precip,
    "rhs": ,
    "ps": None,
    "rsds": None,
    "rlds": None,
    "wind": [np.ma.log, np.ma.exp],
}

def rescale(y, y_orig):
    """rescale "standard" data y to y_original"""
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()

def re_standard(y, y_orig):
    """standard" data y to y_original"""
    return y * (y_orig.max() - y_orig.min())

def re_precip(y, y_orig):
    """rescale and transform data with lower boundary y to y_original"""
    y[y<=0.000001157407] = 0   # amounts to .1 mm per day if unit is mm per sec
    y = np.log(y)
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()

def logit(data):
    """ scaled logit for input data of values in [0, 100]
    as for rhs. minval and maxval differ by purpose from these
    in dictionaries below. """
    minval = 0
    maxval = 100
    return 2.0 * np.ma.arctanh(2.0 * (data - minval) / (maxval - minval) - 1.0)

#  set of inverse transform functions
retransform_dict = {
    "tasmin": re_standard,
    "tas": re_standard,
    "tasmax": re_standard,
    "pr": re_precip,
    "rhs": [logit, expit],
    "ps": None,
    "rsds": None,
    "rlds": None,
    "wind": [np.ma.log, np.ma.exp],
}



# transformations come in tuples
# [transform, inverse_transform]
transform = {
    "tasmin": None,
    "tas": None,
    "tasmax": None,
    "pr": [np.ma.log, np.ma.exp],
    "rhs": [logit, expit],
    "ps": None,
    "rsds": None,
    "rlds": None,
    "wind": [np.ma.log, np.ma.exp],
}

unit = {
    "tasmax": "K",
    "tas": "K",
    "tasmin": "K",
    "pr": "mm/s",
    "rhs": "%",
    "ps": "hPa",
    "rsds": u"J/cm\u00B2",
    "rlds": u"J/cm\u00B2",
    "wind": "m/s",
}

minval = {
    "tasmax": None,
    "tas": None,
    "tasmin": None,
    "pr": 0.000001157407,  # amounts to .1 mm per day if unit is mm per sec
    "rhs": 0.01,
    "ps": None,
    "rsds": 0.0,
    "rlds": 0.0,
    "wind": 0.0,
}

maxval = {
    "tasmax": None,
    "tas": None,
    "tasmin": None,
    "pr": None,
    "rhs": 99.9,
    "ps": None,
    "rsds": 3025.0,
    "rlds": 3025.0,
    "wind": None,
}
