
import numpy as np

def logit(data):
    """ scaled logit for input data of values in [0, 100]
    as for rhs """
    return 2.0*np.arctanh(2.0*data/100 - 1.0)


def expit(data):
    """ scaled inverse for input data of values in [0, 100]
    as for rhs """

    return 100*0.5*(1.0 + np.tanh(0.5 * data))


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
    "pr": .000001157407, # amounts to .1 mm per day if unit is mm per sec
    "rhs": 0.0,
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
    "rhs": 101.0,
    "ps": None,
    "rsds": 3025.0,
    "rlds": 3025.0,
    "wind": None,
}
