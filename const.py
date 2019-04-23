#!/usr/bin/python


import numpy as np

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
