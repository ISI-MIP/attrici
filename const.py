#!/usr/bin/python


import numpy as np


varsInFile = np.array(
    [
        "ta",
        "mo",
        "jahr",
        "tmax",
        "tmit",
        "tmin",
        "nied",
        "relf",
        "ludr",
        "dadr",
        "sonn",
        "bewo",
        "stra",
        "wind",
    ]
)
varsToDetrend = np.array(
    ["tmax", "tmit", "tmin", "nied", "relf", "ludr", "stra", "wind"]
)
indexVarsToDetrend = np.array(
    [np.where(varsInFile == varToDetrend)[0][0] for varToDetrend in varsToDetrend]
)

unit = {
    "tasmax": "K",
    "tas": "K",
    "tasmin": "K",
    "pr": "mm",
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
    "pr": 0.0,
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
