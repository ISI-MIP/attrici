import numpy as np


threshold = {
    "tas": (0,None),
    "tasrange": (0.01,None),
    "tasskew": (0.0001, 0.9999),
    "pr": (0.0000011574,None),
    "prsnratio": (0.01, 0.99),
    "rhs": (0.01, 99.99),
    "ps": (0,None),
    "rsds": (0,None),
    "rlds": (0,None),
    "wind": (0.01,),
}

bound = {
    "tas": (0.0,None),
    "tasrange": (0.0,None),
    "tasskew": (0.0, 1.),
    "pr": (0.0,None),
    "prsnratio": (0.0,1.0),
    "rhs": (0.01, 99.99),
    "ps": (0,None),
    "rsds": (0,1),
    "rlds": (0,None),
    "wind": (0.0,None),
}


def check_bounds(data, variable):

    lower = bound[variable][0]
    upper = bound[variable][1]

    if lower is not None and data.min() < lower:
        raise ValueError(data.min(), "is smaller than lower bound",lower,".")

    if upper is not None and data.max() > upper:
        raise ValueError(data.max(), "is bigger than upper bound",upper,".")


def scale_to_unity(data, variable):

    """ Take a pandas Series and scale it linearly to
    lie within [0, 1]. Return pandas Series as well as the
    data minimum and the scale. """

    scale = data.max() - data.min()
    scaled_data = (data - data.min()) / scale

    return scaled_data, data.min(), scale


def rescale_to_original(scaled_data, datamin, scale):

    """ Use a given datamin and scale to rescale to original. """

    return scaled_data * scale + datamin


def scale_to_unity_and_mask(data, variable):

    print("Mask", (data <= threshold[variable][0]).sum(),"values below lower bound.")
    data[data <= threshold[variable][0]] = np.nan
    print("Mask", (data >= threshold[variable][1]).sum(),"values above upper bound.")
    data[data >= threshold[variable][1]] = np.nan

    scale = data.max() - data.min()
    scaled_data = (data) / scale

    return scaled_data, data.min(), scale



def mask_and_scale_precip(data, variable):

    pr_thresh = 0.000001157407  # 0.1 mm per day

    # get scale and datamin before masking
    scale = data.max() - data.min()
    datamin = data.min()
    masked_data = data.copy()
    masked_data[masked_data < pr_thresh] = np.nan
    # do not use datamin to shift data to avoid zero
    scaled_data = masked_data / scale

    return scaled_data, datamin, scale


def refill_and_rescale_precip(scaled_data, datamin, scale):

    return scaled_data * scale


mask_and_scale = {
    "gmt": [scale_to_unity, rescale_to_original],
    "tas": [scale_to_unity, rescale_to_original],
    "ps": [scale_to_unity, rescale_to_original],
    "rlds": [scale_to_unity, rescale_to_original],
    "wind": [scale_to_unity, rescale_to_original],
    "tasrange": [scale_to_unity_and_mask, refill_and_rescale_precip],
    "pr": [mask_and_scale_precip, refill_and_rescale_precip],
}


# def y_norm(y_to_scale, y_orig):
#     return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())


# def y_inv(y, y_orig):
#     """rescale data y to y_original"""
#     return y * (y_orig.max() - y_orig.min()) + y_orig.min()


# def scale(y_to_scale, y_orig, gmt=False):
#     if gmt:
#         lower = 0
#     if not gmt:
#         if len(threshold[s.variable]) <= 2:
#             lower = threshold[s.variable][0]
#             print("got lower bound")
#             y_to_scale[y_to_scale <= lower] = np.nan
#         if len(threshold[s.variable]) == 2:
#             upper = threshold[s.variable][1]
#             print("got upper bound")
#             y_to_scale[y_to_scale >= upper] = np.nan
#     return (y_to_scale - np.nanmin(y_orig) + lower) / (
#         np.nanmax(y_orig) - np.nanmin(y_orig)
#     )


# def precip(y_to_scale, y_orig):
#     """scale and transform data with lower boundary y to y_original"""
#     y_to_scale[
#         y_to_scale <= 0.000001157407
#     ] = 0  # amounts to .1 mm per day if unit is mm per sec
#     y_to_scale = np.log(y_to_scale)
#     return scale(y_to_scale, y_orig)


# # FIXME: double check
# def rhs(y_to_scale, y_orig):
#     """scale and transform data with lower boundary y to y_original"""
#     y_to_scale = 2.0 * np.ma.arctanh(2.0 * y_to_scale / (100 - 0) - 1.0)
#     return scale(y_to_scale, y_orig)


# def wind(y_to_scale, y_orig):
#     """scale and transform wind data with lower boundary y to y_original"""
#     y_to_scale = np.log(y_to_scale)
#     return (y_to_scale - np.nanmin(y_orig)) / (np.nanmax(y_orig) - np.nanmin(y_orig))


# transform_dict = {
#     "tasskew": scale,
#     "tasrange": scale,
#     "tas": scale,
#     "pr": precip,
#     "prsn_rel": precip,
#     "rhs": rhs,
#     "ps": None,
#     "rsds": None,
#     "rlds": None,
#     "wind": wind,
# }


# def rescale(y, y_orig):
#     """rescale "standard" data y to y_original"""
#     return y * (np.nanmax(y_orig) - np.nanmin(y_orig))


# def re_standard(y, y_orig):
#     """standard" data y to y_original"""
#     return rescale(y, y_orig) + np.nanmin(y_orig)


# def re_precip(y, y_orig):
#     """rescale and transform data with lower boundary y to y_original"""
#     return np.exp(y)


# # FIXME: double check
# def re_rhs(y, y_orig):
#     """ scaled inverse logit for input data of values in [0, 100]
#     as for rhs. minval and maxval differ by purpose from these
#     in dictionaries below."""
#     return 100 * 0.5 * (1.0 + np.ma.tanh(0.5 * y))


# # FIXME: code doubling!
# def re_wind(y, y_orig):
#     """rescale and transform data with lower boundary y to y_original"""
#     return np.exp(y)


# #  set of inverse transform functions
# retransform_dict = {
#     "tasskew": re_standard,
#     "tasrange": re_standard,
#     "tas": re_standard,
#     "pr": re_precip,
#     "prsn_rel": re_precip,
#     "rhs": re_rhs,
#     "ps": re_standard,
#     "rsds": re_standard,
#     "rlds": re_standard,
#     "wind": re_wind,
# }


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
