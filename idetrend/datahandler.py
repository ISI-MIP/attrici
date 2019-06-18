import numpy as np
import pandas as pd


def y_norm(y_to_scale, y_orig):
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())


def y_inv(y, y_orig):
    """rescale data y to y_original"""
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()


def create_dataframe(nct, data_to_detrend, gmt):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
    gmt_scaled = y_norm(gmt_on_data_cal, gmt_on_data_cal)
    y_scaled = y_norm(data_to_detrend, data_to_detrend)

    tdf = pd.DataFrame(
        {
            "ds": ds,
            "t": t_scaled,
            "y": data_to_detrend,
            "y_scaled": y_scaled,
            "gmt": gmt_on_data_cal,
            "gmt_scaled": gmt_scaled,
        }
    )

    return tdf


# def save_dataframe(settings, df):

#     df.to_csv()

# def mcs_helper(nct, data_to_detrend, gmt, variable, i, j):

#     data = data_to_detrend.variables[variable][:, i, j]
#     df = create_dataframe(nct, data, gmt)
#     return (df, i, j)
