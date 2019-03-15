import numpy as np
from scipy import stats


def linear_regr_per_gridcell(np_data_to_detrend, gmt_on_each_day, doy, loni):

    """ minimal version of a linear regression per grid cell """

    data_of_doy = np_data_to_detrend[doy::365,loni]
    gmt_of_doy = gmt_on_each_day[doy::365]
    return  stats.linregress(gmt_of_doy,data_of_doy)


def run_lat_slice_serial(lat_slice_data, gmt_on_each_day, days_of_year):

    """ calculate linear regression stats for all days of years and all grid cells.
    classic serial looping. Return a list of all stats """

    results = []
    for doy in np.arange(days_of_year):
        for loni in np.arange(lat_slice_data.shape[1]):
            result = linear_regr_per_gridcell(
                        lat_slice_data,gmt_on_each_day,doy,loni)
            results.append(result)

    return results


if __name__ == "__main__":

    """ add a quick test for one grid cell of our regression algorithm here. """

    pass
