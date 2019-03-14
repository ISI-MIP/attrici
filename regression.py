import numpy as np
from scipy import stats


def linear_regr_per_gridcell(np_data_to_detrend,gmt_on_each_day,doy,lati,loni):
    
    """ minimal version of a linear regression per grid cell """
    
    data_of_doy = np_data_to_detrend[doy::365,lati,loni]
    gmt_of_doy = gmt_on_each_day[doy::365]
    return  stats.linregress(gmt_of_doy,data_of_doy)


def run_serial_linear_regr():
    
    """ calculate linear regression stats for all days of years and all grid cells.
    classic serial looping. Return a list of all regression stats. """
    
    results = []
    for doy in np.arange(days_of_year):
        for lati in np.arange(np_data_to_detrend.shape[1]):
            for loni in np.arange(np_data_to_detrend.shape[2]):
                result = linear_regr_per_gridcell(
                            np_data_to_detrend,gmt_on_each_day,doy,lati,loni)
                results.append(result)
                
    return results


