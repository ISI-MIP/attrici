import os
import numpy as np
import iris
import iris.coord_categorisation as icc
# from joblib import dump, load
import joblib
# import multiprocessing
# import psutil
from scipy import stats
# import shutil
# import sys
from datetime import datetime
import settings as s

gmt_file = os.path.join(s.data_dir, s.gmt_file)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

gmt = iris.load_cube(gmt_file)
data_to_detrend = iris.load_cube(to_detrend_file)
icc.add_day_of_year(data_to_detrend, 'time')
doys_cube = data_to_detrend.coord('day_of_year').points

# remove 366th day for now. do this more exact later.
np_data_to_detrend = np.array(data_to_detrend[doys_cube != 366].data)

days_of_year = 365
# interpolate monthly gmt values to daily.
# do this more exact later.
gmt_on_each_day = np.interp(np.arange(110*days_of_year),
                            gmt.coord("time").points,gmt.data)

def run_parallel_linear_regr():
    
    """ calculate linear regression stats for all days of years and all grid cells.
    joblib implementation. Return a list of all regression stats. """

    latis = np.arange(np_data_to_detrend.shape[1])
    lonis = np.arange(np_data_to_detrend.shape[2])
    doys = np.arange(days_of_year)

    results = joblib.Parallel(n_jobs=3)(
                joblib.delayed(linear_regr_per_gridcell)(
                    np_data_to_detrend,gmt_on_each_day,doy,lati,loni)
                        for doy in doys for lati in latis for loni in lonis)
    return results



if __name__ == "__main__":

    FTIME = datetime.now()
    run_parallel_linear_regr()
    duration = datetime.now() - FTIME
    print('This took', duration.total_seconds(), 'seconds.')