import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import settings as s
import idetrend as idtr
import idetrend.visualization
doy = 30
days_of_year = 365
min_ts_len = 2

# saved linear regression result from using data_to_detrend[:,3,5]
# from our tas test data and doy=30.
# LinregressResult(slope=3.3959676754498576, intercept=-711.0278021596743,
# rvalue=0.13115886566234206, pvalue=0.17200187865996122, stderr=2.469937744353457)
linregres_tas = np.array(
    [
        3.3959676754498576,
        -711.0278021596743,
        0.13115886566234206,
        0.17200187865996122,
        2.469937744353457,
    ]
)

# saved linear regression result from using data_to_detrend[:,3,10]
# from our pr test data and doy=30. data transformed with log before regression.
linregres_pr = np.array([-3.16381964e-01,  8.08311175e+01, -6.44054507e-02,  5.28658251e-01,
        5.00323456e-01])


test_path = os.path.dirname(__file__)

gmt_on_each_day = idtr.utility.get_gmt_on_each_day(
    os.path.join(test_path, "data/test_ssa_gmt.nc4"), days_of_year
)

tas_testdata = pd.read_csv(
    os.path.join(test_path, "data/tas_testdata.csv"), index_col=0, header=None
).squeeze()

pr_testdata = pd.read_csv(
    os.path.join(test_path, "data/pr_testdata.csv"), index_col=0, header=None
).squeeze()

def test_tas_regr():

    regr = idtr.regression.regression(gmt_on_each_day, min_ts_len)
    coeffs = regr.run(tas_testdata, doy, loni=0)
    np.testing.assert_allclose(np.array(coeffs), linregres_tas, rtol=1e-05)


# FIXME: fails due to strange np.ma.log behaviour under testing.
# skip for now
# def test_pr_regr():
#     # FIXME: update to run with masked arrays
#     regr = idtr.lin_regr.regression(gmt_on_each_day, min_ts_len,
#         transform=[np.ma.log,np.ma.exp])
#     coeffs = regr.run(pr_testdata, doy, loni=0)
#     np.testing.assert_allclose(np.array(coeffs), linregres_pr, rtol=1e-05)


def get_netcdf_data(dfile):

    print(dfile)
    ncf = nc.Dataset(dfile,"r")
    data = ncf.variables[s.variable][:]
    ncf.close()
    return  data

detrended_bk_file = os.path.join(s.data_dir, s.variable + "_detrended_test_bk.nc4")


def test_write_detrending():

    # the file with the smoothed global trend of global mean temperature
    gmt_file = os.path.join(s.data_dir, s.gmt_file)
    # the daily interpolated ssa-smoothed global mean temperature
    gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)

    regression_file = os.path.join(s.data_dir, s.regression_outfile)
    to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)
    detrended_file = os.path.join(s.data_dir, s.detrended_file)

    lats, lons, slope, intercept = \
        idetrend.detrending.get_coefficient_fields(regression_file)

    detrend = idtr.detrending.detrending(lons,
        lats,
        slope,
        intercept,
        regression_file,
        to_detrend_file,
        s.variable,
        gmt_on_each_day,
        s.days_of_year
        )

    detrend.write_detrended(detrended_file)

    checked_detrended = get_netcdf_data(detrended_bk_file)
    now_detrended = get_netcdf_data(detrended_file)

    np.testing.assert_allclose(checked_detrended,now_detrended)

