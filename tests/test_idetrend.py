import os
import numpy as np
import pandas as pd
# import settings as s
import idetrend as idtr

doy = 30
days_of_year = 365

# saved linear regression result from using data_to_detrend[:,3,5]
# from our tas test data and doy=30.
# LinregressResult(slope=3.3959676754498576, intercept=-711.0278021596743,
# rvalue=0.13115886566234206, pvalue=0.17200187865996122, stderr=2.469937744353457)
linregres_saved = np.array(
    [
        3.3959676754498576,
        -711.0278021596743,
        0.13115886566234206,
        0.17200187865996122,
        2.469937744353457,
    ]
)

test_path = os.path.dirname(__file__)

gmt_on_each_day = idtr.utility.get_gmt_on_each_day(
    os.path.join(test_path, "data/test_ssa_gmt.nc4"), days_of_year
)
tas_testdata = pd.read_csv(
    os.path.join(test_path, "data/tas_testdata.csv"), index_col=0, header=None
).squeeze()


def test_linear_regr():

    regr = idtr.lin_regr.regression(gmt_on_each_day)
    coeffs = regr.run(tas_testdata, doy, loni=0)
    np.testing.assert_allclose(np.array(coeffs), linregres_saved, rtol=1e-05)
