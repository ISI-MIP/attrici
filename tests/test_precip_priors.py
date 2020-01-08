import numpy as np
import pandas as pd
import pymc3 as pm
import icounter.gammabernoulli as gb
import pathlib
import icounter.fourier as fourier
import icounter.datahandler as dh
import settings as s

cwd = pathlib.Path.cwd()
df = pd.read_hdf(cwd/"tests"/"data"/"ts_GSWP3_lat-20.25_lon-49.75.h5")

x_fourier = fourier.get_fourier_valid(df, s.modes)
concat_notdone = True
if concat_notdone:
    df = pd.concat([df,x_fourier], axis=1)
    concat_notdone = False

df_valid = dh.get_valid_subset(df, s.subset, s.seed)


def test_precip_longerm_priors():

    # modes are dummy in the longterm case
    model = gb.PrecipitationLongterm([1,1,1,1],"longterm","longterm","longterm")
    model.test = True
    smodel = model.setup(df_valid, df)

    with smodel:
        trace = pm.sample()

    for param, bounds in model.parameter_bounds.items():

        assert trace[param].min() < trace[param].max()

        if bounds[0] is not None:
            assert trace[param].min() > bounds[0]
        else:
            assert trace[param].min() < -5

        if bounds[1] is not None:
            assert trace[param].max() < bounds[1]
        else:
            assert trace[param].max() > 5

    assert np.abs(trace["a_sigma"].mean()) < 0.01, "mode of a_sigma should be close to zero."
