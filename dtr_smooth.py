#!/home/bschmidt/.programs/anaconda3/envs/detrending/bin/python3.7


def dtr_smooth(data, hws=15):
    """This function computes running means with half window size hws.\n
    Package xarray is needed. \n
    Input:\tA XArray Dataset.\n
    Output:\tNetCDF4 file of daily values."""
    import numpy as np

    raw = data
    # get first variable of data
    var = list(raw.data_vars)[0]
    shp = raw[var].shape
    container = np.zeros(shape=shp)
    # loop over all datapoints and timesteps
    # calculate running mean with windowsize ws = 2*hws+1
    # store data as np.array in a container
    for i in range(raw['lat'].size):
        for j in range(raw['lon'].size):
            for time in range(shp[0])[(hws):(-(hws))]:
                raw_cut = raw[var][(time-hws):(time+hws), i, j]
                container[time, i, j] = np.mean(raw_cut)
            # pad beginning with first calculated value
            container[:hws, i, j] = container[hws+1, i, j]
            # pad end with last calculated window
            container[-hws:, i, j] = container[-(hws+1), i, j]
    smooth = raw.drop('wind')
    smooth["wind"] = (('time', 'lat', 'lon'), container)
    smooth.wind.attrs['long_name'] = 'running_mean_of_wind_speed_at_10_meter'
    smooth.wind.attrs['half_window_size'] = str(hws) + ' days'
    return smooth
