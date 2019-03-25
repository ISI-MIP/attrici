#!/home/bschmidt/.programs/anaconda3/envs/detrending/bin/python3.7
import numpy as np
import xarray as xr

def dtr_smooth(data, hws=15):
    """
    This function computes running means with half window size hws.\n
    Depends on package xarray. \n
    Input:\txarray Dataset or DataArray.\n
    Output:\tAs input format.
    """

    # datatype and convert to xarray.Dataset if xarray.DataArray is provided
    if type(data) == type(xr.Dataset()):
        pass
    elif type(data) == type(xr.DataArray(0)):
        data = xr.Dataset({str(data.name): data})
    else:
        raise TypeError('Input data must be of type ' + str(type(xr.Dataset()))
                        + ' or ' + str(type(xr.DataArray())))
    # get first variable of data
    var = list(data.data_vars)[0]
    shp = data[var].shape
                        
    container = np.zeros(shape=shp)
    # loop over all datapoints and timesteps
    # calculate running mean with windowsize ws = 2*hws+1
    # store data as np.array in a container
    if hasattr(data, 'lat'):
        for i in range(data['lat'].size):
            for j in range(data['lon'].size):
                for time in range(shp[0])[(hws):(-(hws))]:
                    data_cut = data[var][(time-hws):(time+hws), i, j]
                    container[time, i, j] = np.mean(data_cut)
                for time in range(hws):
                    # pad beginning with averages of same  day in next 5 years
                    container[time, i, j] = np.mean(container[(time+365):(time+(6*365)):365, i, j])
                    # pad end with last calculated window
                    container[-time, i, j] = np.mean(container[-(time+(6*365)):-(time+365):365, i, j])
        long_name = data[var].attrs['long_name']
        smooth = data.drop(var)
        smooth[var] = (('time', 'lat', 'lon'), container)
    else:
        for time in range(shp[0])[(hws):(-(hws))]:
            data_cut = data[var][(time-hws):(time+hws)]
            container[time] = np.mean(data_cut)
        for time in range(hws):
            # pad beginning with averages of same  day in next 5 years
            container[time] = np.mean(container[(time+365):(time+(6*365)):365])
            # pad end with last calculated window
            container[-time] = np.mean(container[-(time+(6*365)):-(time+365):365])
        smooth = data.drop(var)
        smooth[var] = (('time'), container)
        
    
    # smooth.wind.attrs['long_name'] = 'running_mean_of_' + long_name
    # smooth.wind.attrs['half_window_size'] = str(hws) + ' days'
    return smooth
