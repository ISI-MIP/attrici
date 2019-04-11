import os
import numpy as np
import re
import warnings
import netCDF4 as nc
from scipy import special


def check_data(data, source_path_data):

    """ This function transforms slices as a
    precprocessing step of data for regression analysis.
    NOTE: add more check routines"""

    # get variable name from filename
    var_list = ['rhs', 'hurs', 'huss', 'snowfall', 'ps', 'rlds',
                'rsds', 'wind', 'tasmin', 'tasmax', 'tas']
    # get data folder
    data_folder = os.path.join('/', *source_path_data.split('/')[:-1])

    # fet filename
    data_file = source_path_data.split('/')[-1]

    # get variable (tas needs to be last in var_list to avoid wrong detection
    # when tasmax or tasmin are searched
    variable = [variable for variable in var_list if variable in data_file][0]

    if isinstance(data, nc.Dataset):
        data = data.variables['rhs'][:]

    for lat_slice in range(data.shape[1]):
        # run different transforms for variables
        if variable in ['rhs', 'hurs']:
            # print('Data variable is relative humidity!')
            # raise warnings if data contains values that need replacing
            if np.sum(np.logical_or(data < 0, data > 110)) > 0:
                #  warnings.warn('Data contains values far out of range. ' +
                #                'Replacing values by numbers just in range!')
                data[data > 110] = 99.9999
                data[data < 0] = .0001
                warnings.warn('Had to replace values that are far out of range!' +
                              '\nData seems to be erroneous!')
            if np.sum(np.logical_or(data > 100, data < 110)) > 0:
                #  warnings.warn('Some values suggest oversaturation. ' +
                              #  'Replacing values by numbers just in range!')
                data[data > 100] = 99.9999
                warnings.warn('Replaced values of oversaturation!')
            if np.sum(np.logical_or(data == 0, data == 100)) > 0:
                #  warnings.warn('Data contains values 0 and/or 100. ' +
                #                '\nReplacing values by numbers just in range!')
                data[data == 100] = 99.9999
                data[data == 0] = .0001
                warnings.warn('Replaced values bordering open interval (0, 100)!')

        elif variable == 'tas':
            print('Data variable is daily mean temperature!')

        elif variable == 'tasmax':
            print('Data variable is daily maximum temperature!')
            tas_file = os.path.join(data_folder,
                                    re.sub(variable,
                                           'tas',
                                           source_path_data.split('/')[-1]))
            tas = nc.Dataset(tas_file, "r").variables['tas'][:, lat_slice, :]

            # check values of data

            if np.min(data - tas) <= 0:
                raise Exception('\nAt least one value of tasmax appears to be ' +
                                'smaller than corresponding tas value!')

        elif variable == 'tasmin':
            print('Data variable is daily minimum temperature!')
            tas_file = os.path.join(data_folder,
                                    re.sub(variable,
                                           'tas',
                                           source_path_data.split('/')[-1]))
            tas = nc.Dataset(tas_file, "r").variables['tas'][:, lat_slice, :]

            # check values of data
            if np.min(tas - data) <= 0:
                raise Exception('\nAt least one value of tasmin appears to be ' +
                                'larger than corresponding tas value!')

        elif variable == 'snowfall':
            pr_file = os.path.join(data_folder,
                                   re.sub(variable,
                                          'pr',
                                          source_path_data.split('/')[-1]))
            pr = nc.Dataset(pr_file, "r").variables['pr'][:, lat_slice, :]
            if np.sum(variable >= pr) > 0:
                raise Exception('\nAt least one value of snowfall appears to be ' +
                                'larger than corresponding precipitation value!')
        elif variable == 'pr':
            print('Data variable is daily precipitation!')
        elif variable == 'ps':
            print('Data variable is near surface pressure!')
        elif variable == 'huss':
            print('Data variable is specific humidity!')
        elif variable == 'rsds':
            print('Data variable is shortwave incoming radiation!')
        elif variable == 'rlds':
            print('Data variable is longwave incoming radiation!')
        elif variable == 'wind':
            print('Data variable is near surface wind speed!')
        else:
            raise Exception('Detected variable name not correct!')
    print('Data checked for forbidden values!', flush=True)
    return data
