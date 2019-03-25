#!/home/bschmidt/.programs/anaconda3/envs/detrending/bin/python3

from pyts.decomposition import SSA
import numpy as np
import iris
import iris.coord_categorisation as icc

def gmt_by_ssa(data, window, ):
    ''' This function calculates the global mean temperature (gmt) from a 3d temperature iris.cube object
    by spatial averaging, calculation of monthly means and singular spectrum analysis (ssa)'''
    #  Let iris figure out cell boundaries and calculate
    #  global mean temperatures weighted by cell area
    data.coord('latitude').guess_bounds()
    data.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(data)
    # .collapsed method applieas a function to a grid and outputs
    col = data.collapsed(['longitude', 'latitude'],
                                    iris.analysis.MEAN,
                                    weights=grid_areas)
    # add auxiliary variables for monthly mean calculations
    icc.add_year(col, 'time', name='year')
    icc.add_month_number(col, 'time', name='month')
    # calculate montly means
    monmean = col.aggregated_by(['month', 'year'], iris.analysis.MEAN)

    # extract data from cube and add dimension length 1 for ssa algorithm
    monmean = np.array(monmean.data, ndmin=2)
    # print(monmean)

    # calculate ssa
    ssa = SSA(window_size)
    X_ssa = ssa.fit_transform(monmean)

    return X_ssa


