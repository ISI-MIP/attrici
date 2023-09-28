#!/usr/bin/env python
# coding: utf-8
# Visual check of final cfact


import sys, os
from pathlib import Path
import argparse
from datetime import datetime

import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import settings as s


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tile")
    parser.add_argument("variable_hour")
    args = parser.parse_args()

    tile = args.tile 
    variable_hour = args.variable_hour  #e.g tas0 or hurs
    variable = ''.join(i for i in variable_hour if not i.isdigit())
 

    TIME0 = datetime.now()

    fact_dir =  Path(f"/p/projects/ou/rd3/dmcci/basd_era5-land_to_efas-meteo/basd_for_attrici/")
    #Path(f"/p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/test_field/attrici_03_era5_t{tile}_{variable_hour}_rechunked/")
    fact = xr.open_dataset(fact_dir / f"rechunked_ERA5_{variable_hour}_ERA5_1950_2020_t_{tile}_basd_redim_f.nc")
    #fact = xr.load_dataset("/data/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/test_field/reloaded_merged_ts_t3_tas18_sm.nc4")

    #print(fact)

    #cfact_dir = Path(f"/p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/test_field/attrici_03_era5_t{tile}_{variable_hour}_rechunked/cfact/") / variable
    #fact = xr.open_dataset("/mnt/c/Users/Anna/Documents/UNI/PIK/automated_processing/notebooks/rechunked_ERA5_tas0_ERA5_1950_2020_t_00001_basd_redim_f.nc")
    
    cfact_dir = Path(f"/p/tmp/dominikp/attrici/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/cfact") / variable
    if not os.path.isfile(cfact_dir / f"rechunked_ERA5_{variable_hour}_ERA5_1950_2020_t_{tile}_basd_redim_f_cfact_rechunked_valid.nc4"):
        cfact_dir = Path(f"/p/tmp/annabu/projects/attrici/output/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/cfact") / variable
        print("Searching in", cfact_dir)
    if not os.path.isfile(cfact_dir / f"rechunked_ERA5_{variable_hour}_ERA5_1950_2020_t_{tile}_basd_redim_f_cfact_rechunked_valid.nc4"):
        cfact_dir = Path(f"/p/projects/ou/rd3/dmcci/basd_era5-land_to_efas-meteo/attrici_output_anna/final_cfacts/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/cfact") / variable
        print("Searching in", cfact_dir)
    
    cfact = xr.open_dataset(cfact_dir / f"rechunked_ERA5_{variable_hour}_ERA5_1950_2020_t_{tile}_basd_redim_f_cfact_rechunked_valid.nc4")
    #print(cfact)

    ## out dir for plots
    out_dir = cfact_dir / "visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestep = 25400   # day in 2019
    date_of_timestep = str(fact.time[timestep].values).split("T")[0]
    print(f"Date of selected timestep {timestep}: " , date_of_timestep)

    ### cfact timeseries for one coord pair
    lat = 10 #3  # select by index  TODO get idx/coords of random land cell or any cell with values
    lon = 1#10
    ts_c = cfact[variable].sel(lon=lon, lat=lat, method='nearest')
    ts_f = fact[variable].sel(lon=lon, lat=lat, method='nearest')
    ts_c = ts_c[timestep: ]  # get only timesteps of the last years
    ts_f = ts_f[timestep: ]

    fig, ax = plt.subplots()
    ts_f.plot(ax=ax, color="Red", label="fact")
    ts_c.plot(ax=ax, label="cfact", alpha=0.75)
    plt.legend(loc="upper right")
    plt.savefig(out_dir / f'cfact_ts_{variable_hour}_{lat}_{lon}.png')
    plt.close()


    ### reduce processing time by getting subset of time dimension
    TIME2 = datetime.now()
    fact = fact[variable][timestep:, :, :]
    cfact = cfact[variable][timestep:, :, :]
    print( f"creating temporal subsets took {(datetime.now() - TIME2).total_seconds()} seconds")



    ### visual check for one timestep
    value_min = np.min(cfact[0, :, :]) - 0.5  # TODO find most extreme values in cfact or fact
    value_max = np.max(cfact[0, :, :]) + 0.5

    if variable == "pr":
        value_min = np.min(cfact[0, :, :]) - 0.0001  # TODO find most extreme values in cfact or fact
        value_max = np.max(cfact[0, :, :]) + 0.0001


    fig, (ax1, ax2) = plt.subplots(figsize = (18, 3), ncols = 2)
  
    #f = ax1.imshow(fact[0,  :, :], cmap="RdYlBu", vmin=value_min, vmax=value_max) #, norm=norm) #, vmin=value_min, vmax=value_max)
    f = ax1.imshow(fact[0, :, :], vmin=value_min, vmax=value_max)
    fig.colorbar(f, ax=ax1)
    ax1.set_title(f"fact {variable_hour}, \ntimestep= {date_of_timestep}")

    #norm = mcolors.DivergingNorm(vmin=-0.1, vmax=np.max(cfact[timestep,  :, :]), vcenter=0.0) # aligne around 0
    c = ax2.imshow(cfact[0, :, :], vmin=value_min, vmax=value_max)  
    #c = cfact[0,:,:].plot(cmap="RdYlBu", vmin=value_min, vmax=value_max)
    fig.colorbar(c, ax=ax2)
    ax2.set_title(f"cfact {variable_hour}, \ntimestep= {date_of_timestep}")
    plt.savefig(out_dir / f'fact_vs_cfact_{variable_hour}_time{date_of_timestep}.png')
    plt.close()
    

    ## visual check for mean over last years until 2021

    # stack lat and lon into a single dimension, apply mean over timesteps and unstack to lat lon again
    TIME1 = datetime.now()
    fact_mean = fact.stack(coords=['lat','lon'])
    fact_mean = fact_mean.groupby("coords").apply( lambda x: np.mean(x) )
    fact_mean = fact_mean.unstack('coords')

    cfact_mean = cfact.stack(coords=['lat','lon'])
    cfact_mean = cfact_mean.groupby("coords").apply( lambda x: np.mean(x) )
    cfact_mean = cfact_mean.unstack('coords')
    print( f"took {round( (datetime.now() - TIME1).total_seconds() / 60 )} minutes to get temporal mean from subset ")


    TIME1 = datetime.now()
    fig, (ax1, ax2) = plt.subplots(figsize = (18, 3), ncols = 2)
    
    value_min = np.min(cfact_mean[ :, :]) - 0.5 
    value_max = np.max(cfact_mean[ :, :]) + 0.5

    if variable == "pr":
        value_min = np.min(cfact[0, :, :]) - 0.0001  # TODO find most extreme values in cfact or fact
        value_max = np.max(cfact[0, :, :]) + 0.0001

    ## mean over time dim
    org = ax1.imshow(fact_mean[ :, :], vmin=value_min, vmax=value_max) 
    fig.colorbar(org, ax=ax1)
    ax1.set_title(f"temporal mean for {date_of_timestep} until 2021, \nfact {variable_hour}")

    org = ax2.imshow(cfact_mean[ :, :], vmin=value_min, vmax=value_max) 
    fig.colorbar(org, ax=ax2)
    ax2.set_title(f"temporal mean for {date_of_timestep} until 2021, \ncfact {variable_hour}")
    plt.savefig(out_dir / f'fact_vs_cfact_mean_{variable_hour}_{date_of_timestep}-2021.png')
    plt.close()

    print( f"entire processing took {round( (datetime.now() - TIME0).total_seconds() / 60) } minutes")



if __name__ == "__main__":
    main()

