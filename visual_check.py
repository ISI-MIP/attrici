#!/usr/bin/env python
# coding: utf-8
# Visual check of final cfact

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import settings as s
import xarray as xr


def main():

    TIME0 = datetime.now()

    fact_dir = s.input_dir
    fact = xr.open_dataset(fact_dir / s.source_file)

    cfact_dir = s.output_dir / "cfact" / s.variable
    

    cfact = xr.open_dataset(
        cfact_dir
        / f"rechunked_ERA5_{s.variable}{s.hour}_ERA5_1950_2020_t_{s.tile}_basd_redim_f_cfact_rechunked_valid.nc4"
    )

    # out dir for plots
    out_dir = cfact_dir / "visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestep = 25400  # day in 2019
    date_of_timestep = str(fact.time[timestep].values).split("T")[0]
    print(f"Date of selected timestep {timestep}: ", date_of_timestep)

    ### cfact timeseries for one coord pair
    lat = 10  # 3  # select by index  TODO get idx/coords of random land cell or any cell with values
    lon = 1  # 10
    ts_c = cfact[s.variable].sel(lon=lon, lat=lat, method="nearest")
    ts_f = fact[s.variable].sel(lon=lon, lat=lat, method="nearest")
    ts_c = ts_c[timestep:]  # get only timesteps of the last years
    ts_f = ts_f[timestep:]

    fig, ax = plt.subplots()
    ts_f.plot(ax=ax, color="Red", label="fact")
    ts_c.plot(ax=ax, label="cfact", alpha=0.75)
    plt.legend(loc="upper right")
    plt.savefig(out_dir / f"cfact_ts_{s.variable}{s.hour}_{lat}_{lon}.png")
    plt.close()

    ### reduce processing time by getting subset of time dimension
    TIME2 = datetime.now()
    fact = fact[s.variable][timestep:, :, :]
    cfact = cfact[s.variable][timestep:, :, :]
    print(
        f"creating temporal subsets took {(datetime.now() - TIME2).total_seconds()} seconds"
    )

    ### visual check for one timestep
    value_min = (
        np.min(cfact[0, :, :]) - 0.5
    )  # TODO find most extreme values in cfact or fact
    value_max = np.max(cfact[0, :, :]) + 0.5

    if s.variable == "pr":
        value_min = (
            np.min(cfact[0, :, :]) - 0.0001
        )  # TODO find most extreme values in cfact or fact
        value_max = np.max(cfact[0, :, :]) + 0.0001

    fig, (ax1, ax2) = plt.subplots(figsize=(18, 3), ncols=2)

    # f = ax1.imshow(fact[0,  :, :], cmap="RdYlBu", vmin=value_min, vmax=value_max) #, norm=norm) #, vmin=value_min, vmax=value_max)
    f = ax1.imshow(fact[0, :, :], vmin=value_min, vmax=value_max)
    fig.colorbar(f, ax=ax1)
    ax1.set_title(f"fact {s.variable}{s.hour}, \ntimestep= {date_of_timestep}")

    # norm = mcolors.DivergingNorm(vmin=-0.1, vmax=np.max(cfact[timestep,  :, :]), vcenter=0.0) # aligne around 0
    c = ax2.imshow(cfact[0, :, :], vmin=value_min, vmax=value_max)
    # c = cfact[0,:,:].plot(cmap="RdYlBu", vmin=value_min, vmax=value_max)
    fig.colorbar(c, ax=ax2)
    ax2.set_title(f"cfact {s.variable}{s.hour}, \ntimestep= {date_of_timestep}")
    plt.savefig(out_dir / f"fact_vs_cfact_{s.variable}{s.hour}_time{date_of_timestep}.png")
    plt.close()

    ## visual check for mean over last years until 2021

    # stack lat and lon into a single dimension, apply mean over timesteps and unstack to lat lon again
    TIME1 = datetime.now()
    fact_mean = fact.stack(coords=["lat", "lon"])
    fact_mean = fact_mean.groupby("coords").apply(lambda x: np.mean(x))
    fact_mean = fact_mean.unstack("coords")

    cfact_mean = cfact.stack(coords=["lat", "lon"])
    cfact_mean = cfact_mean.groupby("coords").apply(lambda x: np.mean(x))
    cfact_mean = cfact_mean.unstack("coords")
    print(
        f"took {round( (datetime.now() - TIME1).total_seconds() / 60 )} minutes to get temporal mean from subset "
    )

    TIME1 = datetime.now()
    fig, (ax1, ax2) = plt.subplots(figsize=(18, 3), ncols=2)

    value_min = np.min(cfact_mean[:, :]) - 0.5
    value_max = np.max(cfact_mean[:, :]) + 0.5

    if s.variable == "pr":
        value_min = (
            np.min(cfact[0, :, :]) - 0.0001
        )  # TODO find most extreme values in cfact or fact
        value_max = np.max(cfact[0, :, :]) + 0.0001

    ## mean over time dim
    org = ax1.imshow(fact_mean[:, :], vmin=value_min, vmax=value_max)
    fig.colorbar(org, ax=ax1)
    ax1.set_title(
        f"temporal mean for {date_of_timestep} until 2021, \nfact {s.variable}{s.hour}"
    )

    org = ax2.imshow(cfact_mean[:, :], vmin=value_min, vmax=value_max)
    fig.colorbar(org, ax=ax2)
    ax2.set_title(
        f"temporal mean for {date_of_timestep} until 2021, \ncfact {s.variable}{s.hour}"
    )
    plt.savefig(
        out_dir / f"fact_vs_cfact_mean_{s.variable}{s.hour}_{date_of_timestep}-2021.png"
    )
    plt.close()

    print(
        f"entire processing took {round( (datetime.now() - TIME0).total_seconds() / 60) } minutes"
    )


if __name__ == "__main__":
    main()
