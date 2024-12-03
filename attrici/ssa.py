import xarray as xr

from attrici.preprocessing import calc_gmt_by_ssa


def ssa(input, variable, window_size, subset, output):
    input_dataset = xr.open_dataset(input)
    gmt = input_dataset[variable]
    times = input_dataset["time"]
    ssa_values, ssa_times = calc_gmt_by_ssa(
        gmt, times, window_size=window_size, subset=subset
    )

    output_dataset = xr.Dataset()
    output_dataset["time"] = ssa_times
    output_dataset["ssa"] = xr.DataArray(ssa_values, dims=["time"])
    output_dataset.to_netcdf(output)
