"""Singular Spectrum Analysis (SSA)."""

from pathlib import Path

import xarray as xr

from attrici.preprocessing import calc_gmt_by_ssa
from attrici.util import get_data_provenance_metadata, timeit


@timeit
def ssa(filename, variable, window_size, subset, output):
    """
    Perform Singular Spectrum Analysis (SSA) on a specified variable in a NetCDF
    file and save the results.

    Parameters
    ----------
    filename : str
        Path to the input NetCDF file containing the dataset to be analyzed.

    variable : str
        The name of the variable within the NetCDF file on which SSA will be performed.

    window_size : int
        The size of the window for the SSA calculation.

    subset : int
        The step size for subsetting the input arrays.

    output : str
        Path to the output NetCDF file where the SSA results will be saved.

    Returns
    -------
    None
        The SSA results are saved as a new NetCDF file at the specified output path.
    """

    input_dataset = xr.open_dataset(filename)
    gmt = input_dataset[variable]
    times = input_dataset["time"]
    ssa_values, ssa_times = calc_gmt_by_ssa(
        gmt, times, window_size=window_size, subset=subset
    )

    output_dataset = xr.Dataset(
        data_vars={
            variable: xr.DataArray(
                ssa_values, coords={"time": ssa_times}, dims=("time",)
            )
        },
        attrs=get_data_provenance_metadata(
            input_file=Path(filename).name,
            subset=subset,
            window_size=window_size,
        ),
    )
    output_dataset.to_netcdf(output)
