from pathlib import Path

import xarray as xr

from attrici.preprocessing import calc_gmt_by_ssa
from attrici.util import get_data_provenance_metadata, timeit


@timeit
def ssa(filename, variable, window_size, subset, output):
    input_dataset = xr.open_dataset(filename)
    gmt = input_dataset[variable]
    times = input_dataset["time"]
    ssa_values, ssa_times = calc_gmt_by_ssa(
        gmt, times, window_size=window_size, subset=subset
    )

    output_dataset = xr.Dataset(
        data_vars={variable: ssa_values},
        coords={"time": ssa_times},
        attrs=get_data_provenance_metadata(
            input_file=Path(filename).name,
            subset=subset,
            window_size=window_size,
        ),
    )
    output_dataset.to_netcdf(output)
