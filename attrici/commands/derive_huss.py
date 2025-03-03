"""
ATTRICI CLI command: derive-huss

```
usage: attrici derive-huss --hurs HURS --ps PS --tas TAS output_filename

positional arguments:
  output_filename  Merged output file

options:
  --hurs HURS      File containing timeseries for hurs
  --ps PS          File containing timeseries for ps
  --tas TAS        File containing timeseries for tas
```
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from netCDF4 import Dataset
from tqdm import tqdm


def calc_huss_weedon2010(hurs_in, ps_in, tas_in):
    """
    Calculate specific humidity from relative humidity, air pressure, and
    temperature using the equations of

    > Buck (1981) Journal of Applied Meteorology 20,
    > 1527-1532, doi:10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

    as described in

    > Weedon et al. (2010) WATCH Technical Report 22,
    > https://www.eu-watch.org/publications/technical-reports

    Parameters
    ----------
    hurs_in : array_like
        Relative humidity [%]
    ps_in : array_like
        Air pressure [Pa]
    tas_in : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Specific humidity [kg kg-1]
    """

    hurs = hurs_in / 100  # relative humidity [1]
    ps = ps_in / 100  # air pressure [mb]
    tas = tas_in - 273.15  # temperature [degC]

    # constants for calculation of saturation water vapor pressure over water and ice
    # after Weedon2010, i.e., using Buck1981 curves e_w4, e_i3 and f_w4, f_i4
    aw = 6.1121  # [mb]
    ai = 6.1115  # [mb]
    bw = 18.729
    bi = 23.036
    cw = 257.87  # [degC]
    ci = 279.82  # [degC]
    dw = 227.3  # [degC]
    di = 333.7  # [degC]
    xw = 7.2e-4
    xi = 2.2e-4
    yw = 3.20e-6
    yi = 3.83e-6
    zw = 5.9e-10
    zi = 6.4e-10

    # saturation water vapor pressure part of the equation
    saturation_water_vapor_pressure_water = (
        aw * np.exp((bw - tas / dw) * tas / (tas + cw))
    ) * (1.0 + xw + ps * (yw + zw * tas**2))
    saturation_water_vapor_pressure_ice = (
        ai * np.exp((bi - tas / di) * tas / (tas + ci))
    ) * (1.0 + xi + ps * (yi + zi * tas**2))

    saturation_water_vapor_pressure = np.where(
        tas >= 0,
        saturation_water_vapor_pressure_water,
        saturation_water_vapor_pressure_ice,
    )

    # ratio of the specific gas constants of dry air and water vapor after Weedon2010
    Rd_over_Rv = 0.62198

    # sat. water vapor pressure -> saturation specific humidity -> specific humidity
    huss = hurs * Rd_over_Rv / (ps / saturation_water_vapor_pressure + Rd_over_Rv - 1.0)

    var_limits_lower = 1e-7
    var_limits_upper = 1e-1

    return np.minimum(np.maximum(huss, var_limits_lower), var_limits_upper)


def run(args):
    """
    Derive specific humidity from relative humidity, air pressure, and temperature.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """

    logger.info(f"Loading input files {args.hurs}, {args.ps}, {args.tas}")

    hurs_ds = xr.open_dataset(
        args.hurs, chunks="auto", decode_times=False, decode_cf=False
    )

    ps_ds = xr.open_dataset(args.ps, chunks="auto", decode_times=False, decode_cf=False)
    if np.any(ps_ds.lat != hurs_ds.lat) or np.any(ps_ds.lon != hurs_ds.lon):
        raise ValueError("lat/lon mismatch between hurs and ps")

    tas_ds = xr.open_dataset(
        args.tas, chunks="auto", decode_times=False, decode_cf=False
    )
    if np.any(tas_ds.lat != hurs_ds.lat) or np.any(tas_ds.lon != hurs_ds.lon):
        raise ValueError("lat/lon mismatch between hurs and tas")

    var_names = [
        var_name
        for var_name in ["y", "cfact"]
        if var_name in hurs_ds and var_name in ps_ds and var_name in tas_ds
    ]
    for var_name in var_names:
        if hurs_ds[var_name].units != "%":
            raise ValueError(
                f"Unexpected units for {var_name} in {args.hurs} (expected '%')"
            )
        if ps_ds[var_name].units != "Pa":
            raise ValueError(
                f"Unexpected units for {var_name} in {args.ps} (expected 'Pa')"
            )
        if tas_ds[var_name].units != "K":
            raise ValueError(
                f"Unexpected units for {var_name} in {args.tas} (expected 'K')"
            )

    ds = hurs_ds
    with Dataset(args.output_filename, "w") as nc:
        nc.createDimension("lat", len(ds.lat))
        nc.createVariable("lat", "f4", ("lat",))
        nc["lat"][:] = ds.lat
        for attr in ds["lat"].attrs:
            if not attr.startswith("_"):
                nc["lat"].setncattr(attr, ds["lat"].attrs[attr])

        nc.createDimension("lon", len(ds.lon))
        nc.createVariable("lon", "f4", ("lon",))
        nc["lon"][:] = ds.lon
        for attr in ds["lon"].attrs:
            if not attr.startswith("_"):
                nc["lon"].setncattr(attr, ds["lon"].attrs[attr])

        for attr in ds.attrs:
            nc.setncattr(attr, ds.attrs[attr])

        nc.createDimension("time", ds["time"].size)
        if "time" in ds:
            nc.createVariable("time", ds["time"].dtype, ("time",))
            nc["time"][:] = ds["time"].values
            for attr in ds["time"].attrs:
                if not attr.startswith("_"):
                    nc["time"].setncattr(attr, ds["time"].attrs[attr])

        for var_name in var_names:
            var = ds[var_name]
            nc.createVariable(
                var_name,
                var.dtype,
                ("time", "lat", "lon"),
                chunksizes=[c[0] for c in var.chunks],
                fill_value=var.attrs.get("_FillValue", None),
                zlib=True,
            )
            nc[var_name].setncattr("units", "kg kg-1")
            nc[var_name].setncattr("standard_name", "specific_humidity")
            nc[var_name].setncattr("long_name", "Near-Surface Specific Humidity")

        for x in tqdm(range(len(ds.lat)), desc="lat"):
            for y in tqdm(range(len(ds.lon)), desc="lon"):
                for var_name in var_names:
                    nc[var_name][:, x, y] = calc_huss_weedon2010(
                        hurs_ds[var_name][:, x, y],
                        ps_ds[var_name][:, x, y],
                        tas_ds[var_name][:, x, y],
                    )

    logger.info(f"Saved output to {args.output_filename}")


def add_parser(subparsers):
    """
    Add an argparse parser for the 'derive-huss' command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object to which the new parser will be added.
    """
    parser = subparsers.add_parser(
        "derive-huss",
        help="Derive specific humidity from relative humidity, air pressure, and"
        " temperature",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--hurs",
        type=Path,
        required=True,
        help="File containing timeseries for hurs",
    )
    parser.add_argument(
        "--ps",
        type=Path,
        required=True,
        help="File containing timeseries for ps",
    )
    parser.add_argument(
        "--tas",
        type=Path,
        required=True,
        help="File containing timeseries for tas",
    )
    parser.add_argument("output_filename", type=Path, help="Merged output file")
    parser.set_defaults(func=run)
