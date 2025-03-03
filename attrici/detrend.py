"""
The `detrend` module is the main one in ATTRICI. It provides the functionality for
detrending climatological data using statistical models.

Usage:
The `detrend` function needs to be called with an instance of the configuration
class `Config` to perform detrending on the specified dataset.

Example:
```python
from datetime import date
from pathlib import Path
from attrici.detrend import Config, detrend

config = Config(
    gmt_file=Path("./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc"),
    input_file=Path("./tests/data/20CRv3-ERA5_germany_obs.nc"),
    variable="tas",
    output_dir=Path("./example-output"),
    start_date=date(2000, 1, 1),
    stop_date=date(2020, 12, 31),
    solver="pymc5"
)
detrend(config)
```

For the implementations of the used solvers, see `attrici.estimation`.

For command line usage, see `attrici.commands.detrend`.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import tomlkit
import xarray as xr
from loguru import logger
from tqdm import tqdm

from attrici.util import get_data_provenance_metadata, timeit
from attrici.variables import create_variable


@dataclass
class Config:
    """Object to hold full configuration for a detrending run."""

    gmt_file: Path
    """Path to (SSA-smoothed) Global Mean Temperature file"""
    input_file: Path
    """Path to input file"""
    variable: str
    """Variable to detrend"""
    output_dir: Path
    """Output directory for the results"""
    gmt_variable: str = "tas"
    """Variable name in GMT file"""
    mask_file: Path | None = None
    """Optional path to file with masking information"""
    trace_file: Path | None = None
    """Optional path to file with trace from previous fit"""
    cells: list[tuple[float, float]] | None = None
    """Optional list of lat,lon tuples to process, otherwise all cells are processed"""
    modes: int = 4
    """Number of modes for fourier series of model"""
    bootstrap_sample_count: int = 0
    """Number of bootstrap samples"""
    overwrite: bool = False
    """Overwrite existing files"""
    write_trace: bool = False
    """Save trace to file"""
    fit_only: bool = False
    """Calculate fit only"""
    progressbar: bool = False
    """Show progress bar"""
    report_variables: list[str] | tuple[str] = ("all",)
    """List of variables to include in the output """
    seed: int = 0
    """Seed for deterministic randomisation"""
    solver: str = "pymc5"
    """Solver library for statistical modelling"""
    start_date: date | None = None
    """Optional start date YYYY-MM-DD"""
    stop_date: date | None = None
    """Optional stop date YYYY-MM-DD"""
    task_count: int = 1
    """"Number of tasks for parallel processing"""
    task_id: int = 0
    """Task ID for parallel processing"""
    timeout: int = 60 * 60
    """Maximum time in seconds for sampler for a single grid cell"""
    cache_dir: Path | None = None
    """Use cached results from this directory or write new ones"""
    compile_timeout: int = 600
    """Timeout for PyMC5 model compilation in s"""
    full_extrapolation: bool = False
    """Extrapolate few missing days of GMT instead of stretching it to the full time
    series"""

    def as_dict(self):
        """Return configuration object as dictionary"""
        return vars(self)

    def to_toml(self):
        """Return configuration object as TOML"""
        return tomlkit.dumps(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.as_dict().items()
                if v is not None
            }
        )


def get_task_indices(indices_len, task_id, task_count):
    """
    Get the indices of the grid cells that this task should work on.

    Parameters
    ----------
    indices_len: int
        The number of grid cells
    task_id : int
        The task ID
    task_count : int
        The number of tasks

    Returns
    -------
    np.ndarray
        The indices of the grid cells that this task should work on
    """
    if task_id < 0 or task_id >= task_count:
        raise ValueError("Task ID must be between 0 and task count")

    if indices_len % (task_count) == 0:
        logger.info("Grid cells can be equally distributed to tasks")
        calls_per_arrayjob = np.ones(task_count) * indices_len // (task_count)
    else:
        logger.info(
            "Number of tasks not a divisor of number of grid cells"
            " - some tasks will be empty"
        )
        calls_per_arrayjob = np.ones(task_count) * indices_len // (task_count) + 1
        discarded_jobs = np.where(np.cumsum(calls_per_arrayjob) > indices_len)
        calls_per_arrayjob[discarded_jobs] = 0
        calls_per_arrayjob[discarded_jobs[0][0]] = (
            indices_len - calls_per_arrayjob.sum()
        )

    # Calculate the starting and ending values for this task based
    # on the task id and the number of runs per task.
    cum_calls_per_arrayjob = calls_per_arrayjob.cumsum(dtype=int)
    start_num = 0 if task_id == 0 else cum_calls_per_arrayjob[task_id - 1]
    end_num = cum_calls_per_arrayjob[task_id] - 1
    if end_num < start_num:
        logger.info("No runs assigned for this task")
    else:
        logger.info(
            "This is task {} which will do runs {} to {}",
            task_id,
            start_num,
            end_num,
        )
    return np.arange(start_num, end_num + 1)


def save_compressed_netcdf(ds, filename, chunks=None, encoding=None):
    """
    Save an xarray Dataset to a compressed NetCDF file.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    filename : str or Path
        The filename or path where the NetCDF file will be saved.
    chunks : dict, optional
        Dictionary specifying the chunk sizes for each dimension.
        If `None`, no chunking is applied.
    encoding : dict, optional
        Dictionary specifying the encoding options for each variable.
        If `None`, default encoding is used.
    """

    def _get_full_encoding(varname):
        """
        Get the full encoding for a variable.

        Parameters
        ----------
        varname : str
            The name of the variable

        Returns
        -------
        dict
            The full encoding for the variable
        """
        enc = encoding.get(varname, {})
        enc["zlib"] = True
        chunks = ds[varname].chunks
        if chunks is not None:
            enc["chunksizes"] = list(c[0] for c in chunks)
        return enc

    if encoding is None:
        encoding = {}
    if chunks is not None:
        ds = ds.chunk(chunks)
    ds.to_netcdf(
        filename,
        encoding={varname: _get_full_encoding(varname) for varname in ds.data_vars},
    )


def write_trace(config, trace, lat, lon):
    """
    Write the trace data, i.e. the result from fitting to the data, to a NetCDF file.

    Parameters
    ----------
    config : Config
        Configuration object
    trace : dict
        Trace data dictionary
    lat : float
        Latitude of the cell
    lon : float
        Longitude of the cell
    """
    trace_filename = (
        Path(config.output_dir)
        / "trace"
        / config.variable
        / f"lat_{lat:g}"
        / f"trace_lat{lat:g}_lon{lon:g}.nc"
    )
    trace_filename.parent.mkdir(parents=True, exist_ok=True)
    trace_ds = xr.Dataset(
        coords={"lat": [lat], "lon": [lon]},
    )
    for k, v in trace.items():
        d = xr.DataArray(v)
        for dim in list(d.dims):
            d = d.rename({dim: f"{k}_{dim}"})
        trace_ds[k] = d.expand_dims(dim=("lat", "lon"))
    trace_ds.attrs = get_data_provenance_metadata(attrici_config=config.to_toml())
    save_compressed_netcdf(trace_ds, trace_filename)

    logger.info("Saved trace to {}", trace_filename)


def fit_and_detrend_cell(
    config, data, predictor, subset_times, model_class, trace=None
):
    """
    Fit the model to the data and perform quantile mapping for a single grid cell.

    Parameters
    ----------
    config : Config
        Configuration object
    data : xarray.DataArray
        The time series for the grid cell (including time, lat, lon dimensions)
    predictor : xarray.DataArray
        The predictor time series (e.g. global mean temperature; including time
        dimension)
    subset_times : xarray.DataArray
        The subset of time points to use
    model_class : class
        The class of the model to use (e.g. `attrici.estimation.ModelPymc5`)
    trace : dict, optional
        The trace data dictionary (if fitting was already done)
    """
    lat = data.lat.item()
    lon = data.lon.item()

    # set random seed individually for each cell - always use numpy random functions!
    # (to avoid dependency on parallelization or processing order)
    np.random.seed((config.seed + int(lat * 1e9) + int(lon * 1e6)) % 2**32)

    if not config.fit_only:
        output_filename = (
            Path(config.output_dir)
            / "timeseries"
            / config.variable
            / f"lat_{lat:g}"
            / f"ts_lat{lat:g}_lon{lon:g}.nc"
        )

        if output_filename.exists():
            if config.overwrite:
                logger.warning(
                    "Existing data in {} will be overwritten", output_filename
                )
            else:
                logger.warning(
                    "Existing data in {} found. Calculation skipped", output_filename
                )
                return
        elif config.overwrite:
            logger.warning(
                "No existing data in {}. Running calculation", output_filename
            )

    data[np.isinf(data)] = np.nan

    variable = create_variable(config.variable, data)

    if not variable.y_scaled.notnull().any().item():
        logger.warning("No valid data for lat,lon {:g},{:g} - skipping", lat, lon)
        return

    if trace is not None:
        variable.scaling = {
            k[len("scaling_") :]: v
            for k, v in trace.items()
            if k.startswith("scaling_")
        }

    statistical_model = variable.create_model(
        model_class, predictor.sel(time=subset_times), config.modes
    )

    if trace is None:
        logger.info("Starting fitting")
        trace = statistical_model.fit_cached(
            # the following are used to define the result from the cache perspective
            # (hence, changes in these would result in new computation)
            {
                "data": data,
                "modes": config.modes,
                "predictor": predictor,
                "seed": config.seed,
                "solver": config.solver,
                "subset_times": subset_times,
            },
            cache_dir=config.cache_dir,
            timeout=config.timeout,
            progressbar=config.progressbar,
        )

    if config.write_trace:
        for k, v in variable.scaling.items():
            trace[f"scaling_{k}"] = v
        write_trace(config, trace, lat, lon)

    if config.fit_only:
        return

    logger.info("Starting quantile mapping")

    distribution_ref = statistical_model.estimate_distribution(
        trace, predictor=predictor, progressbar=config.progressbar
    )

    predictor_cfact = predictor.copy()
    predictor_cfact[:] = 0
    distribution_cfact = statistical_model.estimate_distribution(
        trace, predictor=predictor_cfact, progressbar=config.progressbar
    )

    def log_invalid_count(indices, name):
        """
        Log the number of invalid values from quantile mapping.

        Parameters
        ----------
        indices : xarray.DataArray or numpy.ndarray
            Boolean array indicating the positions of invalid values.
        name : str
            Name of the type of invalid values (e.g., "NaN", "Inf").
        """
        count = indices.sum().item()
        if count > 0:
            logger.info(
                "There are {} {} values from quantile mapping"
                " (replaced with original value)",
                count,
                name,
            )

    cfact_scaled = variable.y_scaled.astype(np.float64)
    cfact_scaled[:] = variable.quantile_mapping(distribution_ref, distribution_cfact)
    cfact = variable.rescale(cfact_scaled)
    cfact.attrs = data.attrs

    replaced = np.zeros_like(cfact)
    indices = cfact.isnull()
    cfact[indices] = data[indices]
    replaced[indices] = np.nan
    log_invalid_count(indices, "NaN")

    indices = cfact.isin([np.inf])
    cfact[indices] = data[indices]
    replaced[indices] = np.inf
    log_invalid_count(indices, "Inf")

    indices = cfact.isin([-np.inf])
    cfact[indices] = data[indices]
    replaced[indices] = -np.inf
    log_invalid_count(indices, "-Inf")

    # check if resulting data is also valid
    variable.validate(cfact)

    logp = statistical_model.estimate_logp(trace)

    logger.info("Writing output")

    def array_on_cell(d, **kwargs):
        """
        Create a DataArray with the given data on the cell dimensions.

        Parameters
        ----------
        d : xarray.DataArray or numpy.ndarray or list[float]
            The data to put on the cell
        kwargs : dict, optional
            Additional arguments passed on to `xr.DataArray`

        Returns
        -------
        xarray.DataArray
            The data array with the data on the cell
        """
        if isinstance(d, xr.DataArray):
            d = d.expand_dims(dim=("lat", "lon"), axis=(1, 2))
        else:
            d = d.reshape((len(data.time), 1, 1))
        return xr.DataArray(
            d,
            coords={"time": data.time, "lat": [data.lat], "lon": [data.lon]},
            dims=("time", "lat", "lon"),
            **kwargs,
        )

    ds = xr.Dataset(
        {
            "y": array_on_cell(data, attrs=data.attrs),
            "gmt_scaled": predictor,
            "y_scaled": array_on_cell(variable.y_scaled),
            "cfact_scaled": array_on_cell(cfact_scaled),
            "cfact": array_on_cell(cfact, attrs=data.attrs),
            "logp": xr.DataArray(
                [[logp]],
                coords={"lat": [data.lat], "lon": [data.lon]},
                dims=("lat", "lon"),
            ),
            "replaced": array_on_cell(replaced),
        },
    )
    for k, v in distribution_ref.__dict__.items():
        ds[k] = array_on_cell(v)
    for k, v in distribution_cfact.__dict__.items():
        ds[f"{k}_cfact"] = array_on_cell(v)
    if "all" not in config.report_variables:
        ds = ds[config.report_variables]

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs = get_data_provenance_metadata(attrici_config=config.to_toml())
    save_compressed_netcdf(
        ds,
        output_filename,
        chunks={"time": "auto", "lat": 1, "lon": 1},
        encoding={"replaced": {"_FillValue": 0}},
    )
    logger.info("Saved timeseries to {}", output_filename)

    if config.bootstrap_sample_count > 0:
        if (
            np.any(subset_times != data.time)
            or np.any(subset_times != predictor.time)
            or len(subset_times) != len(data.time)
            or len(subset_times) != len(predictor.time)
        ):
            raise ValueError("Bootstrap can only be done on full time series")

        logger.info("Starting block bootstrap")

        # create blocks of indices of data points for each year, i.e. a list of lists of
        # indices, each list containing the indices of the data points for one year
        blocks = [
            block_indices
            for year, block_indices in sorted(
                data.time.groupby("time.year").groups.items(),
                key=lambda group: group[0],
            )
        ]

        def get_random_block_indices(target_length):
            """
            Get data point indices from a random block while adjusting for target
            block length.

            Parameters
            ----------
            target_length : int
                The target length of the block

            Returns
            -------
            np.ndarray
                The indices of the data points from the block
            """
            block = np.random.randint(0, len(blocks))
            block_indices = blocks[block]
            if len(block_indices) < target_length:
                # due to different lengths of years, we might need to take some indices
                # from the neighbouring blocks
                missing = target_length - len(block_indices)
                if block >= len(blocks) - 1:
                    # last block -> take some indices from the block before
                    block_indices = np.concatenate(
                        (blocks[block - 1][-missing:], block_indices)
                    )
                else:
                    # otherwise take some indices from the next block
                    block_indices = np.concatenate(
                        (block_indices, blocks[block + 1][:missing])
                    )
            return block_indices[:target_length]

        original_y_scaled = variable.y_scaled.copy()
        original_quantiles = distribution_ref.cdf(original_y_scaled)
        bootstrapped_expected_values = []
        bootstrap_replaced = np.zeros_like(original_y_scaled)
        for _ in tqdm(range(config.bootstrap_sample_count)):
            # we here generate bootstrapped data by reshuffling the quantiles of the
            # original data in blocks of one year each
            new_quantiles = np.concatenate(
                [
                    original_quantiles[get_random_block_indices(len(block_indices))]
                    for block_indices in blocks
                ]
            )
            # then we derive new data time series from the bootstrapped quantiles
            # by inverting the CDF as for each original time step, i.e. deriving
            # sampling from the original distribution according to the shuffled
            # quantiles
            variable.y_scaled.values = distribution_ref.invcdf(new_quantiles)

            # some derived values might be out of bounds or NaN/Inf, so we replace them
            # with the original values (and count how often this happens)
            invalid_indices = np.isnan(variable.y_scaled) | np.isinf(variable.y_scaled)
            bootstrap_replaced[invalid_indices] += 1
            variable.y_scaled[invalid_indices] = original_y_scaled[invalid_indices]

            # now we fit the model to the bootstrapped data and derive the expected
            # values from the fitted distributions
            statistical_model = variable.create_model(
                model_class, predictor, config.modes
            )
            new_trace = statistical_model.fit()
            new_distribution = statistical_model.estimate_distribution(
                new_trace, predictor=predictor
            )
            bootstrapped_expected_values.append(
                variable.rescale(new_distribution.expectation())
            )

        # quantiles to calculate for the bootstrapped expected values
        quantiles = [0, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 1]
        bootstrap = xr.Dataset(
            {
                # expected values from the distribution fit to the original data
                "expected_values": array_on_cell(
                    variable.rescale(distribution_ref.expectation()), attrs=data.attrs
                ),
                # standard deviation of the expected values from the distribution fit to
                # the bootstrapped data
                "bootstrap_std": array_on_cell(
                    np.std(bootstrapped_expected_values, axis=0), attrs=data.attrs
                ),
                # mean of the expected values from the distribution fit to the
                # bootstrapped data
                "bootstrap_mean": array_on_cell(
                    np.mean(bootstrapped_expected_values, axis=0), attrs=data.attrs
                ),
                # quantiles of the expected values from the distribution fit to the
                # bootstrapped data
                "bootstrap_quantiles": xr.DataArray(
                    np.asarray(
                        [
                            np.percentile(bootstrapped_expected_values, q * 100, axis=0)
                            for q in quantiles
                        ]
                    ).reshape((len(quantiles), len(data.time), 1, 1)),
                    coords={
                        "quantile": quantiles,
                        "time": data.time,
                        "lat": [data.lat],
                        "lon": [data.lon],
                    },
                    dims=("quantile", "time", "lat", "lon"),
                    attrs=data.attrs,
                ),
                # number of times a value was replaced in the bootstrapping process
                # (i.e. the number of times the value was NaN or Inf)
                "bootstrap_replaced": array_on_cell(bootstrap_replaced),
            }
        )

        bootstrap.attrs = get_data_provenance_metadata(attrici_config=config.to_toml())
        bootstrap_filename = (
            Path(config.output_dir)
            / "bootstrap"
            / config.variable
            / f"lat_{lat:g}"
            / f"bootstrap_lat{lat:g}_lon{lon:g}.nc"
        )
        bootstrap_filename.parent.mkdir(parents=True, exist_ok=True)
        save_compressed_netcdf(
            bootstrap,
            bootstrap_filename,
            chunks={"quantile": "auto", "time": "auto", "lat": 1, "lon": 1},
        )
        logger.info("Saved bootstrap to {}", bootstrap_filename)


@timeit
def detrend(config: Config):
    """
    Run the detrending process, i.e. load the data, fit the model, perform quantile
    mapping, and save the results.

    Parameters
    ----------
    config : Config
        Configuration object
    """
    logger.info("Detrending with config:\n{}", config.to_toml())

    gmt = xr.open_dataset(config.gmt_file)[config.gmt_variable]

    # check dimensions and for valid values
    if gmt.dims != ("time",):
        raise ValueError("GMT data must have dimension 'time'")
    if np.any(np.isnan(gmt)) or np.any(np.isinf(gmt)):
        raise ValueError("GMT data must not contain NaN or infinite values")

    obs_data = xr.open_dataset(config.input_file)[config.variable]

    # check if dimensions are correct
    if "lat" not in obs_data.dims or "lon" not in obs_data.dims:
        if "latitude" in obs_data.dims and "longitude" in obs_data.dims:
            obs_data = obs_data.rename({"latitude": "lat", "longitude": "lon"})
        else:
            raise ValueError(
                "Input data must have lat and lon dimensions(or latitude and longitude)"
            )
    if set(obs_data.dims) != {"lat", "lon", "time"}:
        raise ValueError("Input data must have dimensions lat, lon, time")

    obs_data = obs_data.stack(latlon=("lat", "lon"))

    if config.cells:
        try:
            obs_data = obs_data.sel(latlon=[(lat, lon) for lat, lon in config.cells])
        except KeyError as e:
            logger.error("Not all cells could be found in input data")
            raise e

    if config.mask_file:
        mask_file = xr.open_dataset(config.mask_file)
        mask = mask_file["mask"].stack(latlon=("lat", "lon"))
        mask = mask.where(mask == 1).dropna("latlon")["latlon"].values
        obs_data = obs_data.sel(latlon=mask)

    if config.full_extrapolation:
        # `gmt.time` is a subset of `obs_data.time` (e.g. every 10th day)
        # hence, interpolate these values to the full time series
        # the last few days are extrapolated
        gmt_on_obs_times = gmt.interp(
            time=obs_data.time, kwargs={"fill_value": "extrapolate"}
        )
        gmt_scaled = (gmt_on_obs_times - gmt_on_obs_times.min()) / (
            gmt_on_obs_times.max() - gmt_on_obs_times.min()
        )
    else:
        t_scaled = (obs_data.time - obs_data.time.min()) / (
            obs_data.time.max() - obs_data.time.min()
        )
        gmt_on_obs_times = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
        gmt_scaled_values = (gmt_on_obs_times - gmt_on_obs_times.min()) / (
            gmt_on_obs_times.max() - gmt_on_obs_times.min()
        )
        gmt_scaled = xr.DataArray(
            gmt_scaled_values, coords={"time": obs_data.time}, dims=("time",)
        )

    startdate = config.start_date
    if startdate is None:
        startdate = obs_data.time[0]
    else:
        startdate = np.datetime64(startdate)
    stopdate = config.stop_date
    if stopdate is None:
        stopdate = obs_data.time[-1]
    else:
        stopdate = np.datetime64(stopdate)

    subset_times = obs_data.time[
        (obs_data.time >= startdate) & (obs_data.time <= stopdate)
    ]

    if config.solver == "pymc5":
        from attrici.estimation.model_pymc5 import ModelPymc5, initialize

        initialize(config.compile_timeout, use_tmp_compiledir=config.task_count > 1)
        model_class = ModelPymc5
    elif config.solver == "scipy":
        from attrici.estimation.model_scipy import ModelScipy

        model_class = ModelScipy
    elif config.solver == "pymc3":
        from attrici.estimation.model_pymc3 import ModelPymc3, initialize

        initialize(config.compile_timeout, use_tmp_compiledir=config.task_count > 1)
        model_class = ModelPymc3
    else:
        raise ValueError(f"Unknown solver {config.solver}")

    if config.trace_file:
        trace = (
            xr.open_dataset(config.trace_file)
            .stack(latlon=("lat", "lon"))
            .sel(latlon=obs_data.latlon)
        )
        if config.write_trace:
            logger.warning("Ignoring --write-trace as trace file is provided")
            config.write_trace = False
    else:
        trace = None

    if config.fit_only and not config.write_trace:
        logger.warning(
            "Fitting only, but no trace file will be written (forgot --write-trace?)"
        )

    logger.info("A total of {} grid cells to estimate", len(obs_data.latlon))

    indices = get_task_indices(len(obs_data.latlon), config.task_id, config.task_count)
    for i, latlon in enumerate(indices):
        data = obs_data.isel(latlon=latlon)

        logger.info(
            "This is task {} working on lat,lon {:g},{:g} (cell {}/{})",
            config.task_id,
            data.lat.item(),
            data.lon.item(),
            i + 1,
            len(indices),
        )

        if trace is not None:
            cell_trace = {
                k: v.values for k, v in trace.sel(latlon=data.latlon).data_vars.items()
            }
            logger.info("Using trace from previous fit")
        else:
            cell_trace = None

        fit_and_detrend_cell(
            config,
            data,
            predictor=gmt_scaled,
            subset_times=subset_times,
            model_class=model_class,
            trace=cell_trace,
        )
