"""Detrend."""

import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import tomlkit
import xarray as xr
from func_timeout import FunctionTimedOut, func_timeout
from loguru import logger

from attrici import variables
from attrici.util import get_data_provenance_metadata, timeit

MODEL_FOR_VAR = {
    "tas": variables.Tas,
    "tasrange": variables.Tasrange,
    "tasskew": variables.Tasskew,
    "pr": variables.Pr,
    "hurs": variables.Hurs,
    "wind": variables.Wind,
    "sfcWind": variables.Wind,
    "ps": variables.Ps,
    "rsds": variables.Rsds,
    "rlds": variables.Rlds,
}


@dataclass
class Config:
    """Configuration object for detrending run."""

    gmt_file: Path
    """Path to (SSA-smoothed) Global Mean Temperature file"""
    input_file: Path
    """Path to input file"""
    mask_file: Path
    """Optional path to file with masking information"""
    variable: str
    """Variable to detrend"""
    output_dir: Path
    """Output directory for the results"""
    modes: int = 4
    """Number of modes for fourier series of model"""
    overwrite: bool = False
    """Overwrite existing files"""
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
    use_cache: bool = False
    """Use cached results and write new ones"""

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


def get_task_indices(indices, task_id, task_count):
    """Get the indices of the grid cells that this task should work on.

    Parameters
    ----------
    indices : np.ndarray
        The indices of the grid cells
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

    if len(indices) % (task_count) == 0:
        logger.info("Grid cells can be equally distributed to tasks")
        calls_per_arrayjob = np.ones(task_count) * len(indices) // (task_count)
    else:
        logger.info(
            "Number of tasks not a divisor of number of grid cells"
            " - some tasks will be empty"
        )
        calls_per_arrayjob = np.ones(task_count) * len(indices) // (task_count) + 1
        discarded_jobs = np.where(np.cumsum(calls_per_arrayjob) > len(indices))
        calls_per_arrayjob[discarded_jobs] = 0
        calls_per_arrayjob[discarded_jobs[0][0]] = (
            len(indices) - calls_per_arrayjob.sum()
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
    return indices[start_num : end_num + 1]


def try_to_load_trace(filename):
    """Load trace file.

    Parameters
    ----------
    filename : str | os.PathLike

    Returns
    -------
    dictionary
    """
    if not filename.exists():
        return None
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)  # TODO use a different format than pickle
    except Exception:
        logger.exception("Problem with saved trace. Redo parameter estimation.")
    return None


def save_trace(trace, filename):
    """Save trace information.

    Parameters
    ----------
    trace : dictionary
    filename : str | os.PathLike
    """
    free_params = {
        key: value
        for key, value in trace.items()
        if key.startswith("weights") or key == "logp"
    }
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(
            free_params, f, protocol=pickle.HIGHEST_PROTOCOL
        )  # TODO use a different format than pickle


def log_invalid_count(indices, name):
    count = indices.sum().item()
    if count > 0:
        logger.info(
            "There are {} {} values from quantile mapping"
            " (replaced with original value)",
            count,
            name,
        )


def detrend_cell(config, data, gmt_scaled, subset_times, lat, lon, model_class):
    output_filename = (
        Path(config.output_dir)
        / "timeseries"
        / config.variable
        / f"lat_{lat}"
        / f"ts_lat{lat}_lon{lon}.nc"
    )

    if output_filename.exists():
        if config.overwrite:
            logger.warning("Existing data in {} will be overwritten", output_filename)
        else:
            logger.warning(
                "Existing data in {} found. Calculation skipped", output_filename
            )
            return
    elif config.overwrite:
        logger.warning("No existing data in {}. Running calculation", output_filename)

    data[np.isinf(data)] = np.nan

    variable = MODEL_FOR_VAR[config.variable](data)

    statistical_model = variable.create_model(
        model_class, gmt_scaled.sel(time=subset_times), config.modes
    )

    trace = None
    if config.use_cache:
        trace_filename = (
            config.output_dir
            / "traces"
            / config.variable
            / f"lat_{lat}"
            / f"traces_lat{lat}_lon{lon}.pkl"
        )
        trace = try_to_load_trace(trace_filename)

    if trace is None:
        try:
            trace = func_timeout(
                config.timeout,
                lambda: statistical_model.fit(progressbar=config.progressbar),
            )
        except FunctionTimedOut:
            logger.error("Sampling at {} {} timed out", lat, lon)
            return
        if config.use_cache:
            save_trace(trace, trace_filename)

    statistical_model.trace = trace  # TODO to be replaced by caching library

    distribution_ref = statistical_model.estimate_distribution(
        predictor=gmt_scaled, progressbar=config.progressbar
    )

    gmt_scaled_cfact = gmt_scaled.copy()
    gmt_scaled_cfact[:] = 0
    distribution_cfact = statistical_model.estimate_distribution(
        predictor=gmt_scaled_cfact, progressbar=config.progressbar
    )

    logger.info("Starting quantile mapping")
    cfact_scaled = variable.y_scaled.astype(np.float64)
    cfact_scaled[:] = variable.quantile_mapping(distribution_ref, distribution_cfact)
    cfact = variable.rescale(cfact_scaled)
    cfact.attrs = data.attrs

    indices = cfact.isnull()
    cfact[indices] = data[indices]
    log_invalid_count(indices, "NaN")

    indices = cfact.isin([np.inf])
    cfact[indices] = data[indices]
    log_invalid_count(indices, "Inf")

    indices = cfact.isin([-np.inf])
    cfact[indices] = data[indices]
    log_invalid_count(indices, "-Inf")

    logger.info("Writing output")
    ds = xr.Dataset(
        {
            "y": data,
            "gmt_scaled": gmt_scaled,
            "y_scaled": variable.y_scaled,
            "cfact_scaled": cfact_scaled,
            "logp": statistical_model.estimate_logp(progressbar=config.progressbar),
            "cfact": cfact,
        },
        coords=data.coords,
    )
    for k, v in distribution_ref.__dict__.items():
        ds[k] = variable.y_scaled.copy()
        ds[k][:] = v
    for k, v in distribution_cfact.__dict__.items():
        ds[f"{k}_ref"] = variable.y_scaled.copy()
        ds[f"{k}_ref"][:] = v
    if "all" not in config.report_variables:
        ds = ds[config.report_variables]

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs = get_data_provenance_metadata(attrici_config=config.to_toml())
    ds.to_netcdf(output_filename)
    logger.info("Saved timeseries to {}", output_filename)


@timeit
def detrend(config: Config):
    """Run detrending.

    Parameters
    ----------
    config : Config
        Configuration object
    """
    logger.info("Detrending with config:\n{}", config.to_toml())

    gmt = xr.open_dataset(config.gmt_file)["tas"]

    obs_ds = xr.open_dataset(config.input_file)
    obs_lats = obs_ds["lat"]
    obs_lons = obs_ds["lon"]
    obs_data = obs_ds[config.variable]

    if config.mask_file:
        mask_file = xr.open_dataset(config.mask_file)
        if not np.allclose(mask_file["lat"], obs_lats) or not np.allclose(
            mask_file["lon"], obs_lons
        ):
            raise ValueError("Mask grid does not match data grid")
        mask = mask_file["mask"]
    else:
        mask = np.ones((len(obs_lats), len(obs_lons)))
    lon_grid, lat_grid = np.meshgrid(np.arange(len(obs_lons)), np.arange(len(obs_lats)))
    lat_lon_indices = [
        (float(obs_lats[lat_index]), lat_index, float(obs_lons[lon_index]), lon_index)
        for lat_index, lon_index in zip(
            lat_grid[mask == 1].flatten(), lon_grid[mask == 1].flatten()
        )
    ]

    logger.info("A total of {} grid cells to estimate", len(lat_lon_indices))

    indices = get_task_indices(lat_lon_indices, config.task_id, config.task_count)

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
        from attrici.estimation.model_pymc5 import ModelPymc5

        model_class = ModelPymc5
    elif config.solver == "scipy":
        from attrici.estimation.model_scipy import ModelScipy

        model_class = ModelScipy
    else:
        raise ValueError(f"Unknown solver {config.solver}")

    for lat, lat_index, lon, lon_index in indices:
        logger.info(
            "This is task {} working on lat,lon {},{}", config.task_id, lat, lon
        )

        detrend_cell(
            config,
            obs_data[:, lat_index, lon_index],
            gmt_scaled,
            subset_times,
            lat,
            lon,
            model_class,
        )
