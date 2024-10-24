import argparse
import importlib.metadata
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import tomlkit
from loguru import logger

from attrici.commands import add_config_argument


def get_config_dict(args):
    """Turn `args` into dictionary, but skip
    `func` and `print_config` fields,
    `None` values (cannot be handled by TOML),
    and convert `Path` objects to strings.
    """
    config = {}
    for k, v in vars(args).items():
        if v is not None and k not in {"func", "print_config"}:
            config[k] = str(v) if isinstance(v, Path) else v
    return config


def run(args):  # noqa: PLR0915 TODO
    if args.print_config:
        print(tomlkit.dumps(get_config_dict(args)))
        return

    import numpy as np
    import xarray as xr
    from func_timeout import FunctionTimedOut, func_timeout

    import attrici.datahandler as dh
    import attrici.estimator as est
    from attrici import __version__ as attrici_version

    metadata = {
        "attrici_config": tomlkit.dumps(get_config_dict(args)),
        "attrici_version": attrici_version,
        "attrici_packages": "\n".join(
            sorted(
                set(
                    [
                        f"{dist.name}=={dist.version}"
                        for dist in importlib.metadata.distributions()
                    ]
                ),
                key=str.casefold,
            )
        ),
        "attrici_python_version": sys.version,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger("pymc3").propagate = False  # TODO needed to silence verbose pymc3

    if args.task_id < 0 or args.task_id >= args.task_count:
        raise ValueError("Task ID must be between 0 and task count")

    gmt = xr.open_dataset(args.gmt_file)["tas"]

    obs_data = xr.open_dataset(args.input_file)
    obs_times = obs_data["time"]
    obs_lats = obs_data["lat"]
    obs_lons = obs_data["lon"]

    if args.mask_file:
        mask_file = xr.open_dataset(args.mask_file)
        if not np.allclose(mask_file["lat"], obs_lats) or not np.allclose(
            mask_file["lon"], obs_lons
        ):
            raise ValueError("Mask grid does not match data grid")
        mask = mask_file["mask"]
    else:
        mask = np.ones((len(obs_lats), len(obs_lons)))
    lon_grid, lat_grid = np.meshgrid(np.arange(len(obs_lons)), np.arange(len(obs_lats)))
    indices = np.asarray(
        [lat_grid[mask == 1].flatten(), lon_grid[mask == 1].flatten()]
    ).T

    logger.info("A total of {} grid cells to estimate", len(indices))

    if len(indices) % (args.task_count) == 0:
        logger.info("Grid cells can be equally distributed to tasks")
        calls_per_arrayjob = (
            np.ones(args.task_count) * len(indices) // (args.task_count)
        )
    else:
        logger.info(
            "Number of tasks not a divisor of number of grid cells"
            " - some tasks will be empty"
        )
        calls_per_arrayjob = (
            np.ones(args.task_count) * len(indices) // (args.task_count) + 1
        )
        discarded_jobs = np.where(np.cumsum(calls_per_arrayjob) > len(indices))
        calls_per_arrayjob[discarded_jobs] = 0
        calls_per_arrayjob[discarded_jobs[0][0]] = (
            len(indices) - calls_per_arrayjob.sum()
        )

    assert calls_per_arrayjob.sum() == len(indices)

    # Calculate the starting and ending values for this task based
    # on the task id and the number of runs per task.
    cum_calls_per_arrayjob = calls_per_arrayjob.cumsum(dtype=int)
    start_num = 0 if args.task_id == 0 else cum_calls_per_arrayjob[args.task_id - 1]
    end_num = cum_calls_per_arrayjob[args.task_id] - 1
    run_numbers = np.arange(start_num, end_num + 1, 1, dtype=int)
    if len(run_numbers) == 0:
        logger.info("No runs assigned for this task")
    else:
        logger.info(
            "This is task {} which will do runs {} to {}",
            args.task_id,
            start_num,
            end_num,
        )

    estimator = est.Estimator(
        output_dir=args.output_dir,
        seed=args.seed,
        progressbar=args.progressbar,
        variable=args.variable,
        modes=args.modes,
        report_variables=args.report_variables,
        start_date=args.start_date,
        stop_date=args.stop_date,
    )

    TIME0 = datetime.now()

    for n in run_numbers[:]:
        lat_index, lon_index = indices[n]
        lat = float(obs_lats[lat_index])
        lon = float(obs_lons[lon_index])

        logger.info(
            "This is task {}, run number {}, lat,lon {},{}",
            args.task_id,
            n,
            lat,
            lon,
        )
        outdir_for_cell = dh.make_cell_output_dir(
            args.output_dir, "timeseries", lat, lon, args.variable
        )
        fname_cell = dh.get_cell_filename(outdir_for_cell, lat, lon)

        if fname_cell.exists():
            if args.overwrite:
                logger.warning(
                    "Existing data in {} will be overwritten",
                    fname_cell,
                )
            else:
                logger.warning(
                    "Existing data in {} found. Calculation skipped",
                    fname_cell,
                )
                continue
        elif args.overwrite:
            logger.warning(
                "No existing data in {}. Running calculation",
                fname_cell,
            )

        data = obs_data[args.variable][:, lat_index, lon_index]
        df, datamin, scale = dh.create_dataframe(obs_times, data, gmt, args.variable)

        try:
            logger.info(
                "Took {:.1f} seconds till estimator.estimate_parameters is started",
                (datetime.now() - TIME0).total_seconds(),
            )
            trace, dff = func_timeout(
                args.timeout,
                estimator.estimate_parameters,
                args=(df, lat, lon, TIME0),
                kwargs=dict(use_cache=args.use_cache),
            )
        except FunctionTimedOut:
            logger.error("Sampling at {} {} timed out", lat, lon)
            continue

        df_with_cfact = estimator.estimate_timeseries(dff, trace, datamin, scale)
        dh.save_to_disk(
            df_with_cfact,
            fname_cell,
            lat,
            lon,
            **metadata,
        )

    logger.info(
        "Estimation completed for all cells. It took {:.1f} minutes",
        (datetime.now() - TIME0).total_seconds() / 60,
    )


def iso_date(value):
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def fourier_modes(value):
    try:
        modes = [int(m) for m in value.split(",")]
        FOURIER_SERIES_LENGTH = 4
        if len(modes) == 1:
            return modes * FOURIER_SERIES_LENGTH
        elif len(modes) == FOURIER_SERIES_LENGTH:
            return modes
        else:
            raise ValueError("Fourier modes must be one or four integers")
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "detrend",
        help="Detrend a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current config as TOML and exit",
    )

    # input
    group = parser.add_argument_group(title="Input")
    group.add_argument(
        "--gmt-file",
        type=Path,
        help="(SSA-smoothed) Global Mean Temperature file",
        required=True,
    )
    group.add_argument("--input-file", type=Path, help="Input file", required=True)
    group.add_argument("--mask-file", type=Path, help="Mask file")
    group.add_argument(
        "--variable", type=str, help="Variable to detrend", required=True
    )

    # output
    group = parser.add_argument_group(title="Output")
    group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for the results",
    )
    group.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    # run parameters
    group = parser.add_argument_group(title="Run parameters")
    group.add_argument(
        "--modes",
        type=fourier_modes,
        default=[4, 4, 4, 4],
        help="Number of modes for fourier series of model (either one integer, used for"
        " all four series, or four comma-separated integers)",
    )
    group.add_argument("--progressbar", action="store_true", help="Show progress bar")
    group.add_argument(
        "--report-variables",
        nargs="+",
        default=["all"],
        help="Variables to report, e.g. `--report-variables ds y cfact logp`",
    )
    group.add_argument(
        "--seed", type=int, default=0, help="Seed for deterministic randomisation"
    )
    group.add_argument(
        "--start-date",
        type=iso_date,
        help="Start date for the detrending period",
    )
    group.add_argument(
        "--stop-date",
        type=iso_date,
        help="Stop date for the detrending period",
    )
    group.add_argument(
        "--task-id", type=int, default=0, help="Task ID for parallel processing"
    )
    group.add_argument(
        "--task-count",
        type=int,
        default=1,
        help="Number of tasks for parallel processing",
    )
    group.add_argument(
        "--timeout",
        type=int,
        default=60 * 60,
        help="Maximum time in seconds for sampler for a single grid cell",
    )
    group.add_argument(
        "--use-cache", action="store_true", help="Use cached results and write new ones"
    )

    # model parameters
    group = parser.add_argument_group(title="Model parameters")
    group.add_argument(
        "--tune",
        type=int,
        default=500,
        help="Number of draws to tune model",
    )
    group.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Number of sampling draws per chain",
    )
    group.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of chains to calculate (min 2 to check for convergence)",
    )

    parser.set_defaults(func=run)
