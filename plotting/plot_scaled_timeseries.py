import settings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plotting.helper_functions import get_parser
import icounter.const as c


def main(runid, lat, lon, window, start, end, ax=None):
    """
    For mode = dry_days:        Plots timeseries of the observed and counterfactual dry_day freq.
                                From the hd5 timeseries files, as well as the modeled Bernoulli parameter.
    for mode = wet_days:        Plots timeseries of obs. and counterf. precip. amounts on wet days.
    for mode = wet_days_scaled: Plots times. of obs an counterf. of scaled (to standard-deviation 1) precip amounts
                                on wet days as well as the mu (expactition) parameter of the modeled Gamma dist.
    Parameters
    ----------
    runid
    lat
    lon
    mode
    window
    ax

    Returns
    -------

    """
    data_dir = settings.output_dir.parents[0]
    variable = settings.variable
    if lat is None:
        lats_dir = [lat_dir for lat_dir in (data_dir/runid/'timeseries'/variable).iterdir()]
    else:
        lats_dir = [data_dir/runid/'timeseries'/variable/f'lat_{lat}']
    for lat_dir in lats_dir:
        if lon is None:
            files = [file for file in lat_dir.iterdir()]
        else:
            files = [data_dir/runid/'timeseries'/variable/f'lat_{lat}'/f'ts_GSWP3_lat{lat}_lon{lon}.h5']

        for file in files:
            figure_dir = data_dir / "figures" / runid / lat_dir.name
            figure_dir.mkdir(parents=True, exist_ok=True)
            print(lat_dir.name, file.stem)

            df = pd.read_hdf(file)
            df = df.set_index('ds')

            if variable == 'pr':
                fig, axs = plt.subplots(4, 1, figsize=(20, 16))
                axs = axs.reshape(-1)
                mode = 'dry_days'
                for key in ['y', 'cfact']:
                    df_mean = (df[key] <= c.threshold[variable][0]).resample(window).mean()
                    df_mean.plot(alpha=.5, title=mode, ax=axs[0])
                for key in ['pbern', 'pbern_ref']:
                    df[key].resample(window).mean().plot(alpha=.5, ax=axs[0])
                mode = 'wet_days'
                for key in ['y', 'cfact']:
                    df_mean = df[key].loc[df[key] > c.threshold[variable][0]].resample(window).mean()
                    df_mean.plot(alpha=.5, title=mode, ax=axs[1])
                mode = 'wet_days_scaled'
                df_means = df[['y_scaled',  'mu']].loc[df['y'] > c.threshold[variable][0]].resample(window).mean()
                df_ref_means = df[['cfact_scaled', 'mu_ref']].loc[df['cfact'] > c.threshold[variable][0]].resample(window).mean()
                pd.concat([df_means,df_ref_means], axis=1).plot(alpha=.5, title=mode, ax=axs[2])
                mode = 'all'
                df.loc[df['y_scaled'].isna(), "y_scaled"] = 0
                df_means = df['y_scaled'].resample(window).mean()
                df_ref_means = df['cfact_scaled'].resample(window).mean()
                pd.concat([df_means, df_ref_means], axis=1).plot(alpha=.5, title=mode, ax=axs[3])
            elif variable == 'tas':
                df[['y_scaled', 'mu']][start:end].resample(window).mean().plot(alpha=.5)

            plt.legend()
            plt.savefig(figure_dir / f'{file.stem}.jpg', dpi=80)


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            for mode in ['dry_days']:
                main(runid=runid, lat=o.lat, lon=o.lon, window=o.window, start=o.start, end=o.end)
    else:
        print("no runid provided")