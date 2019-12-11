import settings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plotting.helper_functions import get_parser


def main(runid, lat, lon, rolling_window, ax=None):

    data_dir = settings.output_dir.parents[0]
    variable = settings.variable

    if lon is None:
        files = [file for file in
                 (data_dir/runid/'timeseries'/variable/f'lat_{lat}').iterdir()]
    else:
        files = [data_dir/runid/'timeseries'/variable/f'lat_{lat}'/f'ts_GSWP3_lat{lat}_lon{lon}.h5']

    for file in files:
        df = pd.read_hdf(file)
        df = df.set_index('ds')
        df_selected = df[['y_scaled', 'cfact_scaled', 'mu', 'mu_ref']]
        rolling_yearly_means = df_selected.rolling(rolling_window, min_periods=1).mean()
        rolling_yearly_means.plot(alpha=.5, title=f'mean with rolling_window {rolling_window}', ax=ax)
        if ax is None:
            plt.show()


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            main(runid=runid, lat=o.lat, lon=o.lon,
                 rolling_window=o.rolling_window)
    else:
        print("no runid provided")