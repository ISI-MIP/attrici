from pathlib import Path
import argparse
import numpy as np


def get_path(data_dir, var, dataset, runid):
    return data_dir/Path(runid)/"cfact"/var/Path(
        var+"_"+dataset.upper()+"_cfactual_monmean.nc4")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid', nargs='*', help='provide name of the experiment.')
    return parser

def get_ylims(ylim):
    return np.array(ylim)[:,0].min(), np.array(ylim)[:,1].max()