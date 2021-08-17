import os
import re
import warnings
import argparse
from glob import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import features

DATAFILES = 'P[0-9][0-9][0-9].csv.gz'
SAMPLE_RATE = 100


def main(args):

    featframes = Parallel(n_jobs=args.n_jobs)(
        delayed(make_featframe)(datafile, winsec=args.winsec)
        for datafile in tqdm(glob(os.path.join(args.datadir, DATAFILES))))

    featframes = pd.concat(featframes)

    featframes.to_pickle(args.outfile)


def make_featframe(datafile, winsec=10, sample_rate=SAMPLE_RATE):

    pattern = re.compile(r'(P\d{3})')  # 'P123'
    pid = pattern.search(os.path.basename(datafile)).group(1).upper()

    data = pd.read_csv(datafile,
                       index_col='time', parse_dates=['time'],
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})

    featframe = []

    for t, w in data.resample(f"{winsec}s", origin='start'):

        xyz = w[['x', 'y', 'z']].to_numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to sort modes")
            anno = w['annotation'].mode(dropna=False).iloc[0]

        if not is_good_window(xyz, sample_rate, winsec):
            continue

        featframe.append({'time': t,
                          **features.extract_features(xyz, sample_rate),
                          'annotation': anno,
                          'pid': pid})

    featframe = pd.DataFrame(featframe).sort_values('time')

    return featframe


def is_good_window(xyz, sample_rate, winsec):
    ''' Check there are no NaNs and len is good '''

    # Check window length is correct
    window_len = sample_rate * winsec
    if len(xyz) != window_len:
        return False

    # Check no nans
    if np.isnan(xyz).any():
        return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data/')
    parser.add_argument('--outfile', '-o', default='data.pkl')
    parser.add_argument('--winsec', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()

    main(args)
