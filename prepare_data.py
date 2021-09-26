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
ANNOFILE = 'annotation-label-dictionary.csv'
SAMPLE_RATE = 100


def main(args):

    os.system(f'mkdir -p {args.outdir}')

    # Extract windows
    windows = Parallel(n_jobs=args.n_jobs)(
        delayed(make_windows)(datafile, winsec=args.winsec)
        for datafile in tqdm(glob(os.path.join(args.datadir, DATAFILES)))
    )

    X = np.concatenate([_X for _X, _ in windows])
    featframe = pd.concat([_featframe for _, _featframe in windows])
    del windows

    # Now add the labels
    anno_label_dict = pd.read_csv(os.path.join(args.datadir, ANNOFILE),
                                  index_col='annotation', dtype='string')

    featframe = pd.concat(
        [featframe.reset_index(drop=True),
         anno_label_dict.reindex(featframe['annotation']).reset_index(drop=True)],
        axis=1
    )

    featframe.to_pickle(os.path.join(args.outdir, 'featframe.pkl'))

    # Save these separately as numpy arrays
    time = featframe['time'].to_numpy()
    pid = featframe['pid'].to_numpy()
    anno = featframe['annotation'].to_numpy()
    Y_willetts = featframe['label:Willetts2018'].to_numpy().astype('str')
    Y_doherty = featframe['label:Doherty2018'].to_numpy().astype('str')
    Y_walmsley = featframe['label:Walmsley2020'].to_numpy().astype('str')

    np.save(os.path.join(args.outdir, 'X'), X)
    np.save(os.path.join(args.outdir, 'time'), time)
    np.save(os.path.join(args.outdir, 'pid'), pid)
    np.save(os.path.join(args.outdir, 'annotation'), anno)
    np.save(os.path.join(args.outdir, 'Y_willetts'), Y_willetts)
    np.save(os.path.join(args.outdir, 'Y_doherty'), Y_doherty)
    np.save(os.path.join(args.outdir, 'Y_walmsley'), Y_walmsley)

    # List of feature names
    np.savetxt(os.path.join(args.outdir, 'features.txt'), features.get_feature_names(), fmt='%s')


def make_windows(datafile, winsec=10, sample_rate=SAMPLE_RATE, dropna=True):

    pattern = re.compile(r'(P\d{3})')  # 'P123'
    pid = pattern.search(os.path.basename(datafile)).group(1).upper()

    data = pd.read_csv(datafile,
                       index_col='time', parse_dates=['time'],
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})

    featframe = []
    X = []

    for t, w in data.resample(f"{winsec}s", origin='start'):

        xyz = w[['x', 'y', 'z']].to_numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to sort modes")
            anno = w['annotation'].mode(dropna=False).iloc[0]

        if dropna and pd.isna(anno):  # skip if annotation is NA
            continue

        if not is_good_window(xyz, sample_rate, winsec):  # skip if bad window
            continue

        X.append(xyz)
        featframe.append({
            'time': t,
            **features.extract_features(xyz, sample_rate),
            'annotation': anno,
            'pid': pid})

    X = np.stack(X)
    featframe = pd.DataFrame(featframe)

    return X, featframe


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
    parser.add_argument('--outdir', '-o', default='prepared_data')
    parser.add_argument('--winsec', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()

    main(args)
