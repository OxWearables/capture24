import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import urllib.request as urllib
import zipfile
import warnings

from features import extract_features
from utils import check_files_exist

DATAFILES = 'capture24/P[0-9][0-9][0-9].csv.gz'
ANNOFILE = 'capture24/annotation-label-dictionary.csv'
SAMPLE_RATE = 100


def download_capture24(datadir, overwrite=False):
    """ Download and extract the capture-24 dataset """
    if overwrite or not os.path.exists(os.path.join(datadir, 'capture24.zip')):
        url = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001" + \
              "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"

        with tqdm(total=6.9e9, unit="B", unit_scale=True, unit_divisor=1024, 
                  miniters=1, ascii=True, desc="Downloading capture24.zip") as pbar:
            urllib.urlretrieve(url, filename=os.path.join(datadir, "capture24.zip"),
                               reporthook=lambda b, bsize, tsize: pbar.update(bsize))

    capture24dir = os.path.join(datadir, 'capture24')

    if overwrite or len(glob(os.path.join(datadir, DATAFILES))) < 151:
        with zipfile.ZipFile(os.path.join(datadir, "capture24.zip"), "r") as f:
            os.makedirs(capture24dir, exist_ok=True)
            for member in tqdm(f.namelist(), desc="Unzipping"):
                try:
                    f.extract(member, datadir)
                except zipfile.error:
                    pass
    else:
        print(f"Using saved capture-24 data at \"{capture24dir}\".")


def load_data(datafile):
    """ Load the data files with correct dtypes """
    data = pd.read_csv(
        datafile,
        index_col='time', parse_dates=['time'],
        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}
    )
    return data


def make_windows(data, anno_df, anno_cols, winsec=30, sample_rate=SAMPLE_RATE, dropna=True, verbose=False):
    """ Segment accelerometer data into winsec*sample_rate length windows """
    X, y_anno, y_cols, T = [], [], {col: [] for col in anno_cols}, []

    for t, w in tqdm(data.resample(f"{winsec}s", origin='start'), disable=not verbose):
        if len(w) < 1:
            continue

        t = t.to_numpy()
        x = w[['x', 'y', 'z']].to_numpy()
        annot = w['annotation']

        if dropna and pd.isna(annot).all():  # skip if annotation is NA
            continue

        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to sort modes")
            y_anno_value = annot.mode(dropna=False).iloc[0]
            y_anno.append(y_anno_value)

            for col in anno_cols:
                y_col_value = anno_df.loc[annot.dropna(), f'label:{col}'].mode(dropna=False).iloc[0]
                y_cols[col].append(y_col_value)

        X.append(x)
        T.append(t)

    X = np.stack(X)
    T = np.stack(T)

    return X, y_anno, y_cols, T


def is_good_window(x, sample_rate, winsec):
    ''' Check there are no NaNs and len is good '''

    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True


def load_all_and_make_windows(datadir, anno_cols, outdir, n_jobs, overwrite=False, **kwargs):
    """ Make windows from all available data, extract features and store locally """
    if not overwrite and check_files_exist(outdir, ['X.npy', 'Y_anno.npy', 'T.npy', 'P.npy'] + [f'Y_{col}.npy' for col in anno_cols]):
        print(f"Using files saved at \"{outdir}\".")
        return

    def worker(datafile):
        X, y_anno, y_cols, T = make_windows(load_data(datafile), anno_df, anno_cols, **kwargs)

        pid = Path(datafile)

        for _ in pid.suffixes:
            pid = Path(pid.stem)

        pid = str(pid)

        P = np.array([pid] * len(X))

        return X, y_anno, y_cols, T, P

    datafiles = glob(os.path.join(datadir, DATAFILES))
    anno_df = pd.read_csv(os.path.join(datadir, ANNOFILE), index_col="annotation", dtype=str)

    X, y_anno, y_cols, T, P = zip(*Parallel(n_jobs=n_jobs)(
        delayed(worker)(datafile) 
        for datafile in tqdm(datafiles, desc="Load all and make windows")))

    X = np.vstack(X)
    T = np.hstack(T)
    P = np.hstack(P)

    Y_anno = np.hstack(y_anno)

    Y_cols_df = pd.DataFrame(y_cols)
    Y_cols = {col: np.hstack(Y_cols_df[col]) for col in Y_cols_df}

    X_feats = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(extract_features)(x) for x in tqdm(X, desc="Extracting features")))

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'X.npy'), X)
    np.save(os.path.join(outdir, 'Y_anno.npy'), Y_anno)
    for col, vals in Y_cols.items():
        np.save(os.path.join(outdir, f'Y_{col}.npy'), vals)
    np.save(os.path.join(outdir, 'T.npy'), T)
    np.save(os.path.join(outdir, 'P.npy'), P)
    X_feats.to_pickle(os.path.join(outdir, 'X_feats.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='data')
    parser.add_argument('--outdir', '-o', default='prepared_data')
    parser.add_argument('--annots', '-a', default='Walmsley2020,WillettsSpecific2018')
    parser.add_argument('--winsec', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    download_capture24(args.datadir, args.overwrite)
    load_all_and_make_windows(args.datadir, args.annots.split(","), args.outdir, args.n_jobs, 
                              args.overwrite, winsec=args.winsec)
