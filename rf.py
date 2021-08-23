# %%
import os
import argparse
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier

import utils


def main(args):
    # For reproducibility
    np.random.seed(42)

    data = pd.read_pickle(os.path.join(args.datadir, 'featframe.pkl'))
    featcols = np.loadtxt(os.path.join(args.datadir, 'features.txt'), dtype='str')
    labelcol = args.label

    if args.smoke_test:
        data = data.sample(frac=0.1, random_state=42)

    # Use P001-P100 as derivation set and the rest as test set
    data_deriv = data[data['pid'].str.contains('P0[0-9][0-9]|P100')]
    data_test = data[~data['pid'].str.contains('P0[0-9][0-9]|P100')]

    X = data_deriv[featcols].to_numpy()
    Y = data_deriv[labelcol].to_numpy()

    X_test = data_test[featcols].to_numpy()
    Y_test = data_test[labelcol].to_numpy()

    clf = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        replacement=True,
        sampling_strategy='not minority',
        oob_score=True,
        n_jobs=args.n_jobs,
        random_state=42,
    )
    clf.fit(X, Y)

    Y_pred = clf.predict(X_test)

    print("RF performance:")
    utils.metrics_report(Y_test, Y_pred, n_jobs=args.n_jobs)

    hmm_params = utils.train_hmm(clf.oob_decision_function_, Y, clf.classes_)
    Y_pred_hmm = utils.viterbi(Y_pred, hmm_params)

    print("RF + HMM performance:")
    utils.metrics_report(Y_test, Y_pred_hmm, n_jobs=args.n_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--label', default='label:Willetts2018')
    parser.add_argument('--n_estimators', type=int, default=3000)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    main(args)
