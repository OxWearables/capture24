import argparse
import numpy as np
import os
import pandas as pd

from classifier import Classifier
from eval import performance_table

N_JOBS = 12


def main(X, ys, P, annotations, models, optimisedir='', n_jobs=N_JOBS, seed=42, **kwargs):

    train_ids, test_ids = train_test_split(P)

    X_train, P_train =\
        X[train_ids], P[train_ids]

    X_test, P_test =\
        X[test_ids], P[test_ids]

    for annot in annotations:
        y_preds = {}
        y_train, y_test = ys[annot][train_ids], ys[annot][test_ids]

        for model in models:
            classifier = Classifier(model, seed, 
                                    optimisedir=(os.path.join(optimisedir, f"{model}_{annot}.pkl")), **kwargs)

            classifier.fit(X_train, y_train, P_train)
            y_preds[model] = classifier.predict(X_test, P_test)

        performance_table(y_test, y_preds, P_test, title=f"{annot} labels", n_jobs=n_jobs)


def train_test_split(P):
    test_ids = [f'P{i}' for i in range(101, 152)]
    mask_test = np.isin(P, test_ids)
    mask_train = ~mask_test

    return mask_train, mask_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='prepared_data')
    parser.add_argument('--annots', '-a', default='WillettsSpecific2018,Walmsley2020')
    parser.add_argument('--models', '-m', default='rf,rf_hmm,xgb,xgb_hmm')
    parser.add_argument('--optimisedir', '-o', default='optimised_params')
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    annotations = args.annots.split(',')

    X = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    ys = {annot: np.load(os.path.join(args.datadir, f"Y_{annot}.npy")) for annot in annotations}
    P = np.load(os.path.join(args.datadir, "P.npy"))

    if args.smoke_test:
        np.random.seed(42)
        len_y = len(next(iter(ys.values())))
        smoke_idx = np.random.randint(len_y, size=int(0.01 * len_y))

        X, P = X[smoke_idx], P[smoke_idx]
        ys = {annot: ys[annot][smoke_idx] for annot in ys}

    main(X, ys, P, annotations, args.models.split(','), args.optimisedir, N_JOBS)
