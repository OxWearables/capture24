import argparse
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from eval import metrics_report
from hmm import HMM
from rf import RandomForestClassifierWrapper
from xgb import XGBoostClassifierWrapper

class Classifier:
    def __init__(self, model_type, seed=42, **kwargs):
        self.type = model_type
        self.seed = seed
        self.window_classifier = None
        self.smoother = None
        self._initialise_model(**kwargs)

    def __str__(self):
        return (
            "Classifier:\n"
            f"  Model type: {self.type}\n"
            f"  Model: {self.window_classifier}\n"
            f"  Smoother: {self.smoother}\n"
        )

    def _initialise_model(self, **kwargs):
        if 'RF' in self.type.upper().split('_'):
            self.window_classifier = RandomForestClassifierWrapper(oob_score=True, random_state=self.seed, **kwargs)

        elif 'XGB' in self.type.upper().split('_'):
            self.window_classifier = XGBoostClassifierWrapper(random_state=self.seed, **kwargs)

        else:
            raise ValueError("Model type must contain 'rf' or 'xgb'")

        if 'HMM' in self.type.upper().split('_'):
            self.smoother = HMM()

    def fit(self, X, y, groups=None):
        if self.smoother is None:
            self.window_classifier.fit(X, y)

        else:
            if 'RF' in self.type.upper().split('_'):
                self.window_classifier.fit(X, y)
                self.smoother.fit(self.window_classifier.model.oob_decision_function_, y, groups)

            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)

                for train_idx, val_idx in gss.split(X, y, groups=groups):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    group_val = groups[val_idx] if groups is not None else None

                self.window_classifier.fit(X_train, y_train)

                y_val_proba = self.window_classifier.predict_proba(X_val)

                self.smoother.fit(y_val_proba, y_val, group_val)


    def predict(self, X, groups=None):
        if self.smoother is None:
            return self.window_classifier.predict(X)

        else:
            return self.smoother.predict(self.window_classifier.predict(X), groups)

    def predict_proba(self, X, groups=None):
        if self.smoother is None:
            return self.window_classifier.predict_proba(X)

        else:
            return self.smoother.predict_proba(self.window_classifier.predict(X), groups)

    def optimise(self, X, y, groups=None, **kwargs):
        self.window_classifier.optimise(X, y, groups, **kwargs)


class Smoother:
    def __init__(self, model_type, **kwargs):
        self.type = model_type
        self.model = self._initialise_model(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='prepared_data')
    parser.add_argument('--annot', '-a', default='WillettsSpecific2018')
    parser.add_argument('--model_type', '-m', default='xgb_hmm')
    parser.add_argument('--optimisedir', '-o', default='optimised_params')
    args = parser.parse_args()

    X = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    y = np.load(os.path.join(args.datadir, f"Y_{args.annot}.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))

    optimisedir = os.path.join(args.optimisedir, f"{args.model_type}_{args.annot}.pkl")

    model = Classifier(args.model_type, optimisedir=optimisedir)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in gss.split(X, y, groups=P):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        group_train, group_test = P[train_idx], P[test_idx]

    model.fit(X_train, y_train, group_train)
    y_pred = model.predict(X_test, group_test)

    metrics_report(y_test, y_pred, group_test, args.model_type)
