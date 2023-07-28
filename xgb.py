import argparse
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import load_dict, save_dict


DEFAULT_ATTRIBUTES = {
    'n_estimators': 250,
    'objective': 'multi:softmax',
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': 12,
    'random_state': 42
}

DEFAULT_PARAM_GRID = {
    'max_depth': hp.choice('max_depth', np.arange(3, 19, dtype=int)),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.choice('reg_alpha', np.arange(40, 181, dtype=int)),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.choice('min_child_weight', np.arange(0, 11, dtype=int)),
    'n_estimators': hp.choice('n_estimators', np.arange(60, 301, 5, dtype=int))
}


def load_xgb(optimisedir='', **kwargs):
    optimised_attributes = load_dict(optimisedir) if optimisedir and os.path.exists(optimisedir) else {}

    model_attributes = {**DEFAULT_ATTRIBUTES, **optimised_attributes, **kwargs}

    model = XGBClassifier(**model_attributes)

    return model


class XGBoostClassifierWrapper:
    def __init__(self, **kwargs):
        self.model = load_xgb(**kwargs)
        self.le = LabelEncoder()

    def __str__(self):
        return (
            "XGBoost Classifier:\n"
            f"  Model Parameters: {self.model.get_params()}\n"
        )

    def fit(self, X, y):
        y = self.le.fit_transform(y)
        self.model.fit(X, y)

    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def optimise(self, X, y, groups=None, param_grid=DEFAULT_PARAM_GRID, 
                 outdir='optimised_params/xgb.pkl'):
        def objective(space):
            clf = load_xgb(**space)

            sample_weight = compute_sample_weight('balanced', y)

            f1_scores = cross_val_score(clf, X, y, groups=groups, cv=3, 
                                        scoring=make_scorer(f1_score, average='macro'),
                                        fit_params={'sample_weight': sample_weight})
            mean_f1 = np.mean(f1_scores)

            return {'loss': -mean_f1, 'status': STATUS_OK}

        y = self.le.fit_transform(y)     

        trials = Trials()

        best = fmin(fn=objective, space=param_grid, algo=tpe.suggest, max_evals=100, 
                    trials=trials, verbose=1, rstate=np.random.default_rng(42))

        optimised_params = space_eval(param_grid, best)

        save_dict(optimised_params, outdir)

        self.model = load_xgb(outdir) 

        return optimised_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='prepared_data')
    parser.add_argument('--annot', '-a', default='WillettsSpecific2018')
    parser.add_argument('--optimisedir', '-o', default='optimised_params')
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    X = pd.read_pickle(os.path.join(args.datadir, "X_feats.pkl")).values
    y = np.load(os.path.join(args.datadir, f"Y_{args.annot}.npy"))
    P = np.load(os.path.join(args.datadir, "P.npy"))

    if args.smoke_test:
        np.random.seed(42)
        smoke_idx = np.random.randint(len(y), size=int(0.01 * len(y)))

        X, y, P = X[smoke_idx], y[smoke_idx], P[smoke_idx]

    xgb = XGBoostClassifierWrapper()
    smoke_flag = "_smoke" if args.smoke_test else ""
    params = xgb.optimise(X, y, P, outdir=f'{args.optimisedir}/xgb_{args.annot}{smoke_flag}.pkl')

    print(f"Best params: {params}")
