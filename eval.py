from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import display_df


def performance_table(y_true, y_preds, groups=None, title='', nboots=100, n_jobs=4, return_df=False):
    performance_df = pd.DataFrame({y_label: metrics_report(y_true, y_pred, groups, nboots=nboots, 
                                                           n_jobs=n_jobs, verbose=False)
                                   for y_label, y_pred in y_preds.items()}).transpose()

    display_df(performance_df, title)

    if return_df:
        return performance_df


def metrics_report(y_true, y_pred, groups=None, tag="", nboots=100, n_jobs=4, verbose=True):

    if verbose:
        print(f"\n========================= {tag} =========================")

        # Print the cute sklearn report
        print(metrics.classification_report(y_true, y_pred, zero_division=0))

        # Compute balanced accuracy, f1, phi and kappa scores
        # Use bootstrap for confidence intervals

    def f(idxs):
        _Y_true, _Y_pred = y_true[idxs], y_pred[idxs]

        bacc = metrics.balanced_accuracy_score(_Y_true, _Y_pred)
        f1 = metrics.f1_score(_Y_true, _Y_pred, zero_division=0, average='macro')
        phi = metrics.matthews_corrcoef(_Y_true, _Y_pred)
        kappa = metrics.cohen_kappa_score(_Y_true, _Y_pred)

        return np.asarray([bacc, f1, phi, kappa])

    idxs = np.arange(len(y_true))
    (bacc, f1, phi, kappa) = f(idxs)
    (bacc_low, f1_low, phi_low, kappa_low), (bacc_hi, f1_hi, phi_hi, kappa_hi) =\
        bootstrapCI(f, idxs, groups, nboots, n_jobs)

    if verbose:
        print(f" {tag}/bacc: {bacc:.3f} ({bacc_low:.3f}, {bacc_hi:.3f})")
        print(f"   {tag}/f1: {f1:.3f} ({f1_low:.3f}, {f1_hi:.3f})")
        print(f"  {tag}/phi: {phi:.3f} ({phi_low:.3f}, {phi_hi:.3f})")
        print(f"{tag}/kappa: {kappa:.3f} ({kappa_low:.3f}, {kappa_hi:.3f})")

    return {
        'Balanced Accuracy': f"{bacc:.3f} ({bacc_low:.3f}, {bacc_hi:.3f})",
        'Macro F1': f"{f1:.3f} ({f1_low:.3f}, {f1_hi:.3f})",
        'Matthews Correlation Coefficient': f"{phi:.3f} ({phi_low:.3f}, {phi_hi:.3f})",
        'Cohen\'s kappa score': f"{kappa:.3f} ({kappa_low:.3f}, {kappa_hi:.3f})" 
    }


def bootstrapCI(f, sample, groups=None, nboots=100, n_jobs=4, seed=42):
    """ Bootstrap confidence intervals
    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    """

    np.random.seed(seed)

    mu = f(sample)

    if groups is None:
        boots = np.random.choice(sample, size=(nboots, len(sample)))

    else:
        boots = [np.concatenate([sample[groups == group] for group in boot]) 
                 for boot in np.random.choice(groups, size=(nboots, len(np.unique(groups))))]

    if n_jobs != 0:
        mus = Parallel(n_jobs=n_jobs)(delayed(f)(boot) for boot in boots)
    else:
        mus = [f(boot) for boot in boots]
    mus = np.stack(mus)

    mu_hi, mu_low = mu - np.percentile(mus - mu, (2.50, 97.50), axis=0)

    return mu_low, mu_hi
