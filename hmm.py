import numpy as np

from utils import ordered_unique

class HMM:
    def __init__(self, startprob=None, emissionprob=None, transmat=None, n_iter=100, random_state=None):
        self.startprob = startprob
        self.emissionprob = emissionprob
        self.transmat = transmat
        self.n_iter = n_iter
        self.random_state = random_state
        self.labels = None

    def __str__(self):
        return (
            "HMM Model:\n"
            f"{ {'prior': self.startprob, 'emission': self.emissionprob, 'transition': self.transmat} }"
        )

    def fit(self, Y_pred, Y_true, groups=None):
        self.labels = np.unique(Y_true)
        self.startprob = self.compute_prior(Y_true, self.labels)
        self.emissionprob = self.compute_emission(Y_pred, Y_true, self.labels)
        self.transmat = self.compute_transition(Y_true, self.labels, groups)

    def predict(self, Y, groups=None):
        params = {
            'prior': self.startprob,
            'emission': self.emissionprob,
            'transition': self.transmat,
            'labels': self.labels,
        }

        if groups is None:
            Y_vit, _ = self._viterbi(Y, params)

        else:
            Y_vit = np.concatenate([
                self._viterbi(Y[groups == g], params)
                for g in ordered_unique(groups)
            ])

        return Y_vit

    def predict_proba(self, Y, groups=None):
        params = {
            'prior': self.startprob,
            'emission': self.emissionprob,
            'transition': self.transmat,
            'labels': self.labels,
        }
        if groups is None:
            _, probs = self._viterbi(Y, params, True)

        else:
            probs = np.concatenate([
                self._viterbi(Y[groups == g], params, True)
                for g in ordered_unique(groups)
            ])
        return probs

    def optimise(self, **kwargs):
        return

    @staticmethod
    def compute_transition(Y, labels=None, groups=None):
        """ Compute transition matrix from sequence """
        if labels is None:
            labels = np.unique(Y)

        def _compute_transition(Y):
            transition = np.vstack([
                np.sum(Y[1:][(Y == label)[:-1]].reshape(-1, 1) == labels, axis=0)
                for label in labels
            ])
            return transition

        if groups is None:
            transition = _compute_transition(Y)
        else:
            transition = sum((
                _compute_transition(Y[groups == g])
                for g in ordered_unique(groups)
            ))

        transition = transition / np.sum(transition, axis=1).reshape(-1, 1)

        return transition

    @staticmethod
    def compute_emission(Y_score, Y_true, labels=None):
        """ Compute emission matrix from predicted scores and true sequences """
        if labels is None:
            labels = np.unique(Y_true)

        if Y_score.ndim == 1:
            Y_pred = np.vstack([
                (Y_score == label).astype('float')[:, None]
                for label in labels
            ])

        else:
            Y_pred = Y_score

        emission = np.vstack(
            [np.mean(Y_pred[Y_true == label], axis=0) for label in labels]
        )

        return emission

    @staticmethod
    def compute_prior(Y_true, labels=None, uniform=True):
        """ Compute prior probabilities from sequence """
        if labels is None:
            labels = np.unique(Y_true)

        if uniform:
            # all labels with equal probability
            prior = np.ones(len(labels)) / len(labels)

        else:
            # label probability equals observed rate
            prior = np.mean(Y_true.reshape(-1, 1) == labels, axis=0)

        return prior

    def _viterbi(self, Y, hmm_params, return_probs=False):
        ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''
        if len(Y) == 0:
            return np.empty_like(Y)

        def log(x):
            SMALL_NUMBER = 1e-16
            return np.log(x + SMALL_NUMBER)

        prior = hmm_params['prior']
        emission = hmm_params['emission']
        transition = hmm_params['transition']
        labels = hmm_params['labels']

        nobs = len(Y)
        nlabels = len(labels)

        Y = np.where(Y.reshape(-1, 1) == labels)[1]  # to numeric

        probs = np.zeros((nobs, nlabels))
        probs[0, :] = log(prior) + log(emission[:, Y[0]])
        for j in range(1, nobs):
            for i in range(nlabels):
                probs[j, i] = np.max(
                    log(emission[i, Y[j]]) +
                    log(transition[:, i]) +
                    probs[j - 1, :])  # probs already in log scale
        viterbi_path = np.zeros_like(Y)
        viterbi_path[-1] = np.argmax(probs[-1, :])
        for j in reversed(range(nobs - 1)):
            viterbi_path[j] = np.argmax(
                log(transition[:, viterbi_path[j + 1]]) +
                probs[j, :])  # probs already in log scale

        viterbi_path = labels[viterbi_path]  # to labels

        if return_probs:
            return np.exp(probs)  # Return the probabilities in non-log scale

        return viterbi_path
