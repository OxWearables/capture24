import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from scipy.ndimage import median_filter
import statsmodels.tsa.stattools as stattools


MIN_WINDOW_SEC = 2  # seconds


def extract_features(xyz, sample_rate=100):
    ''' Extract commonly used HAR time-series features. xyz is a window of shape (N,3) '''

    if np.isnan(xyz).any():
        return {}

    if len(xyz) <= MIN_WINDOW_SEC * sample_rate:
        return {}

    feats = {}

    v = np.linalg.norm(xyz, axis=1)
    v = median_filter(v, size=5, mode='nearest')
    v = v - 1  # detrend: "remove gravity"
    v = np.clip(v, -2, 2)  # clip abnormaly high values

    # Moments features
    feats.update(moments_features(v, sample_rate))

    # Quantile features
    feats.update(quantile_features(v, sample_rate))

    # Autocorrelation features
    feats.update(autocorr_features(v, sample_rate))

    # Spectral features
    feats.update(spectral_features(v, sample_rate))

    # FFT features
    feats.update(fft_features(v, sample_rate))

    # Peak features
    feats.update(peaks_features(v, sample_rate))

    return feats


def moments_features(v, sample_rate=None):
    """ Moments """
    avg = np.mean(v)
    std = np.std(v)
    if std > .01:
        skew = np.nan_to_num(stats.skew(v))
        kurt = np.nan_to_num(stats.kurtosis(v))
    else:
        skew = kurt = 0
    feats = {
        'avg': avg,
        'std': std,
        'skew': skew,
        'kurt': kurt,
    }
    return feats


def quantile_features(v, sample_rate=None):
    """ Quantiles (min, 25th, med, 75th, max) """
    feats = {}
    feats['min'], feats['q25'], feats['med'], feats['q75'], feats['max'] = np.quantile(v, (0, .25, .5, .75, 1))
    return feats


def autocorr_features(v, sample_rate):
    """ Autocorrelation features """

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore invalid div warnings
        u = np.nan_to_num(stattools.acf(v, nlags=2 * sample_rate))

    peaks, _ = signal.find_peaks(u, prominence=.1)
    if len(peaks) > 0:
        acf_1st_max_loc = peaks[0]
        acf_1st_max = u[acf_1st_max_loc]
        acf_1st_max_loc /= sample_rate  # in secs
    else:
        acf_1st_max = acf_1st_max_loc = 0.0

    valleys, _ = signal.find_peaks(-u, prominence=.1)
    if len(valleys) > 0:
        acf_1st_min_loc = valleys[0]
        acf_1st_min = u[acf_1st_min_loc]
        acf_1st_min_loc /= sample_rate  # in secs
    else:
        acf_1st_min = acf_1st_min_loc = 0.0

    acf_zeros = np.sum(np.diff(np.signbit(u)))

    feats = {
        'acf_1st_max': acf_1st_max,
        'acf_1st_max_loc': acf_1st_max_loc,
        'acf_1st_min': acf_1st_min,
        'acf_1st_min_loc': acf_1st_min_loc,
        'acf_zeros': acf_zeros,
    }

    return feats


def spectral_features(v, sample_rate):
    """ Spectral entropy, average power, dominant frequencies """

    feats = {}

    freqs, powers = signal.periodogram(v, fs=sample_rate, detrend='constant', scaling='density')
    powers /= (len(v) / sample_rate)    # unit/sec

    feats['pentropy'] = stats.entropy(powers[powers > 0])
    feats['power'] = np.sum(powers)

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]

    TOPN = 3
    feats.update({f"f{i + 1}": 0 for i in range(TOPN)})
    feats.update({f"p{i + 1}": 0 for i in range(TOPN)})
    for i, j in enumerate(peak_ranks[:TOPN]):
        feats[f"f{i + 1}"] = peak_freqs[j]
        feats[f"p{i + 1}"] = peak_powers[j]

    return feats


def fft_features(v, sample_rate, nfreqs=5):
    """ Power of frequencies 0Hz, 1Hz, 2Hz, ... using Welch's method """

    _, powers = signal.welch(
        v, fs=sample_rate,
        nperseg=sample_rate,
        noverlap=sample_rate // 2,
        detrend='constant',
        scaling='density',
        average='median'
    )

    feats = {f"fft{i}": powers[i] for i in range(nfreqs + 1)}

    return feats


def peaks_features(v, sample_rate):
    """ Features of the signal peaks """

    feats = {}
    u = butterfilt(v, 5, fs=sample_rate)  # lowpass 5Hz
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['npeaks'] = len(peaks) / (len(v) / sample_rate)  # peaks/sec
    if len(peak_props['prominences']) > 0:
        feats['peaks_avg_promin'] = np.mean(peak_props['prominences'])
        feats['peaks_min_promin'] = np.min(peak_props['prominences'])
        feats['peaks_max_promin'] = np.max(peak_props['prominences'])
    else:
        feats['peaks_avg_promin'] = feats['peaks_min_promin'] = feats['peaks_max_promin'] = 0

    return feats


def butterfilt(x, cutoffs, fs, order=4, axis=0):
    """ Butterworth filter """
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            btype = 'bandpass'
            Wn = (hicut / nyq, lowcut / nyq)
        else:
            btype = 'low'
            Wn = lowcut / nyq
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    return y


def get_feature_names():
    """ Hacky way to get the list of feature names """

    feats = extract_features(np.zeros((500, 3)), 100)
    return list(feats.keys())
