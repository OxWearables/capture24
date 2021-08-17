import numpy as np
import scipy.stats as stats
import scipy.signal as signal


def extract_features(xyz, sample_rate=100):
    ''' Extract commonly used HAR time-series features. xyz is a window of shape (N,3) '''

    feats = {}

    # Basics stats
    feats['xMed'], feats['yMed'], feats['zMed'] = np.median(xyz, axis=0)
    feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)
    feats['xIQR'], feats['yIQR'], feats['zIQR'] = stats.iqr(xyz, axis=0)

    v = np.linalg.norm(xyz, axis=1)

    feats['median'] = np.median(v)
    feats['min'] = np.min(v)
    feats['max'] = np.max(v)
    feats['q25'] = np.quantile(v, .25)
    feats['q75'] = np.quantile(v, .75)

    x, y, z = xyz.T

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        # xy, xy, zx correlation
        feats['xyCorr'] = np.nan_to_num(np.corrcoef(x,y)[0,1])
        feats['yzCorr'] = np.nan_to_num(np.corrcoef(y,z)[0,1])
        feats['zxCorr'] = np.nan_to_num(np.corrcoef(z,x)[0,1])
        # 1s autocorrelation
        feats['xxCorr1s'] = np.nan_to_num(np.corrcoef(x[:-sample_rate], x[sample_rate:]))[0,1]
        feats['yyCorr1s'] = np.nan_to_num(np.corrcoef(y[:-sample_rate], y[sample_rate:]))[0,1]
        feats['zzCorr1s'] = np.nan_to_num(np.corrcoef(z[:-sample_rate], z[sample_rate:]))[0,1]

    # Spectral features
    feats.update(spectral_features(v, sample_rate))

    # Peak features
    feats.update(peak_features(v, sample_rate))

    return feats


def spectral_features(v, sample_rate):
    """ Spectral entropy, 1st & 2nd dominant frequencies """

    feats = {}

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(v, fs=sample_rate,
                                 nperseg=3 * sample_rate,
                                 noverlap=2 * sample_rate,
                                 detrend=False,
                                 average='median')

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        feats['pentropy'] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to focus on the relevant freqs
    freqs, powers = signal.welch(v, fs=sample_rate,
                                 nperseg=3 * sample_rate,
                                 noverlap=2 * sample_rate,
                                 detrend='constant',
                                 average='median')

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats['f1'] = peak_freqs[peak_ranks[0]]
        feats['f2'] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats['f1'] = feats['f2'] = peak_freqs[peak_ranks[0]]
    else:
        feats['f1'] = feats['f2'] = 0

    return feats


def peak_features(v, sample_rate):
    """ Features of the signal peaks. A proxy to step counts. """

    feats = {}
    u = butterfilt(v, (.6, 5), fs=sample_rate, order=8)
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['numPeaks'] = len(peaks)
    if len(peak_props['prominences']) > 0:
        feats['peakPromin'] = np.median(peak_props['prominences'])
    else:
        feats['peakPromin'] = 0

    return feats


def butterfilt(x, cutoffs, fs, order=8, axis=0):
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
