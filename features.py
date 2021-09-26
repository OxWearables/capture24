import numpy as np
import scipy.stats as stats
import scipy.signal as signal


def extract_features(xyz, sample_rate=100):
    ''' Extract commonly used HAR time-series features. xyz is a window of shape (N,3) '''

    feats = {}

    x, y, z = xyz.T

    feats['xmin'], feats['xq25'], feats['xmed'], feats['xq75'], feats['xmax'] = np.quantile(x, (0, .25, .5, .75, 1))
    feats['ymin'], feats['yq25'], feats['ymed'], feats['yq75'], feats['ymax'] = np.quantile(y, (0, .25, .5, .75, 1))
    feats['zmin'], feats['zq25'], feats['zmed'], feats['zq75'], feats['zmax'] = np.quantile(z, (0, .25, .5, .75, 1))

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        # xy, xy, zx correlation
        feats['xycorr'] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats['yzcorr'] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats['zxcorr'] = np.nan_to_num(np.corrcoef(z, x)[0, 1])

    v = np.linalg.norm(xyz, axis=1)

    feats['min'], feats['q25'], feats['med'], feats['q75'], feats['max'] = np.quantile(v, (0, .25, .5, .75, 1))

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        # 1s autocorrelation
        feats['corr1s'] = np.nan_to_num(np.corrcoef(v[:-sample_rate], v[sample_rate:]))[0, 1]

    # Angular features
    feats.update(angular_features(xyz, sample_rate))

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
        feats['p1'] = peak_powers[peak_ranks[0]]
        feats['p2'] = peak_powers[peak_ranks[1]]
    elif len(peaks) == 1:
        feats['f1'] = feats['f2'] = peak_freqs[peak_ranks[0]]
        feats['p1'] = feats['p2'] = peak_powers[peak_ranks[0]]
    else:
        feats['f1'] = feats['f2'] = 0
        feats['p1'] = feats['p2'] = 0

    return feats


def peak_features(v, sample_rate):
    """ Features of the signal peaks. A proxy to step counts. """

    feats = {}
    u = butterfilt(v, (.6, 5), fs=sample_rate)
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['numPeaks'] = len(peaks)
    if len(peak_props['prominences']) > 0:
        feats['peakPromin'] = np.median(peak_props['prominences'])
    else:
        feats['peakPromin'] = 0

    return feats


def angular_features(xyz, sample_rate):
    """ Roll, pitch, yaw.
    Hip and Wrist Accelerometer Algorithms for Free-Living Behavior
    Classification, Ellis et al.
    """

    feats = {}

    # Raw angles
    x, y, z = xyz.T

    roll = np.arctan2(y, z)
    pitch = np.arctan2(x, z)
    yaw = np.arctan2(y, x)

    feats['avgroll'] = np.mean(roll)
    feats['avgpitch'] = np.mean(pitch)
    feats['avgyaw'] = np.mean(yaw)
    feats['sdroll'] = np.std(roll)
    feats['sdpitch'] = np.std(pitch)
    feats['sdyaw'] = np.std(yaw)

    # Gravity angles
    xyz = butterfilt(xyz, 0.5, fs=sample_rate)

    x, y, z = xyz.T

    roll = np.arctan2(y, z)
    pitch = np.arctan2(x, z)
    yaw = np.arctan2(y, x)

    feats['rollg'] = np.mean(roll)
    feats['pitchg'] = np.mean(pitch)
    feats['yawg'] = np.mean(yaw)

    return feats


def butterfilt(x, cutoffs, fs, order=10, axis=0):
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

    feats = extract_features(np.zeros((1000, 3)), 100)
    return list(feats.keys())
