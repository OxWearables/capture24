import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


class Augment(object):
    def __init__(
        self,
        jitter_sigma=0.1, jitter_prob=0,
        shift_window=1, shift_prob=0,
        twarp_sigma=0.1, twarp_knots=4, twarp_prob=0,
        mwarp_sigma=0.1, mwarp_knots=4, mwarp_prob=0,
    ):

        self.jitter_sigma = jitter_sigma
        self.shift_window = shift_window
        self.twarp_sigma = twarp_sigma
        self.twarp_knots = twarp_knots
        self.mwarp_sigma = mwarp_sigma
        self.mwarp_knots = mwarp_knots

        self.jitter_prob = jitter_prob
        self.shift_prob = shift_prob
        self.twarp_prob = twarp_prob
        self.mwarp_prob = mwarp_prob

    def __call__(self, x):
        if self.jitter_prob > np.random.rand() and self.jitter_sigma > 0:
            x = jitter(x, self.jitter_sigma)
        if self.shift_prob > np.random.rand() and self.shift_window > 0:
            x = shift(x, self.shift_window)
        if self.twarp_prob > np.random.rand() and self.twarp_sigma > 0:
            x = time_warp(x, self.twarp_sigma, self.twarp_knots)
        if self.mwarp_prob > np.random.rand() and self.mwarp_sigma > 0:
            x = magnitude_warp(x, self.mwarp_sigma, self.mwarp_knots)
        return x


def jitter(x, sigma=0.1):
    eps = np.random.randn(*x.shape).astype(x.dtype) * sigma
    x_new = x + eps
    return x_new


def shift(x, window=2):
    """ Note: assumes x in channels last format """
    w = np.random.randint(-window, window)
    x_new = np.roll(x, w, axis=0)
    return x_new


def time_warp(x, sigma=0.1, knots=4):
    """ Note: assumes x in channels last format """
    n = len(x)
    t = warped_timesteps(n, sigma, knots, dtype=x.dtype)
    f = interp1d(t, x, axis=0, assume_sorted=True, copy=False)
    x_new = f(np.arange(n, dtype=x.dtype))
    return x_new


def magnitude_warp(x, sigma=0.1, knots=4):
    """ Note: assumes x in channels last format """
    n = len(x)
    x_new = x * random_curve(n, sigma, knots, dtype=x.dtype).reshape(n, 1)
    return x_new


def random_curve(n, sigma=0.1, knots=4, dtype='f4'):
    x = np.arange(0, n, (n - 1) / (knots + 1), dtype=dtype)
    y = np.random.randn(knots + 2).astype(dtype) * sigma + 1.0
    cs = CubicSpline(x, y)
    curve = cs(np.arange(n, dtype=dtype))
    curve = curve.astype(dtype)
    return curve


def warped_timesteps(n, sigma=0.1, knots=4, dtype='f4'):
    t = np.cumsum(random_curve(n, sigma, knots, dtype))
    # Shift and scale so that endpoints are fixed
    t = (t - t[0]) / (t[-1] - t[0]) * (n - 1)
    return t
