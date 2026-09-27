"""Random modulation sequences, one value of mu per slab.

Every generator takes a numpy Generator as its first argument and returns an (M, N) array.
`rng(*tags)` hashes its tags with SHA-256 into a SeedSequence, so each ensemble is fixed by
its tags alone and does not depend on the order in which ensembles are drawn.

Spectral conventions: sequence j = 0..N-1 has discrete Fourier bins q = 0..N/2 at angular
frequency w_q = 2 pi q / N (per slab). A band (lo, hi) refers to that frequency axis.
"""
import hashlib

import numpy as np


def rng(*tags):
    """Generator seeded from the SHA-256 digest of the tags."""
    key = "|".join(f"{t:.6g}" if isinstance(t, float) else str(t) for t in tags)
    digest = hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(int.from_bytes(digest, "little")))


def band_bins(N, band):
    """Positive-frequency DFT bins whose angular frequency lies in the closed band."""
    q = np.arange(N // 2 + 1)
    w = 2.0 * np.pi * q / N
    return q[(w >= band[0]) & (w <= band[1])]


def kappa2(N, n_constrained):
    """Out-of-band spectral enhancement for a fixed-variance sequence with the DC bin and
    n_constrained positive-frequency bins (and their mirrors) removed."""
    return (N - 1.0) / (N - 1.0 - 2.0 * n_constrained)


def _normalize(x, m0):
    x = x - x.mean(axis=1, keepdims=True)
    return x * (m0 / x.std(axis=1, keepdims=True))


def white(g, M, N, m0):
    """Independent Gaussian slabs, each row shifted to zero mean and scaled to std m0."""
    return _normalize(g.standard_normal((M, N)), m0)


def filtered(g, M, N, m0, bins, amplitude=0.0):
    """Gaussian sequence whose Fourier amplitudes in `bins` are multiplied by `amplitude`,
    with the DC bin removed and each row rescaled to std m0.

    amplitude = 0 gives an exactly stealthy sequence: the rescaling is a scalar, so the
    removed bins stay exactly zero.
    """
    F = np.fft.rfft(g.standard_normal((M, N)), axis=1)
    F[:, 0] = 0.0
    F[:, bins] *= amplitude
    x = np.fft.irfft(F, n=N, axis=1)
    return x * (m0 / x.std(axis=1, keepdims=True))


def stealthy(g, M, N, m0, bins):
    """Band-stealthy sequence: zero spectral weight in `bins`."""
    return filtered(g, M, N, m0, bins, 0.0)


def notched(g, M, N, m0, bins, depth_db):
    """Finite notch: in-band power reduced by depth_db decibels instead of removed."""
    return filtered(g, M, N, m0, bins, 10.0 ** (-depth_db / 20.0))


def shuffled(g, x):
    """Random permutation of every row: same values, correlations destroyed."""
    return g.permuted(x, axis=1)


def epsilon_stealthy(g, M, N, m0, bins):
    """Stealthiness imposed on the permittivity modulation delta = eps_r/eps_bar - 1
    instead of on mu. Returns the mu seen by the wave equation, mu = 1/(1 + delta) - 1."""
    delta = stealthy(g, M, N, m0, bins)
    return 1.0 / (1.0 + delta) - 1.0


def stealthy_from_base(g, M, N, m0, bins, base="gaussian", n_iter=0):
    """Exactly stealthy sequence started from a non-Gaussian marginal.

    `n_iter` alternating projections (spectral projection, then rank-order mapping back
    onto the starting marginal) pull the marginal towards the base distribution. The
    spectral projection is always applied last, so the result is exactly stealthy.
    """
    if base == "gaussian":
        x = g.standard_normal((M, N))
    elif base == "bernoulli":
        x = 2.0 * g.integers(0, 2, (M, N)) - 1.0
    elif base == "uniform":
        x = g.uniform(-np.sqrt(3.0), np.sqrt(3.0), (M, N))
    else:
        raise ValueError(f"unknown base marginal {base!r}")
    target = np.sort(x, axis=1)

    def project(y):
        F = np.fft.rfft(y, axis=1)
        F[:, 0] = 0.0
        F[:, bins] = 0.0
        return np.fft.irfft(F, n=N, axis=1)

    for _ in range(int(n_iter)):
        y = project(x)
        ranks = np.argsort(np.argsort(y, axis=1), axis=1)
        x = np.take_along_axis(target, ranks, axis=1)
    y = project(x)
    return y * (m0 / y.std(axis=1, keepdims=True))


def excess_kurtosis(x):
    """Excess kurtosis of each row (zero for a Gaussian marginal)."""
    x = x - x.mean(axis=1, keepdims=True)
    return (x ** 4).mean(axis=1) / (x ** 2).mean(axis=1) ** 2 - 3.0


def jittered_durations(g, M, N, jitter):
    """Slab durations 1 + U(-jitter, jitter), normalized to unit mean in every row."""
    t = 1.0 + g.uniform(-jitter, jitter, (M, N))
    return t / t.mean(axis=1, keepdims=True)


def periodogram(x):
    """Ensemble-averaged periodogram <|X_q|^2>/N on the bins q = 0..N/2."""
    N = x.shape[1]
    return (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0) / N


def inband_fraction(x, bins):
    """Fraction of the (one-sided) spectral power of each row that lies in `bins`."""
    P = np.abs(np.fft.rfft(x, axis=1)) ** 2
    return P[:, bins].sum(axis=1) / P[:, 1:].sum(axis=1)


def continuous_leakage(x, band, pad=8, interior=0.05):
    """Interior in-band power of the zero-padded spectrum, relative to the median power
    outside the band.

    Removing DFT bins makes the spectrum vanish at the bin frequencies only. Between bins the
    finite-length sequence has a small nonzero spectrum; zero padding by `pad` resolves it.
    The band is shrunk by the fraction `interior` at each end so that the finite roll-off at
    the edges is not counted. Returns one value per row.
    """
    M, N = x.shape
    xp = np.zeros((M, N * pad))
    xp[:, :N] = x - x.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(xp, axis=1)) ** 2
    w = 2.0 * np.pi * np.fft.rfftfreq(N * pad)
    span = band[1] - band[0]
    inner = (w >= band[0] + interior * span) & (w <= band[1] - interior * span)
    outer = (w > 0.0) & ((w < band[0]) | (w > band[1]))
    return P[:, inner].max(axis=1) / np.median(P[:, outer], axis=1)
