"""Ensemble runs: draw modulation sequences, propagate them, and reduce the result to
growth-rate statistics."""
import numpy as np

from .model import log_mean_energy, lyapunov_rate, ols_slopes, propagate


def run_grid(omegas, draw, M, N, windows=((0.2, 0.8),), every=2, coarse=32,
             max_entries=2.4e7, tau=1.0):
    """Propagate M realizations at every frequency in `omegas`.

    draw(i, M) returns the (M, N) modulation sequences for frequency index i, or a tuple
    (mu, durations) for slabs of unequal length; otherwise every step lasts `tau`. Several
    frequencies are propagated together, about max_entries row-step entries per batch.

    Returns a dict:
      forward   (n_windows, n_omega, M)  least-squares slopes of ln E, forward launch
      standing  (n_windows, n_omega, M)  the same for the standing-wave launch
      t         (K,) coarse checkpoint times, every `coarse` steps
      lnEbar    (n_omega, K) ln of the ensemble-mean forward energy at those times
    Mean-energy trajectories are only formed for slabs of equal duration.
    """
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    n_om, n_win = len(omegas), len(windows)
    per_batch = max(1, int(max_entries // (M * N)))
    step = coarse // every
    K = N // coarse
    fwd = np.empty((n_win, n_om, M))
    std = np.empty((n_win, n_om, M))
    lnEbar = np.full((n_om, K), np.nan)
    t_coarse = np.arange(1, K + 1) * float(coarse) * tau
    for start in range(0, n_om, per_batch):
        idx = list(range(start, min(n_om, start + per_batch)))
        mus, durs = [], []
        for i in idx:
            d = draw(i, M)
            mu, dur = d if isinstance(d, tuple) else (d, None)
            mus.append(mu)
            durs.append(dur)
        mu = np.concatenate(mus)
        variable = durs[0] is not None
        dur = np.concatenate(durs) if variable else tau
        om_rows = np.repeat(omegas[idx], M)
        t, lf, ls = propagate(om_rows, mu, tau=dur, every=every)
        del mu, mus
        for j, i in enumerate(idx):
            rows = slice(j * M, (j + 1) * M)
            tt = t[rows] if variable else t
            for w, win in enumerate(windows):
                fwd[w, i] = ols_slopes(tt, lf[rows], win)
                std[w, i] = ols_slopes(tt, ls[rows], win)
            if not variable:
                lnEbar[i] = log_mean_energy(lf[rows, step - 1::step])
    return {"forward": fwd, "standing": std, "t": t_coarse, "lnEbar": lnEbar}


def mean_se(x, axis=-1):
    """Mean and standard error along an axis."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    return x.mean(axis=axis), x.std(axis=axis, ddof=1) / np.sqrt(n)


def lyapunov_grid(omegas, draw, M, N, max_entries=1.6e7):
    """Top Lyapunov exponents (as 2 lambda) for M realizations at each frequency.
    Returns an array of shape (n_omega, M)."""
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    per_batch = max(1, int(max_entries // (M * N)))
    out = np.empty((len(omegas), M))
    for start in range(0, len(omegas), per_batch):
        idx = list(range(start, min(len(omegas), start + per_batch)))
        mu = np.concatenate([draw(i, M) for i in idx])
        lam = lyapunov_rate(np.repeat(omegas[idx], M), mu)
        for j, i in enumerate(idx):
            out[i] = lam[j * M:(j + 1) * M]
    return out
