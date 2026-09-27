"""Exact propagation of a plane wave through a sequence of temporal slabs.

For a spatially uniform, lossless, non-magnetic dielectric with relative permittivity
eps_r(t), every wavevector k evolves independently according to the Hill equation

    D'' + omega^2 [1 + mu(t)] D = 0,    omega = c k / sqrt(eps_bar),    mu = eps_bar/eps_r - 1,

with D and dD/dt continuous at a temporal interface. Time is measured in units of the slab
duration tau and frequencies in units of 1/tau. Inside a slab mu is constant, so one slab
maps (D, D') by an exact real 2x2 matrix of unit determinant.

Energies are measured with the quadratic form E = omega^2 |D|^2 + |D'|^2, which equals
2 omega^2 (|a|^2 + |b|^2) for D = a exp(-i omega t) + b exp(i omega t).
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss


def _cs(w2, tau):
    """Entries of the slab matrix [[c, s], [-w2 s, c]] for squared frequency w2."""
    a = np.sqrt(np.abs(w2))
    x = a * tau
    c = np.cos(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sin(x) / a
    nonpos = w2 <= 0.0
    if np.any(nonpos):
        # mu <= -1 inside a slab (a far tail of the Gaussian ensembles): the solution is
        # hyperbolic there. w2 = 0 gives c = 1, s = tau.
        t = np.broadcast_to(tau, np.shape(w2))[nonpos]
        an = a[nonpos]
        c[nonpos] = np.cosh(an * t)
        with np.errstate(divide="ignore", invalid="ignore"):
            sn = np.sinh(an * t) / an
        s[nonpos] = np.where(an > 0.0, sn, t)
    return c, s


def slab_matrix(w2, tau=1.0):
    """Propagator of (D, D') across one slab of squared frequency w2."""
    c, s = _cs(np.array([float(w2)]), float(tau))
    return np.array([[c[0], s[0]], [-w2 * s[0], c[0]]])


def propagate(omega, mu, tau=1.0, every=2):
    """Propagate the forward-launched solution through the slab sequences in `mu`.

    omega : scalar or (R,) mode frequency of each row.
    mu    : (R, N) modulation sequences.
    tau   : scalar slab duration, or (R, N) array of individual durations.
    every : checkpoint stride in slabs.

    The launch is D(0) = 1, D'(0) = -i omega (a purely forward wave, a = 1, b = 0). Because
    the slab matrices are real, the real part of this solution is the standing-wave
    solution with D(0) = 1, D'(0) = 0, so both launches come from one pass.

    The default stride of two slabs samples the energy densely enough that the bounded,
    non-resonant oscillations of ln E (frequencies below pi/2 for modes near the removed
    band) are not aliased into the fitted slopes.

    Returns (t, lnE_forward, lnE_standing): elapsed time at the checkpoints, (K,) or
    (R, K), and ln[E(t)/E(0)] for the two launches, each (R, K).
    """
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    R, N = mu.shape
    om = np.broadcast_to(np.asarray(omega, dtype=float), (R,)).astype(float)
    om2 = om * om
    tau = np.asarray(tau, dtype=float)
    variable = tau.ndim > 0
    if variable:
        tau = np.broadcast_to(tau, (R, N))
    K = N // every
    D = np.ones(R, dtype=complex)
    Dd = -1j * om
    lnEf = np.empty((R, K))
    lnEs = np.empty((R, K))
    t = np.empty((R, K)) if variable else (np.arange(1, K + 1) * every) * float(tau)
    elapsed = np.zeros(R)
    k = 0
    for n in range(N):
        w2 = om2 * (1.0 + mu[:, n])
        tn = tau[:, n] if variable else tau
        c, s = _cs(w2, tn)
        D, Dd = c * D + s * Dd, c * Dd - (w2 * s) * D
        if variable:
            elapsed += tn
        if (n + 1) % every == 0 and k < K:
            lnEf[:, k] = np.log(om2 * (D.real ** 2 + D.imag ** 2) + Dd.real ** 2 + Dd.imag ** 2)
            lnEs[:, k] = np.log(om2 * D.real ** 2 + Dd.real ** 2)
            if variable:
                t[:, k] = elapsed
            k += 1
    lnEf -= np.log(2.0 * om2)[:, None]
    lnEs -= np.log(om2)[:, None]
    return t, lnEf, lnEs


def ols_slopes(t, y, window=(0.2, 0.8)):
    """Least-squares slope of each row of y against t over a fractional time window.

    The window is [window[0] T, window[1] T] with T the final elapsed time of the row, so
    rows with individual time grids (timing jitter) are handled consistently.
    """
    y = np.atleast_2d(y)
    t = np.broadcast_to(t, y.shape)
    T = t[:, -1:]
    w = ((t >= window[0] * T - 1e-9) & (t <= window[1] * T + 1e-9)).astype(float)
    if not np.all(w.sum(axis=1) >= 3):
        raise ValueError("the fit window keeps fewer than three checkpoints")
    n = w.sum(axis=1)
    tm = (w * t).sum(axis=1) / n
    ym = (w * y).sum(axis=1) / n
    dt = (t - tm[:, None]) * w
    return (dt * (y - ym[:, None])).sum(axis=1) / (dt * (t - tm[:, None])).sum(axis=1)


def log_mean_energy(lnE):
    """ln of the ensemble-mean energy at each checkpoint, from per-row ln E (rows = axis 0)."""
    m = lnE.max(axis=0)
    return m + np.log(np.mean(np.exp(lnE - m), axis=0))


def lyapunov_rate(omega, mu, renorm=64):
    """Top Lyapunov exponent of the slab product, returned as 2 lambda per unit time.

    The first column of the QR iteration is the standing-wave solution D(0) = 1, D'(0) = 0.
    It is propagated in the variables y = (omega D, D'), in which the free evolution is a
    rotation, and renormalized every `renorm` slabs. The estimate is (1/N) ln[E(N)/E(0)]
    for that launch, which is exactly zero for a constant medium.
    """
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    R, N = mu.shape
    om = np.broadcast_to(np.asarray(omega, dtype=float), (R,)).astype(float)
    om2 = om * om
    u = np.ones(R)
    v = np.zeros(R)
    acc = np.zeros(R)
    for n in range(N):
        w2 = om2 * (1.0 + mu[:, n])
        c, s = _cs(w2, 1.0)
        u, v = c * u + (om * s) * v, c * v - (w2 * s / om) * u
        if (n + 1) % renorm == 0 or n == N - 1:
            r = np.hypot(u, v)
            acc += np.log(r)
            u /= r
            v /= r
    return 2.0 * acc / N


def kronecker_rate(omega, m0, nodes=41, tau=1.0):
    """Exact mean-energy growth rate Gamma_2 for independent Gaussian slabs.

    For independent slabs <z_N z_N^*> evolves linearly with E[M (x) M], so
    Gamma_2 = ln rho(E[M (x) M]) / tau at any depth m0. The Gaussian average uses
    Gauss-Hermite quadrature.
    """
    x, w = hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    K = np.zeros((4, 4))
    for xi, wi in zip(x, w):
        M = slab_matrix(omega ** 2 * (1.0 + m0 * xi), tau)
        K += wi * np.kron(M, M)
    return float(np.log(np.max(np.abs(np.linalg.eigvals(K)))) / tau)


def born_rate_iid(omega, m0, tau=1.0):
    """Leading-order Gamma_2 = (omega^2/2) S_mu(2 omega) for independent slabs of variance
    m0^2, whose staircase spectrum is S_mu(w) = m0^2 tau sinc^2(w tau/2)."""
    omega = np.asarray(omega, dtype=float)
    return m0 ** 2 * np.sin(omega * tau) ** 2 / (2.0 * tau)


def two_slab_trace(omega, m0):
    """Trace of the period-2 monodromy for alternating slabs mu = +m0, -m0."""
    A = slab_matrix(omega ** 2 * (1.0 + m0))
    B = slab_matrix(omega ** 2 * (1.0 - m0))
    return float(np.trace(B @ A))


def two_slab_rate(omega, m0):
    """Exact energy growth rate 2 lambda of the periodic two-slab crystal (zero outside
    its momentum gaps)."""
    tr = abs(two_slab_trace(omega, m0))
    if tr <= 2.0:
        return 0.0
    rho = 0.5 * (tr + np.sqrt(tr * tr - 4.0))
    return float(np.log(rho))          # 2 lambda = 2 ln(rho) / (period 2)
