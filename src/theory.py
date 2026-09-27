"""Analytical and semi-analytical predictions.

Diffusion limit
---------------
For weak disorder the backscattered intensity rho = |b|^2 of a forward-launched wave obeys
the single-channel diffusion equation

    d rho = (1 + 2 rho) ds + sqrt(2 rho (1 + rho)) dW,     s = D t,   D = (omega^2/4) S_mu(2 omega),

whose generator is the radial Laplacian of the hyperbolic plane in the distance r defined by
cosh r = 1 + 2 rho. Two exact consequences are used throughout:
    <1 + 2 rho> = exp(2 s)        (mean energy grows at Gamma_2 = 2 D),
    <ln(1 + rho)> = s             (typical energy grows at Gamma_typ = D).
The forward launch has E = 2 omega^2 (1 + 2 rho), so its log-energy approaches its
asymptotic slope only after s ~ 1. The crossover is computed from the closed-form heat
kernel of the hyperbolic plane.

Residual in-band amplification
------------------------------
When S_mu(2 omega) = 0 the leading term vanishes. The substitution A = a exp(i omega Phi/2),
B = b exp(-i omega Phi/2), Phi = int mu dt, removes the diagonal (frequency-modulation)
terms exactly and leaves the coupling c(t) = mu(t) exp[-i(2 omega t + omega Phi)]. At leading
order in c the backscattered intensity is

    <rho(t)> = (omega^2/2) int_0^t (t - tau) Kc(tau) cos(2 omega tau) dtau,
    Kc(tau) = [C(tau) - omega^2 I(tau)^2] exp[-omega^2 V(tau)/2],
    I(tau) = int_0^tau C,     V(tau) = 2 int_0^tau I,

for Gaussian mu with staircase autocorrelation C. Its long-time slope is the rate
Gamma_2 = (omega^2/2) S_c(0), S_c(0) = 2 int_0^inf Kc cos(2 omega tau) dtau. The phase
diffusion of the mode is resummed exactly; the coupling is kept at leading order.

If mu is a quadratic function of a Gaussian variable, mu = -delta + delta^2 + ..., it also
has a third cumulant, and the kernel acquires the term
    i omega T3(tau) exp[-omega^2 V(tau)/2],   T3(tau) = int_0^tau <mu(0) mu(tau) mu(s)>_c ds,
which is of the same order m0^4 as the phase-diffusion residual. It adds
-omega T3 exp(-omega^2 V/2) sin(2 omega tau) to Kc cos(2 omega tau) in both expressions above.
"""
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------------------------
# Brownian motion on the hyperbolic plane


def _log_sinh(x):
    return x + np.log1p(-np.exp(-2.0 * x)) - np.log(2.0)


def radial_density(s, r, n_nodes=160):
    """Density of the hyperbolic distance r at time s, started at r = 0, for the heat
    equation du/ds = Delta u on the plane of curvature -1. Uses the closed-form kernel

        p_s(r) = sqrt(2) exp(-s/4) (4 pi s)^(-3/2) int_r^inf q exp(-q^2/4s) / sqrt(cosh q - cosh r) dq

    with the radial measure 2 pi sinh(r) dr. The substitution q = r + u^2 and
    cosh q - cosh r = 2 sinh(r + u^2/2) sinh(u^2/2) make the integrand smooth.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    out = np.zeros_like(r)
    pos = r > 0
    rp = r[pos][:, None]
    vmax = -rp + np.sqrt(rp * rp + 160.0 * s)        # exp(-40) cut-off of the Gaussian factor
    umax = np.sqrt(vmax)
    u = 0.5 * umax * (x[None, :] + 1.0)
    wu = 0.5 * umax * w[None, :]
    v = u * u
    shape = 2.0 * u / np.sqrt(2.0 * np.sinh(0.5 * v))  # tends to 2 as u -> 0
    log_ratio = _log_sinh(rp) - 0.5 * _log_sinh(rp + 0.5 * v)
    integrand = (rp + v) * np.exp(log_ratio - (2.0 * rp * v + v * v) / (4.0 * s)) * shape
    J = (integrand * wu).sum(axis=1)
    rp = rp[:, 0]
    log_pref = 0.5 * np.log(2.0) - s / 4.0 - 1.5 * np.log(4.0 * np.pi * s) - rp * rp / (4.0 * s)
    out[pos] = 2.0 * np.pi * np.exp(log_pref) * J
    return out


def _simpson(y, x):
    h = x[1] - x[0]
    return h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())


def hyperbolic_moments(s, n_r=1601):
    """Moments of the diffusion at time s. Returns the raw normalization, <ln(1 + rho)>
    (exactly s), <1 + 2 rho> (exactly exp(2 s), resolved for s of order one or less) and
    h(s) = <ln[(1 + 2 rho) / (2 (1 + rho))]>."""
    r = np.linspace(0.0, s + 14.0 * np.sqrt(s) + 1.5 * (1.0 - np.exp(-s)), n_r)
    f = radial_density(s, r)
    Z = _simpson(f, r)
    ln1p = 2.0 * np.log(np.cosh(0.5 * r))                   # ln(1 + rho)
    h = np.log(np.cosh(r)) - np.log(2.0) - ln1p
    return {"norm": Z,
            "ln1p_rho": _simpson(f * ln1p, r) / Z,
            "mean_1p2rho": _simpson(f * np.cosh(r), r) / Z,
            "h": _simpson(f * h, r) / Z}


@lru_cache(maxsize=1)
def _h_table():
    s = np.concatenate([np.geomspace(1e-3, 1.0, 120, endpoint=False),
                        np.linspace(1.0, 60.0, 237)])
    h = np.array([hyperbolic_moments(si)["h"] for si in s])
    return s, h, CubicSpline(np.log(s), h)


def h_forward(s):
    """h(s) = <ln(1 + 2 rho)> - s - ln 2 for the forward launch; h(0) = -ln 2, h -> 0."""
    s = np.asarray(s, dtype=float)
    table_s, _, spline = _h_table()
    out = np.zeros_like(s)
    small = s < table_s[0]
    # <ln(1 + 2 rho)> = 2 s - 2 s^2 + O(s^3) below the tabulated range
    out[small] = -np.log(2.0) + s[small] - 2.0 * s[small] ** 2
    mid = (~small) & (s <= table_s[-1])
    out[mid] = spline(np.log(s[mid]))
    return out


def fit_points(K, window=(0.2, 0.8)):
    """Checkpoint fractions k/K (k = 1..K) retained by a fit window."""
    f = np.arange(1, K + 1) / K
    return f[(f >= window[0] - 1e-9) & (f <= window[1] + 1e-9)]


def g_forward(x, K=2048, window=(0.2, 0.8)):
    """Forward-launch typical slope in units of Gamma_typ, as a function of
    x = Gamma_typ N, on K equally spaced checkpoints and the given fit window."""
    f = fit_points(K, window)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        sk = xi * f
        d = sk - sk.mean()
        out[i] = 1.0 + (d @ h_forward(sk)) / (d @ d)
    return out


@lru_cache(maxsize=16)
def _inversion_table(K, window):
    x = np.geomspace(1e-7, 3e3, 4001)
    y = x * g_forward(x, K, window)
    return np.log(x), np.log(y), np.gradient(y, x)


def gamma2_from_forward(slope, se, N, K=2048, window=(0.2, 0.8)):
    """Mean-energy rate Gamma_2 = 2 Gamma_typ inferred from a forward-launch typical slope
    on a horizon of N slabs by solving slope N = x g(x). Returns (Gamma_2, SE)."""
    slope = np.atleast_1d(np.asarray(slope, dtype=float))
    se = np.broadcast_to(np.asarray(se, dtype=float), slope.shape)
    lx, ly, dydx = _inversion_table(int(K), tuple(window))
    y = slope * N
    G = slope.copy()                     # below the table (and for y <= 0) g = 2, x = y/2
    E = se.copy()
    big = y > np.exp(ly[0])
    if np.any(y > np.exp(ly[-1])):
        raise ValueError("slope beyond the tabulated crossover range")
    x = np.exp(np.interp(np.log(y[big]), ly, lx))
    G[big] = 2.0 * x / N
    E[big] = 2.0 * se[big] / np.interp(np.log(x), lx, dydx)
    return G, E


# ----------------------------------------------------------------------------------------
# Spectra and autocorrelations


def stealthy_autocorrelation(m0, bands, kmax):
    """Discrete autocorrelation C_k (k = 0..kmax) of an infinitely long fixed-variance
    sequence whose spectrum is flat except for the removed bands (on (0, pi])."""
    width = sum(hi - lo for lo, hi in bands)
    k = np.arange(1, kmax + 1, dtype=float)
    C = np.empty(kmax + 1)
    C[0] = 1.0 - width / np.pi
    C[1:] = 0.0
    for lo, hi in bands:
        C[1:] -= (np.sin(hi * k) - np.sin(lo * k)) / (np.pi * k)
    return m0 ** 2 * np.pi / (np.pi - width) * C


def discrete_spectrum(varpi, m0, bands):
    """Two-sided spectral density of the sequence (per slab) at angular frequency varpi."""
    width = sum(hi - lo for lo, hi in bands)
    w = np.abs(np.angle(np.exp(1j * np.asarray(varpi, dtype=float))))
    inside = np.zeros(np.shape(w), dtype=bool)
    for lo, hi in bands:
        inside |= (w >= lo) & (w <= hi)
    return np.where(inside, 0.0, m0 ** 2 * np.pi / (np.pi - width))


def staircase_spectrum(varpi, m0, bands):
    """Spectrum of the continuous-time staircase, S_mu(w) = sinc^2(w/2) S_d(w)."""
    varpi = np.asarray(varpi, dtype=float)
    return np.sinc(varpi / (2.0 * np.pi)) ** 2 * discrete_spectrum(varpi, m0, bands)


# ----------------------------------------------------------------------------------------
# Phase-diffusion estimate of the residual rate


def _cos_moments(n, b):
    """P_p = int_0^1 x^p cos(b (n + x)) dx for p = 0, 1, 2 (exact)."""
    s0, s1 = np.sin(b * n), np.sin(b * (n + 1.0))
    c0, c1 = np.cos(b * n), np.cos(b * (n + 1.0))
    P0 = (s1 - s0) / b
    P1 = s1 / b + (c1 - c0) / b ** 2
    P2 = s1 / b + 2.0 * c1 / b ** 2 - 2.0 * (s1 - s0) / b ** 3
    return P0, P1, P2


def _staircase_t3(C, tau):
    """T3(tau) = int_0^tau <mu(0) mu(tau) mu(s)>_c ds for the staircase, averaged over the
    offset phi of the slab boundaries, with <mu_a mu_b mu_c>_c = 2 (C_ab C_ac + C_ba C_bc +
    C_ca C_cb) for slabs a, b, c and the discrete autocorrelation C of delta.

    With tau = A + f (A integer), the far end lies in slab a = A for phi < 1 - f and in slab
    a = A + 1 otherwise. The parts of the two end slabs that fall outside [0, tau] enter only
    through their total length l = a + 1 - tau, which does not depend on phi, so
        T3 = 2 [2 C_a S1_a - l C_a (C_0 + C_a) + S2_a - l C_0 C_a]
    on each piece, with S1_a = sum_{k<=a} C_k and S2_a = sum_{k<=a} C_k C_{a-k}, and
    T3 = 6 tau C_0^2 while the whole interval lies in one slab.
    """
    S1 = np.cumsum(C)
    S2 = fftconvolve(C, C)[:C.size]
    A = np.floor(tau).astype(int)
    f = tau - A
    A1 = np.minimum(A + 1, C.size - 1)

    def piece(a, lam):
        return 2.0 * (2.0 * C[a] * S1[a] - lam * C[a] * (C[0] + C[a]) + S2[a] - lam * C[0] * C[a])

    near = np.where(A == 0, 6.0 * f * C[0] ** 2, piece(A, 1.0 - f))
    return (1.0 - f) * near + f * piece(A1, 2.0 - f)


class PhaseDiffusion:
    """Residual amplification from the phase-diffusion resummation, for a staircase
    modulation with discrete autocorrelation C_k (k = 0..T).

    The integrals I and V do not depend on omega, so one object serves every frequency.
    The part of the integrand that is first order in the spectrum (C cos) is integrated
    exactly on each slab; the remainder, which is of fourth order in the modulation depth,
    is integrated by the trapezoidal rule on a grid of spacing dtau.

    With `delta_Ck` given, mu = -delta + delta^2 - ... for a Gaussian delta with discrete
    autocorrelation delta_Ck, and the third-cumulant term is included, with
        <mu_a mu_b mu_c>_c = 2 (C_ab C_ac + C_ba C_bc + C_ca C_cb)
    for slabs a, b, c, and T3 averaged over the position of the slab boundaries
    (`_staircase_t3`).

    With `sigma` > 0 the drive is the staircase convolved with a Gaussian of standard
    deviation sigma, whose autocorrelation is the staircase one convolved with a Gaussian of
    standard deviation sigma sqrt(2). The integrand is then smooth and the whole integral is
    evaluated by the trapezoidal rule.
    """

    def __init__(self, Ck, dtau=1.0 / 32.0, delta_Ck=None, sigma=0.0):
        self.Ck = np.asarray(Ck, dtype=float)
        self.T = len(self.Ck) - 1
        m = int(round(1.0 / dtau))
        self.m = m
        self.dtau = 1.0 / m
        n = np.repeat(np.arange(self.T), m)
        f = np.tile(np.arange(m) / m, self.T)
        self.tau = np.append(n + f, float(self.T))
        self.C, self.I = self._interpolate(self.Ck, n, f)
        d = self.dtau
        self.sigma = float(sigma)
        if self.sigma > 0.0:
            if delta_Ck is not None:
                raise ValueError("a smoothed drive with a third cumulant is not implemented")
            full = np.concatenate([self.C[:0:-1], self.C])
            self.C = gaussian_filter1d(full, self.sigma * np.sqrt(2.0) / d, mode="constant",
                                       truncate=10.0)[self.C.size - 1:]
            self.I = np.concatenate([[0.0], np.cumsum(0.5 * (self.C[1:] + self.C[:-1]) * d)])
        self.V = 2.0 * np.concatenate([[0.0], np.cumsum(0.5 * (self.I[1:] + self.I[:-1]) * d)])
        self.T3 = None
        if delta_Ck is not None:
            self.T3 = _staircase_t3(np.asarray(delta_Ck, dtype=float)[:self.T + 1], self.tau)

    def _interpolate(self, Ck, n, f):
        """Piecewise-linear C(tau) of the staircase on the fine grid, and its integral."""
        c0 = Ck[:-1][n]
        C = np.append(c0 + (Ck[1:][n] - c0) * f, Ck[-1])
        I = np.concatenate([[0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * self.dtau)])
        return C, I

    def _remainder(self, omega, j):
        tau = self.tau[:j]
        damp = np.exp(-0.5 * omega ** 2 * self.V[:j])
        first = 0.0 if self.sigma > 0.0 else 1.0        # C cos is integrated exactly elsewhere
        g = (self.C[:j] * (damp - first) - omega ** 2 * self.I[:j] ** 2 * damp) \
            * np.cos(2.0 * omega * tau)
        if self.T3 is not None:
            g -= omega * self.T3[:j] * damp * np.sin(2.0 * omega * tau)
        return g

    def rate(self, omega):
        """Stationary Gamma_2 = (omega^2/2) S_c(0) at mode frequency omega."""
        first = 0.0
        if self.sigma == 0.0:
            n = np.arange(self.T, dtype=float)
            P0, P1, _ = _cos_moments(n, 2.0 * omega)
            c0, c1 = self.Ck[:-1], self.Ck[1:]
            first = float(np.sum(c0 * P0 + (c1 - c0) * P1))
        g = self._remainder(omega, len(self.tau))
        w = np.full(g.size, self.dtau)
        w[0] = w[-1] = 0.5 * self.dtau
        return 0.5 * omega ** 2 * 2.0 * (first + float(w @ g))

    def log_energy(self, omega, t):
        """Predicted <ln E(t)> - ln E(0) = 2 <rho(t)> of the forward launch at integer
        times t (valid while the growth is small)."""
        t = np.asarray(t, dtype=int)
        tmax = int(t.max())
        b = 2.0 * omega
        n = np.arange(tmax, dtype=float)
        P0, P1, P2 = _cos_moments(n, b)
        c0 = self.Ck[:tmax]
        dc = self.Ck[1:tmax + 1] - c0
        on = 0.0 if self.sigma > 0.0 else 1.0
        F0 = on * np.concatenate([[0.0], np.cumsum(c0 * P0 + dc * P1)])
        F1 = on * np.concatenate([[0.0], np.cumsum(c0 * (P1 + n * P0) + dc * (P2 + n * P1))])
        j = tmax * self.m + 1
        g = self._remainder(omega, j)
        tau = self.tau[:j]
        d = self.dtau
        G0 = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * d)])
        tg = tau * g
        G1 = np.concatenate([[0.0], np.cumsum(0.5 * (tg[1:] + tg[:-1]) * d)])
        G0, G1 = G0[t * self.m], G1[t * self.m]
        return omega ** 2 * (t * F0[t] - F1[t] + t * G0 - G1)

    def finite_horizon_slope(self, omega, N, every=2, window=(0.2, 0.8)):
        """Predicted least-squares slope of the forward-launch log-energy on a horizon of N
        slabs, with the same checkpoints and fit window as the simulations."""
        tk = np.rint(fit_points(N // every, window) * N).astype(int)
        y = self.log_energy(omega, tk)
        d = tk - tk.mean()
        return float(d @ y / (d @ d))


def phase_diffusion_bands(m0, bands, omega_min, span=25.0, dtau=1.0 / 32.0, t_min=0, sigma=0.0):
    """PhaseDiffusion for a band-stealthy ensemble (a staircase, or with sigma > 0 the
    staircase smoothed by a Gaussian of that standard deviation). The integration range is
    span/gamma, gamma = omega_min^2 S_mu(0)/2 the slowest phase-diffusion rate, and at least
    t_min."""
    width = sum(hi - lo for lo, hi in bands)
    gamma = 0.5 * omega_min ** 2 * m0 ** 2 * np.pi / (np.pi - width)
    T = max(int(np.ceil(span / gamma)) + 1, int(t_min))
    return PhaseDiffusion(stealthy_autocorrelation(m0, bands, T), dtau, sigma=sigma)


def phase_diffusion_white(m0, omega_min, span=25.0, dtau=1.0 / 32.0):
    """PhaseDiffusion for independent slabs (C_k = m0^2 for k = 0, zero otherwise)."""
    gamma = 0.5 * omega_min ** 2 * m0 ** 2
    Ck = np.zeros(int(np.ceil(span / gamma)) + 2)
    Ck[0] = m0 ** 2
    return PhaseDiffusion(Ck, dtau)


def phase_diffusion_epsilon(m0, bands, omega_min, span=25.0, dtau=1.0 / 32.0, t_min=0):
    """PhaseDiffusion when stealthiness is imposed on delta = eps_r/eps_bar - 1 (Gaussian,
    standard deviation m0) instead of on mu = 1/(1 + delta) - 1 = -delta + delta^2 - ...

    To fourth order in m0 the autocorrelation of mu is (1 + 6 m0^2) C + 2 C^2, C that of
    delta. The term 2 C^2 is the spectrum of delta^2, which refills the removed band with
    the self-convolution of the spectrum of delta; the third cumulant of mu is included
    through `delta_Ck`.
    """
    width = sum(hi - lo for lo, hi in bands)
    gamma = 0.5 * omega_min ** 2 * m0 ** 2 * np.pi / (np.pi - width)
    T = max(int(np.ceil(span / gamma)) + 1, int(t_min))
    C = stealthy_autocorrelation(m0, bands, T)
    return PhaseDiffusion((1.0 + 6.0 * m0 ** 2) * C + 2.0 * C ** 2, dtau, delta_Ck=C)


# ----------------------------------------------------------------------------------------
# Imperfect stealth


def notch_depth_law(depth_db, R_exact, N, n_constrained):
    """Protection factor for a finite notch of the given depth. The leading-order in-band
    rate is the white rate times a^2 kappa_a^2, a = 10^(-depth/20), and adds to the residual
    rate of the exactly stealthy ensemble (protection factor R_exact)."""
    a2 = 10.0 ** (-np.asarray(depth_db, dtype=float) / 10.0)
    k2 = (N - 1.0) / (N - 1.0 - 2.0 * n_constrained * (1.0 - a2))
    return 1.0 / (a2 * k2 + 1.0 / R_exact)


def jitter_leakage_rate(omega, m0, bands, jitter, span=40.0, n_grid=400001):
    """In-band rate created by timing jitter of the slab boundaries.

    With durations 1 + U(-J, J) the boundary times perform a random walk with variance J^2/3
    per slab, so a spectral component at w loses phase coherence at the rate
    gamma_J(w) = w^2 J^2 / 6 and its line becomes a Lorentzian of that half-width. The
    removed band is then refilled by the Lorentzian tails of the neighbouring spectrum:
        Gamma_J = (omega^2/2) int S_mu(w) L(2 omega - w; gamma_J(w)) dw.
    """
    w = np.linspace(-span, span, n_grid)
    dw = w[1] - w[0]
    S = staircase_spectrum(w, m0, bands)
    g = w ** 2 * jitter ** 2 / 6.0
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.empty_like(omega)
    for i, om in enumerate(omega):
        L = g / np.pi / ((2.0 * om - w) ** 2 + g ** 2 + 1e-300)
        out[i] = 0.5 * om ** 2 * np.sum(S * L) * dw
    return out
