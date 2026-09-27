"""Temporal modulation of a lossless dispersive (Lorentz) medium.

The instantaneous model assumes D = eps0 eps_r(t) E. A causal medium responds through its
polarization. Here the polarization is an undamped Lorentz oscillator,

    d2P/dt2 + w_r^2 P = w_p^2(t) E,      D = E + P       (eps0 = mu0 = c = 1),

and for a plane wave of wavenumber k the state z = (D, B, P, Q = dP/dt) obeys

    dD/dt = -i k B,   dB/dt = -i k (D - P),   dP/dt = Q,   dQ/dt = -w_r^2 P + w_p^2 (D - P).

The static permittivity is 1 + w_p^2 / w_r^2. It is modulated as eps(t) = EPS_BAR / (1 + mu(t)),
so that for w_r -> infinity the model reduces to D'' + w0^2 (1 + mu) D = 0 with
k = w0 sqrt(EPS_BAR). The system is Hamiltonian; without modulation the energy
U = (|E|^2 + |B|^2)/2 + (|Q|^2 + w_r^2 |P|^2) / (2 w_p^2) is conserved.

A modulation can now also couple the lower polariton (frequency near w0) to the upper
polariton (near w_r sqrt(EPS_BAR)); the pair-creation resonance lies at the sum of the two
frequencies, far above the removed band. Whether the drive has spectral weight there
depends on how fast it switches, which is why both a band-limited drive and an abrupt
staircase are provided.
"""
import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter1d

EPS_BAR = 2.0


def generators(mu, w0, wr):
    """Generators A (shape (..., 4, 4)) of dz/dt = A z for modulation values mu."""
    mu = np.asarray(mu, dtype=float)
    wp2 = (EPS_BAR / (1.0 + mu) - 1.0) * wr ** 2
    k = w0 * np.sqrt(EPS_BAR)
    A = np.zeros(mu.shape + (4, 4), dtype=complex)
    A[..., 0, 1] = -1j * k
    A[..., 1, 0] = -1j * k
    A[..., 1, 2] = 1j * k
    A[..., 2, 3] = 1.0
    A[..., 3, 0] = wp2
    A[..., 3, 2] = -(wr ** 2 + wp2)
    return A


def lower_polariton(w0, wr):
    """Eigenvector of the unmodulated generator for the forward lower polariton, and its
    frequency."""
    lam, V = np.linalg.eig(generators(0.0, w0, wr))
    j = int(np.argmin(np.abs(lam + 1j * w0)))
    return V[:, j], float(-lam[j].imag)


def upper_polariton_frequency(w0, wr):
    """Frequency of the upper polariton at the same wavenumber."""
    lam = np.linalg.eigvals(generators(0.0, w0, wr))
    return float(np.max(np.abs(lam.imag)))


def energy(z, wr):
    """Energy of field plus oscillator with the static oscillator strength."""
    wp2 = (EPS_BAR - 1.0) * wr ** 2
    D, B, P, Q = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    E = D - P
    return 0.5 * (np.abs(E) ** 2 + np.abs(B) ** 2) + \
        (np.abs(Q) ** 2 + wr ** 2 * np.abs(P) ** 2) / (2.0 * wp2)


def smooth_drive(mu, sigma, sub):
    """Band-limited drive: the slab staircase sampled at `sub` points per slab and convolved
    with a Gaussian of standard deviation sigma (in slab durations). Convolution multiplies
    the spectrum by exp(-sigma^2 w^2/2) in amplitude, so removed bands stay removed."""
    stair = np.repeat(np.atleast_2d(np.asarray(mu, dtype=float)), sub, axis=1)
    return gaussian_filter1d(stair, sigma * sub, axis=1, mode="nearest", truncate=6.0)


def propagate_lorentz(drive, w0, wr, dt, every):
    """Propagate the forward lower polariton through a piecewise-constant drive.

    drive : (R, S) modulation values mu, each held for a step dt (exact exponential per step).
    every : checkpoint stride in steps.

    At every checkpoint the state is also expanded in the eigenmodes of the unmodulated
    generator, which are orthogonal in the energy metric, so the energy splits exactly into a
    lower-polariton part and an upper-polariton part.

    Returns (t, ln[U/U0], ln[U_lower/U0], U_upper/U0), the last three of shape (R, K).
    """
    drive = np.atleast_2d(drive)
    R, S = drive.shape
    lam, V = np.linalg.eig(generators(0.0, w0, wr))
    Vi = np.linalg.inv(V)
    lower = np.abs(lam.imag) < 0.5 * np.abs(lam.imag).max()
    v, _ = lower_polariton(w0, wr)
    z = np.tile(v, (R, 1))
    U0 = energy(z, wr)
    K = S // every
    lnU = np.empty((R, K))
    lnU_low = np.empty((R, K))
    U_up = np.empty((R, K))
    k = 0
    for n in range(S):
        z = np.einsum("rij,rj->ri", expm(generators(drive[:, n], w0, wr) * dt), z)
        if (n + 1) % every == 0 and k < K:
            c = z @ Vi.T
            lnU[:, k] = np.log(energy(z, wr) / U0)
            lnU_low[:, k] = np.log(energy(c[:, lower] @ V[:, lower].T, wr) / U0)
            U_up[:, k] = energy(c[:, ~lower] @ V[:, ~lower].T, wr) / U0
            k += 1
    return np.arange(1, K + 1) * every * dt, lnU, lnU_low, U_up
