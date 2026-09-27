"""Checks of the reusable modules. Run with `python -m pytest -q` from the repository root."""
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import dispersive, model, records, theory  # noqa: E402
from src import ensembles as ens  # noqa: E402
from src import figures as fg  # noqa: E402

BAND = (1.0, 1.6)


def test_rng_is_fixed_by_tags():
    a = ens.rng("x", 1, 0.05).standard_normal(4)
    b = ens.rng("x", 1, 0.05).standard_normal(4)
    c = ens.rng("x", 2, 0.05).standard_normal(4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_slab_matrix_has_unit_determinant():
    for w2 in (0.3, 1.7, -0.4, 0.0):
        assert abs(np.linalg.det(model.slab_matrix(w2)) - 1.0) < 1e-13


def test_propagation_equals_matrix_product():
    g = ens.rng("test", "product")
    mu = 0.2 * g.standard_normal((1, 64))
    om = 0.7
    _, lnEf, lnEs = model.propagate(om, mu, every=64)
    P = np.eye(2)
    for m in mu[0]:
        P = model.slab_matrix(om ** 2 * (1 + m)) @ P
    zf = P @ np.array([1.0, -1j * om])
    zs = P @ np.array([1.0, 0.0])
    assert abs(lnEf[0, -1] - np.log((om ** 2 * abs(zf[0]) ** 2 + abs(zf[1]) ** 2) / (2 * om ** 2))) < 1e-10
    assert abs(lnEs[0, -1] - np.log((om ** 2 * zs[0] ** 2 + zs[1] ** 2) / om ** 2)) < 1e-10


def test_ols_slopes_recovers_a_line():
    t = np.arange(1, 101, dtype=float)
    y = np.vstack([3e-4 * t + 0.1, -2e-5 * t])
    assert np.allclose(model.ols_slopes(t, y, (0.2, 0.8)), [3e-4, -2e-5], rtol=1e-12, atol=1e-18)


def test_lyapunov_estimator_is_zero_on_a_constant_medium():
    for om in (0.5, 0.65, 1.3):
        assert abs(model.lyapunov_rate(om, np.zeros((1, 20000)))[0]) < 1e-14


def test_lyapunov_estimator_on_the_two_slab_crystal():
    m0 = 0.1
    exact = model.two_slab_rate(np.pi / 2, m0)
    est = model.lyapunov_rate(np.pi / 2, np.tile([m0, -m0], 8192)[None, :])[0]
    assert exact > 0 and abs(est / exact - 1) < 1e-3


def test_stealthy_sequences_are_exact():
    N = 4096
    bins = ens.band_bins(N, BAND)
    x = ens.stealthy(ens.rng("test", "stealth"), 16, N, 0.05, bins)
    assert len(bins) == 392
    assert ens.inband_fraction(x, bins).max() < 1e-28
    assert np.allclose(x.std(axis=1), 0.05, rtol=1e-12)
    assert np.abs(x.mean(axis=1)).max() < 1e-15


def test_kappa2():
    assert abs(ens.kappa2(4096, 392) - 4095 / 3311) < 1e-15


def test_exact_rate_reduces_to_born_at_small_depth():
    for om in (0.4, 0.65, 1.2):
        born = float(model.born_rate_iid(om, 0.005))
        assert abs(model.kronecker_rate(om, 0.005) / born - 1) < 1e-4


def test_heat_kernel_identities():
    for s in (0.01, 1.0, 10.0):
        m = theory.hyperbolic_moments(s)
        assert abs(m["norm"] - 1) < 1e-8
        assert abs(m["ln1p_rho"] - s) < 1e-8
    assert abs(theory.hyperbolic_moments(0.5)["mean_1p2rho"] * np.exp(-1.0) - 1) < 1e-6


def test_crossover_limits_and_inversion():
    assert abs(theory.g_forward(1e-4, K=2048)[0] - 2.0) < 1e-3
    assert abs(theory.g_forward(200.0, K=2048)[0] - 1.0) < 1e-6
    N = 4096
    for x in (1e-3, 0.3, 2.0, 30.0):
        slope = x * theory.g_forward(x, K=N // 2)[0] / N
        G, _ = theory.gamma2_from_forward(slope, 0.0, N, K=N // 2)
        assert abs(G[0] / (2 * x / N) - 1) < 1e-6


def test_phase_diffusion_reproduces_born_outside_the_band():
    pdo = theory.phase_diffusion_bands(0.05, [BAND], 0.5)
    for om in (0.3, 1.0, 1.3):
        born = 0.5 * om ** 2 * theory.staircase_spectrum(2 * om, 0.05, [BAND])
        assert abs(pdo.rate(om) / born - 1) < 5e-3


def test_smoothed_drive_theory_reproduces_born_outside_the_band():
    sigma = 0.15
    pdo = theory.phase_diffusion_bands(0.05, [BAND], 0.5, sigma=sigma)
    for om in (0.3, 1.0, 1.3):
        born = 0.5 * om ** 2 * theory.staircase_spectrum(2 * om, 0.05, [BAND]) * np.exp(-(2 * sigma * om) ** 2)
        assert abs(pdo.rate(om) / born - 1) < 5e-3


def test_phase_diffusion_residual_is_fourth_order():
    r1 = theory.phase_diffusion_bands(0.05, [BAND], 0.6).rate(0.65)
    r2 = theory.phase_diffusion_bands(0.10, [BAND], 0.6).rate(0.65)
    assert r1 > 0 and 12 < r2 / r1 < 20


def test_staircase_third_cumulant_against_a_direct_average():
    # average of int_0^tau kappa3(0, s, tau) ds over the position of the slab boundaries
    C = theory.stealthy_autocorrelation(0.05, [BAND], 40)
    tau = np.linspace(0.0, 6.0, 97)
    s = np.linspace(0.0, 6.0, 6 * 1024 + 1)
    direct = np.zeros_like(tau)
    phases = (np.arange(256) + 0.5) / 256          # cell edges aligned with the tau grid
    for phi in phases:
        ns = np.floor(s + phi).astype(int)
        for j, t in enumerate(tau):
            sel = s <= t + 1e-12
            nt = int(np.floor(t + phi))
            c0s, cts, ct0 = C[ns[sel]], C[np.abs(nt - ns[sel])], C[nt]
            g = 2.0 * (ct0 * c0s + ct0 * cts + c0s * cts)
            direct[j] += np.trapezoid(g, s[sel]) / phases.size if sel.sum() > 1 else 0.0
    assert np.abs(theory._staircase_t3(C, tau) - direct).max() < 1e-10 * np.abs(direct).max()


def test_third_cumulant_term_for_stealthy_permittivity():
    # a stealthy delta gives mu = -delta + delta^2 - ..., whose third cumulant changes the
    # residual rate at order m0^4; the Gaussian kernel alone misses the simulated rate
    N, M, m0, om = 4096, 400, 0.05, 0.55
    mu = ens.epsilon_stealthy(ens.rng("test", "epsilon"), M, N, m0, ens.band_bins(N, BAND))
    t, lf, _ = model.propagate(om, mu, every=2)
    s = model.ols_slopes(t, lf, (0.2, 0.8))
    g2, se = theory.gamma2_from_forward(s.mean(), s.std(ddof=1) / np.sqrt(M), N, K=N // 2)
    full = theory.phase_diffusion_epsilon(m0, [BAND], om, t_min=N)
    gauss = theory.PhaseDiffusion(full.Ck, full.dtau)
    assert abs(g2[0] - full.rate(om)) < 3 * se[0]
    assert gauss.rate(om) - g2[0] > 3.5 * se[0]


def test_smooth_drive_is_continuous_and_keeps_the_band_empty():
    N, sub = 256, 32
    bins = ens.band_bins(N, BAND)
    mu = ens.stealthy(ens.rng("test", "drive"), 4, N, 0.05, bins)
    drive = dispersive.smooth_drive(mu, 0.15, sub)
    assert np.abs(np.diff(drive, axis=1)).max() < 0.1 * np.abs(np.diff(mu, axis=1)).max()
    F = np.abs(np.fft.rfft(drive - drive.mean(axis=1, keepdims=True), axis=1)) ** 2
    assert (F[:, bins].sum(axis=1) / F[:, 1:].sum(axis=1)).max() < 1e-6


def test_lorentz_medium_conserves_energy_without_modulation():
    _, lnU, lnU_low, U_up = dispersive.propagate_lorentz(np.zeros((1, 800)), 0.65, 20 * 0.65, 0.25, 1)
    assert np.abs(lnU).max() < 1e-11 and np.abs(lnU_low).max() < 1e-11 and U_up.max() < 1e-20


def test_lorentz_medium_approaches_the_instantaneous_limit():
    # the two energy measures differ by a bounded factor 1 + O(m0), so growth rates are compared
    N, sub = 128, 16
    mu = ens.white(ens.rng("test", "lorentz"), 16, N, 0.1)
    drive = dispersive.smooth_drive(mu, 0.15, sub)
    t_i, lf_i, _ = model.propagate(0.65, drive, tau=1 / sub, every=sub)
    t_d, l_d, _, _ = dispersive.propagate_lorentz(drive, 0.65, 80 * 0.65, 1 / sub, sub)
    s_i = model.ols_slopes(t_i, lf_i, (0.0, 1.0)).mean()
    s_d = model.ols_slopes(t_d, l_d, (0.0, 1.0)).mean()
    assert abs(s_d / s_i - 1) < 0.01


def test_layout_audit():
    fg.setup()
    fig, ax = fg.new()
    ax.plot([0, 1], [0, 1], label="line")
    fg.legend(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    assert fg.audit(fig) == []
    ax.text(0.5, 0.5, "inside", transform=ax.transAxes)
    assert any("intrudes" in p for p in fg.audit(fig))
    with pytest.raises(AssertionError):
        fg.save(fig, "never_written", outdir="/nonexistent")
    plt.close(fig)


def test_layout_audit_requires_labeled_ticks():
    # a log axis spanning less than a decade has no labeled decade ticks
    fg.setup()
    fig, ax = fg.new()
    ax.plot([0, 1], [200, 700], label="line")
    ax.set_yscale("log")
    fg.log_ticks(ax)
    ax.set_ylim(150, 800)
    fg.legend(fig)
    assert any("fewer than two labeled ticks" in p for p in fg.audit(fig))
    fg.fixed_ticks(ax, "y", [200, 300, 500, 700])
    assert fg.audit(fig) == []
    plt.close(fig)


def test_result_files_do_not_depend_on_the_time_of_writing(tmp_path):
    arrays = {"a": np.arange(4.0), "b": np.eye(2)}
    records.save_npz(tmp_path / "one.npz", **arrays)
    with zipfile.ZipFile(tmp_path / "one.npz") as archive:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    loaded = np.load(tmp_path / "one.npz")
    assert all(np.array_equal(loaded[k], v) for k, v in arrays.items())
