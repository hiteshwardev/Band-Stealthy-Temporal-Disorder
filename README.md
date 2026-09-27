# Band-stealthy temporal disorder

Code, data and the executed study notebook for the manuscript *Band-stealthy temporal
disorder suppresses disorder-induced parametric amplification in time-varying media* by
Hitesh Kumar Singh.

A dielectric whose permittivity fluctuates randomly in time amplifies waves parametrically.
To leading order a mode of frequency $\omega$ grows at a rate set by the modulation spectrum
at $2\omega$. Removing a band $\Omega$ from that spectrum at fixed variance, which we call
band-stealthy temporal disorder, removes the leading-order growth of every mode with
$2\omega$ inside $\Omega$. The study measures the residual growth by exact propagation
through temporal slabs, explains it with a phase-diffusion theory that has no free
parameter, quantifies the cost for the unprotected modes, and tests the construction against
the imperfections of a real modulator.

## Main results

Values from `results/summary.json`, for slabs of unit duration, modulation depth
$m_0=0.05$ and sequences of 4096 slabs unless stated otherwise.

| Quantity | Value |
| --- | --- |
| Median protection factor over the protected window, $R=\Gamma_{2,W}/\hat\Gamma_{2,S}$ | 526 ± 11 |
| Phase-diffusion prediction for the same median | 517 |
| Depth exponent of the residual inside the window (prediction) | 3.95 ± 0.02 (4.01) |
| Depth exponent outside the window | 2.00 ± 0.02 |
| Out-of-window rate relative to the exact white rate, measured and $\kappa^2$ | 1.2311 ± 0.0035 and 1.2368 |
| Same number of removed components at zero frequency (hyperuniform): $R$ in the window | 0.83 |
| Usable fraction of the window ($R\ge10$) at $m_0=0.05$ and 0.20 | 0.99 and 0.86 |
| Stealthiness imposed on the permittivity instead of its inverse: $R$ (prediction) | 176 (172) |
| Notch of 40 dB instead of an exact zero: $R$ | 484 |
| Timing jitter of 2% of the slab duration: $R$ | 464 |
| Lorentz medium ($w_r/\omega=40$), band-limited drive: $R$ of the lower polariton | > 680 |

## Repository layout

```
band_stealthy_temporal_disorder.ipynb   the complete study: theory, validation, simulations,
                                        figures and the table of final numbers
src/
    model.py          exact slab propagator, growth-rate estimators, exact reference rates
    ensembles.py      modulation sequences and the seeding of every ensemble
    simulate.py       ensemble runs and their reduction to growth rates
    theory.py         heat-kernel crossover, phase-diffusion theory, imperfection models
    dispersive.py     Lorentz medium and the split of its energy into polariton branches
    figures.py        figure style and the layout check applied to every figure
    records.py        result files that do not change between runs
tests/test_core.py    tests of the modules
figures/              every figure of the manuscript, PNG at 600 dpi and PDF
results/              intermediate data (.npz) and all quoted numbers (summary.json)
requirements.txt      pinned package versions (environment.yml for conda)
```

## Reproducing the study

With Python 3.11:

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                  # 22 tests, about ten seconds
jupyter execute --inplace band_stealthy_temporal_disorder.ipynb
```

With conda, `conda env create -f environment.yml` followed by
`conda activate band-stealthy-temporal-disorder` replaces the first three lines. The
notebook can also be run cell by cell in JupyterLab (`pip install jupyterlab`). A complete
run takes about twenty minutes on two processor cores and needs about 2 GB of memory.
It rewrites every file in `results/` and `figures/` and reads nothing that it has not written
itself.

## Reproducibility

Every random ensemble is drawn from a generator seeded with the SHA-256 digest of
descriptive tags (`src/ensembles.py`), so an ensemble is fixed by its name and does not
depend on the order of the computation. Result files carry no timestamps
(`src/records.py`). Two complete runs, one of them in a fresh environment built from
`requirements.txt`, produced byte-identical files in `results/` and `figures/`. Other
versions of NumPy, SciPy or the underlying linear-algebra library can change the last digits.

## Changes with respect to earlier versions

This repository replaces the earlier versions of the project
(`Temporal-Stealthy-Hyperuniformity` and `TSH-V2`). The following statements changed.

1. **Name.** The construction was called temporal stealthy hyperuniformity. The sequences
   are not hyperuniform: only the zero-frequency component and a band at finite frequency
   are removed, and the spectrum near zero frequency is raised by $\kappa^2$. A control with
   the same number of removed components at the origin, which is hyperuniform, gives no
   protection in the window (notebook Section 12).
2. **Headline number.** Earlier versions quoted a suppression of about $3\times10^2$ from
   the ratio of typical growth slopes of the two ensembles at the same horizon. At a given
   horizon the white ensemble has grown much more than the band-stealthy one, and the
   finite-horizon slope of a forward-launched wave falls from twice the typical rate towards
   the typical rate as the growth accumulates, so that ratio understates the protection. The
   protection factor is now a ratio of mean-energy rates: the exact white rate over the
   band-stealthy rate obtained by inverting the finite-horizon crossover computed from the
   heat kernel of the hyperbolic plane. It is the median over twelve pooled seed families,
   with a bootstrap uncertainty over families.
3. **Lyapunov estimator.** The QR estimator accumulated norms in the variables
   $(D,\dot D)$, in which free evolution is not a rotation, and returned nonzero exponents
   for a constant medium; earlier versions calibrated around this defect. The estimator now
   works in $(\omega D,\dot D)$ and returns zero to rounding for a constant medium, which a
   test enforces.
4. **Dispersive medium.** Earlier statements about a Lorentz medium are superseded. The
   energy is now split exactly into lower- and upper-polariton parts. With a drive that is
   band limited on the time scale of the material resonance, the lower polariton, the
   protected mode, keeps the protection of the instantaneous model. Abrupt switching
   transfers energy to the upper polariton, for white and band-stealthy modulation alike.
5. **Uncertainty of the headline.** Earlier versions quoted the headline from a single seed
   family, without an uncertainty derived from the scatter between families or from the
   dependence on analysis choices. The statistical uncertainty now comes from resampling
   twelve families. Alternative fit windows and horizons are treated as consistency checks
   with statistical errors for their differences; none differs significantly from the
   headline, so no systematic uncertainty is added. Other summary statistics of
   $R(\omega)$ are reported separately because they are not estimates of the median.
6. **Stealthiness imposed on the permittivity.** The prediction now includes the third
   cumulant of the modulation, which sets its frequency dependence.

## License

MIT, see `LICENSE`.
