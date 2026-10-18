# Band-Stealthy Temporal Disorder Suppresses Parametric Amplification in Time-Varying Media

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Abstract

Random temporal modulation can parametrically amplify waves, with the leading-order growth of a mode determined by the modulation spectrum at twice its frequency. This work introduces **band-stealthy temporal disorder**, in which a finite frequency band is removed from the modulation spectrum at fixed variance, suppressing the leading-order amplification of every mode whose parametric resonance lies in the removed band. Exact propagation through sequences of temporal slabs, together with a parameter-free correction for finite observation horizons based on the heat kernel of the hyperbolic plane, gives a median reduction in the mean-energy growth rate by a factor of 526 ± 11 at a modulation depth of 0.05. The residual in-band growth scales approximately as the fourth power of the modulation depth and is described by a phase-diffusion theory without adjustable parameters. At fixed variance, spectral weight removed from the protected band is redistributed to unprotected frequencies. Comparisons with hyperuniform and shifted-band controls show that protection depends on the placement of the spectral notch relative to the parametric resonance, not on the amount of imposed correlation. Constraints on notch depth, timing jitter, the modulation variable and drive smoothness in dispersive media are quantified.

## Core Objectives

- Determine whether engineering the **spectrum** of temporal disorder, rather than only its strength, can suppress disorder-driven parametric amplification in time-varying media.
- Identify which spectral construction protects a mode, and distinguish it from hyperuniform disorder.
- Quantify the price (redistribution to unprotected modes), the residual (higher-order growth) and the realistic limits (notch depth, timing jitter, modulation variable, dispersion).

## Scientific Significance

Temporal modulation underlies photonic time crystals, temporal interfaces and amplified emission, and every real modulator is noisy. Random temporal modulation is a source of gain, so devices built on temporal modulation must contend with it. In space, correlated and stealthy disorder is an established design tool. This project shows that the temporal analogue is not hyperuniformity: the constraint must be placed at **twice the signal frequency**, away from the origin. It gives a quantitative spectral-design rule for noise tolerance in time-varying photonic and wave systems.

## Methodological Approach

- **Exact slab propagation:** sequences of temporal slabs are propagated with exact SL(2,ℝ) transfer matrices, with no time stepping.
- **Ensembles:** white, band-stealthy and shuffled (same values, correlations destroyed) sequences at six modulation depths, 53 mode frequencies, 400 realizations of 4096 slabs per point, with additional horizons and twelve independent families for the headline estimate.
- **Finite-horizon bias removal:** a closed-form crossover function from the hyperbolic-plane heat kernel and a standing-wave identity convert finite-horizon slopes to mean-energy rates.
- **Theory:** a phase-diffusion resummation (Eq. 16 of the manuscript) with no fitted constants, compared against simulation across depths, band edges and imperfections.
- **Validation:** propagator against eighth-order Runge-Kutta, Lyapunov estimator against exact Floquet rates, and Monte Carlo against the exact white-noise rate.

## Contributions

1. A band-stealthy construction for temporal disorder and the demonstration that protection is set by band **placement relative to the parametric resonance**.
2. A phase-diffusion theory that predicts the residual amplification, its rise toward band edges and the loss of protection near edges, with no adjustable parameters.
3. A closed-form correction for finite-horizon bias in typical growth rates, based on the hyperbolic-plane heat kernel.
4. A realizability analysis: finite notch depth, timing jitter, non-Gaussian slab values, band-limited drive, the choice of engineered variable and a lossless Lorentz medium.

5. **Positioning relative to prior work.** The project builds on engineered temporal disorder via a structure factor (Kim et al., *Nat. Phys.* 19, 726, 2023) and on stealthy hyperuniform media in space.

## Key Findings

- Median protection factor **526 ± 11** at 5% modulation depth (range 163–709 over the window interior); it scales as m₀⁻² (fitted exponent −2.02 ± 0.03).
- The in-band residual scales as m₀⁴ (fitted 3.95 ± 0.02); the theory predicts a median of 517, within 2% of the measurement.
- Unprotected modes pay a fixed redistribution factor κ² = 1.2368 (N = 4096); the measured value is 1.2311 ± 0.0035.
- Hyperuniform and shifted-band controls show no protection in the primary window (R ≈ 0.8) and protect only their own bands.
- A 40 dB notch recovers most of the ideal protection (R = 484); 2% timing jitter lowers the median protection by 12% and 5% jitter by 55%.
- Stealthiness should be imposed on the inverse permittivity μ; imposing it on the permittivity modulation instead lowers the median protection to 176.
- In a dispersive Lorentz medium the drive must be smooth on the time scale of the material resonance.

## Scope and Limitations

All results are computational; no experiment was performed. The model is scalar, one-dimensional in wavevector, lossless and spatially uniform. The phase-diffusion theory is not a complete fourth-order theory (it reproduces 66–92% of the exact fourth-order correction for independent slabs). The dispersive study uses a single undamped Lorentz resonance.

## Roles

Hitesh Kumar Singh is the sole author. Hitesh Kumar Singh conceived the study, developed the theory, wrote the code, performed the simulations and analysis, prepared the figures and wrote the manuscript. This included designing the benchmark and validation suite and the reproducible, seeded computational workflow.

## Reproducibility and Availability

The repository contains the code and the executed notebook that generate every figure and number in the manuscript. Random streams are seeded from descriptive tags so that every ensemble is fixed by its name. The code is released under the MIT License.
