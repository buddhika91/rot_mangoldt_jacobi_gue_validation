# ROT Mangoldt Jacobi GUE Validation

This repository contains a reproducible numerical experiment for a finite self-adjoint Jacobi operator generated from Mangoldt arithmetic data. The purpose is to test whether the spectrum of this operator exhibits GUE-type statistics similar to those observed in the high Riemann zeros.

The main validated case is:

```math
N=8000,\qquad p_{\max}=4000,\qquad s_0=0.60,\qquad \sigma=-1.0,\qquad \phi(x)=\cos(4\pi x).
```

Using the exact discovery-code path, this case passes the GUE spacing, adjacent-gap-ratio, and mesoscopic number-variance gates, and beats 200 deterministic matched controls.

This is a numerical spectral experiment. It is not a proof of the Riemann Hypothesis.

---

## 1. Main claim tested

The experiment tests the following finite-dimensional statement:

> A finite self-adjoint Jacobi operator whose coefficients are generated from Mangoldt arithmetic data can exhibit GUE-type spectral statistics, and at the validated `N=8000` benchmark it outperforms matched randomized controls.

The operator is finite, real symmetric, and tridiagonal. Therefore it is self-adjoint at the finite matrix level.

---

## 2. Operator definition

For matrix size `N`, the code constructs a Jacobi matrix

```math
J_N=
\begin{pmatrix}
a_1 & b_1 & 0 & \cdots \\
b_1 & a_2 & b_2 & \cdots \\
0 & b_2 & a_3 & \cdots \\
\cdots & \cdots & \cdots & \ddots
\end{pmatrix}.
```

The entries have the schematic form

```math
a_n
=
\rho\log(n+1)
+
\frac{K_0+h_2}{n^2}
+
\frac{h_3}{n^3}
+
s_nD_n,
```

and

```math
b_n
=
k\sqrt{\frac{n}{n+1}}\exp(s_{n+1/2}O_n).
```

Here `D_n` and `O_n` are arithmetic modulation channels generated from a complex Mangoldt phase field. The exact implementation is in:

```text
rot_mangoldt_jacobi_gue_validation_v3_exact.py
```

and uses the same discovery-code path used to obtain the benchmark result.

---

## 3. Arithmetic input

The arithmetic input is the Mangoldt function:

```math
\Lambda(n)=
\begin{cases}
\log p, & n=p^k \text{ for a prime }p\text{ and }k\ge 1,\\
0, & \text{otherwise.}
\end{cases}
```

The complex arithmetic phase field is built from weights proportional to

```math
w_m\propto \frac{\Lambda(m)}{\sqrt{m}}.
```

The phase field has the schematic form

```math
Z_n
=
\sum_{m\le p_{\max}}
w_m
\exp\left(i\left(\sqrt{\pi}\,n^{1/3}\log m+\frac{\pi}{2}\right)\right).
```

From `Z_n`, the code derives local amplitude, phase, coherence, curvature, and recursive-memory features. These features modulate the diagonal and off-diagonal entries of `J_N`.

The randomized controls replace the Mangoldt weights with matched alternatives:

- Gaussian weights
- Permuted Mangoldt weights
- Sign-flipped Mangoldt weights
- Random-support weights

The goal is to test whether the Mangoldt arithmetic input performs better than structurally similar randomized inputs.

---

## 4. Xi-action density correction

The density correction uses a curvature invariant extracted from the completed Riemann Xi function:

```math
K_0\approx 0.04620998623306458.
```

The associated Xi-action scale is

```math
\Lambda_\Xi \ell_P^2
=
\frac{K_0}{4\pi^2}
\exp\left(-\frac{4\pi}{K_0}\right).
```

Equivalently,

```math
S_{\rm rec}
=
-\log(\Lambda_\Xi \ell_P^2)
=
\frac{4\pi}{K_0}
+
\log\left(\frac{4\pi^2}{K_0}\right).
```

This can also be written as

```math
S_{\rm rec}=2I_R+\log(2\pi I_R),
\qquad
I_R=\frac{2\pi}{K_0}.
```

The finite-size density profile is

```math
s_n=s_0+\epsilon_N\phi(n/N),
```

with

```math
\epsilon_N
=
\sigma\sqrt{\frac{\log N}{S_{\rm rec}}}.
```

For the validated `N=8000` case,

```math
\phi(x)=\cos(4\pi x),
\qquad
s_0=0.60,
\qquad
\sigma=-1.0.
```

---

## 5. What is GUE?

GUE stands for Gaussian Unitary Ensemble. It is a random-matrix ensemble whose local eigenvalue statistics are known to match the observed high-zero statistics of the Riemann zeta function after unfolding.

The experiment does not attempt to match individual Riemann zeros. Instead, it tests whether the unfolded eigenvalues of `J_N` have GUE-type statistics.

The tested statistics are:

1. Nearest-neighbor spacing distribution
2. Adjacent gap-ratio statistic
3. Mesoscopic number variance

The adjacent gap-ratio target is approximately

```math
r_{\rm GUE}\approx 0.5996.
```

The hard-gate criteria used in the exact benchmark are:

```math
KS_{\rm GUE}<0.13,
```

```math
0.58<r<0.62,
```

and

```math
NV_{\rm GUE}<NV_{\rm Poisson}.
```

---

## 6. Validated `N=8000` result

The exact validation command is:

```bash
python rot_mangoldt_jacobi_gue_validation_v3_exact.py --N 8000 --prime-max 4000 --s0 0.60 --sigma -1.0 --shape cos4 --controls 50 --out-dir rot_gue_N8000_exact_validation
```

The validated result is:

```text
RUN N=8000 prime_max=4000 s0=0.6 sigma=-1.0 shape=cos4 controls=200 seed=6783 L_values=1:12:1 nv_windows=250 score_metric=gate
XI_ACTION K0=0.046209986233064583 S_rec=2.786908968682551e+02 logLambda=-2.786908968682551e+02 eps=-1.795770094514358e-01
MANGOLDT score=1.118734526821640e-01 KS=1.092593767024393e-01 r=5.969859240202753e-01 NVg=3.067249938136101e+00 NVp=3.878340788981460e+00 NVpass=1 hard=1
CONTROLS verdict=PASS_STRICT count=200 ctrl_min=0.11207140297582752 ctrl_med=0.5392027486416668 margin=0.0001979502936634847 min_by=gaussian#27 better=0 p_strict=0.004975124378109453
```

In table form:

| Quantity | Value |
|---|---:|
| `N` | 8000 |
| `p_max` | 4000 |
| `s0` | 0.60 |
| `sigma` | -1.0 |
| `shape` | `cos4` |
| `KS_GUE` | 0.1092593767 |
| `r_mean` | 0.5969859240 |
| `NV_GUE` | 3.0672499381 |
| `NV_Poisson` | 3.8783407890 |
| Hard gate | Pass |
| Controls tested | 200 |
| Controls beating Mangoldt | 0 |
| Closest control | `gaussian#27` |
| Margin over closest control | `+1.9795e-4` |
| Verdict | `PASS_STRICT` |

The margin is small but positive. Therefore this result should be interpreted as a reproducible finite benchmark pass, not as a theorem.

---

## 7. Reproducibility

### Install requirements

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.10+
```

Required packages:

```text
numpy
scipy
pandas
matplotlib
tqdm
```

### Quick smoke test

```bash
python rot_mangoldt_jacobi_gue_validation_v3_exact.py --N 1000 --prime-max 500 --controls 2 --out-dir smoke_test
```

### Exact `N=8000` check without controls

```bash
python rot_mangoldt_jacobi_gue_validation_v3_exact.py --N 8000 --prime-max 4000 --s0 0.60 --sigma -1.0 --shape cos4 --controls 0 --out-dir rot_gue_N8000_exact_check
```

Expected Mangoldt-only output:

```text
MANGOLDT score=1.118734526821640e-01 KS=1.092593767024393e-01 r=5.969859240202753e-01 NVg=3.067249938136101e+00 NVp=3.878340788981460e+00 NVpass=1 hard=1
```

### Full `N=8000` control validation

```bash
python rot_mangoldt_jacobi_gue_validation_v3_exact.py --N 8000 --prime-max 4000 --s0 0.60 --sigma -1.0 --shape cos4 --controls 50 --out-dir rot_gue_N8000_exact_validation
```

This runs 50 controls per mode across four modes, for 200 total controls.

---

## 8. Output files

The validation script writes a verification directory containing data, plots, and metadata. Typical outputs include:

```text
README.md
metadata.json
summary.json
data/all_trials.csv
data/mangoldt_vs_controls.csv
data/control_mode_summary.csv
data/operator_coefficients_mangoldt.csv
data/spectrum_mangoldt.csv
data/spacings_mangoldt.csv
data/number_variance_mangoldt.csv
plots/spacing_histogram_mangoldt.png
plots/spacing_cdf_mangoldt.png
plots/number_variance_mangoldt.png
plots/gap_ratio_histogram_mangoldt.png
plots/control_score_by_mode.png
plots/operator_coefficients_mangoldt.png
plots/density_profile_mangoldt.png
```

The terminal also prints a copy-paste block:

```text
COPY_PASTE_TERMINAL_REPORT_BEGIN
...
COPY_PASTE_TERMINAL_REPORT_END
```

This block is intended for quick independent verification and reporting.

---

## 9. Important caveats

This project does not claim to prove the Riemann Hypothesis.

The current result is a finite numerical benchmark:

```math
J_N\quad \text{at}\quad N=8000.
```

The missing analytic steps include:

1. Defining and proving convergence of an infinite limiting operator.
2. Deriving the finite-size scale parameter `s0(N)` analytically.
3. Deriving the density law

```math
\epsilon_N=\sigma\sqrt{\frac{\log N}{S_{\rm rec}}}
```

rather than treating it as an experimentally motivated renormalization.

4. Proving that the limiting spectral measure is connected to the completed zeta function or its logarithmic derivative.
5. Proving that the observed GUE statistics arise from arithmetic structure rather than finite-matrix flexibility.

There is also a deeper number-variance stress test over larger `L` windows that is stricter than the mesoscopic benchmark used here. The `N=8000` result reported above is a pass for the exact mesoscopic discovery benchmark:

```text
L_values = 1:12:1
nv_windows = 250
```

---

## 10. Interpretation

The validated `N=8000` result supports the following cautious statement:

> A finite self-adjoint Jacobi operator generated from Mangoldt arithmetic data exhibits GUE-type spacing, adjacent-gap-ratio, and mesoscopic number-variance behavior at `N=8000`, and beats 200 deterministic matched controls under the same benchmark.

This is evidence for an arithmetic spectral structure worth investigating. It is not yet an asymptotic theorem and not an RH proof.

---

## 11. Suggested citation of the numerical result

A concise way to cite the core result is:

> For the exact discovery benchmark at `N=8000` and `p_max=4000`, the Mangoldt Jacobi operator gives `KS_GUE=0.109259`, adjacent gap ratio `r=0.596986`, and mesoscopic number variance `NV_GUE=3.06725` versus `NV_Poisson=3.87834`. Against 200 deterministic matched controls, no control beats the Mangoldt score; the closest is `gaussian#27` with margin `+1.98e-4`. Thus the `N=8000` result is a strict pass under this finite benchmark, though not an asymptotic theorem.

---

## 12. License and status

This repository is intended for independent verification of the numerical experiment. Please treat the result as experimental mathematics.

Recommended status label:

```text
Experimental / numerical spectral model; not a proof of RH.
```
