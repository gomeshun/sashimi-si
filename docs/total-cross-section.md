# Rutherford total cross section

The user adopted this isolated correction on 2026-09-10 after an independent
review. The public differential API already implements Rutherford scattering:

`d(sigma/m)/dmu = (sigma0/m) / (2 * [1 + (v/w)^2(1-mu)/2]^2)`.

Integrating over `mu=cos(theta)` from -1 to 1 gives
`sigma_total/m = (sigma0/m) / (1 + (v/w)^2)`. No further azimuth factor belongs
in that integral. This agrees with the text after Eq. (2.1) in
[Yang & Yu (2022)](https://arxiv.org/pdf/2205.03392) (HTML Eq. (1)). The
differential formula also appears as Eq. (1.3) in
[Yang et al. (2023)](https://arxiv.org/pdf/2305.16176) (HTML Eq. (3)); the old
docstring incorrectly cited its Maxwell–Boltzmann Eq. (1.2).

The old total divided by the squared denominator. With `sigma0/m=1`:

| v/w | Before | Angular integral and corrected API |
| --- | --- | --- |
| 0 | 1 | 1 |
| 0.1 | 0.98029605 | 0.99009901 |
| 1 | 0.25 | 0.5 |
| 3 | 0.01 | 0.1 |
| 10 | 0.0000980296 | 0.0099009901 |

The tests independently integrate the public differential function with
adaptive quadrature at seven ratios from 0 to 100. They also cover array input,
the zero-velocity normalization, and the inverse-square high-velocity limit.
Seven of eight cases failed before correction; all pass afterward. The
5e-12 comparison tolerance covers adaptive quadrature at the sharply forward
peaked high-velocity point and is not an adjustment to catalog tolerances.

This total-cross-section API is not called by the standard structure pipeline.
Effective/viscosity cross sections and gravothermal evolution are separate;
their expressions, catalog specification and existing regression references
are unchanged. Source provenance identifies the corrected public API. This
change does not switch to Møller scattering or establish new halo calibration.
