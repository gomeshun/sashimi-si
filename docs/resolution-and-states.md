# Resolution and state validation — 2026-09-11

One variable is refined at a time: M0=1e12 Msun, observation z=0, zmax=3,
log accretion M200/Msun=6–10, baseline N_ma=16, dz=.25, concentration Hermite
order 3, host Hermite order 8, ct_th=.77 and pert2_shanks. These reduced settings
are not the complete default configuration. The finest grid is a finite
comparator, not continuum truth, and no error budget is inferred for all
parameter choices. All source SHAs, original warnings/failures, settings and
full arrays remain in the family science bundle; summaries and hashes are in
`validation/science`. Displayed metrics were recomputed from every saved
catalog. Finite arrays, nonnegative weights and m_bound <= fitted Mvir_acc
were verified throughout. Repeated baseline and post-EPS catalogs match bitwise.

| Refinement | Count change [%] | Bound mass fraction change [%] | Other metric change [%] |
| --- | ---: | ---: | ---: |
| N_ma-256 → N_ma-500 | -0.00334034 | 0.503122 | 0.283333 |
| dz-0.01 → dz-0.005 | 0.260742 | 0.774398 | 0.0610177 |
| N_herm-5 → N_herm-7 | 0.127806 | 0.160283 | 0.467465 |
| N_hermNa-64 → N_hermNa-200 | 0.00419723 | 0.00301286 | -0.000957593 |
| baseline → odeint | 0 | -3.82093 | -0.875705 |
| odeint → odeint-tight | 0 | -1.34896e-06 | -3.93565e-07 |

The final column is the weight-averaged SIDM Vmax. The two state catalogs are retained separately.

The larger baseline-to-ODE difference is a stripping-approximation effect.
Tightening ODE tolerances changes these metrics by about 1e-6 percent, so it
does not explain the roughly 3.82% bound-mass difference. The default solver
is unchanged. Neither roundoff agreement nor finite-grid stability establishes
simulation calibration. Before scientific use, refine all influential axes
together at the actual host/particle parameters and inspect the relevant
observable, including threshold/discontinuity sensitivity.

The first high-order EPS run exposed a removable variance-gap singularity.
Its warning/failure remains preserved. The selected-domain finite-limit fix
was tested independently; new complete catalogs at orders 64 and 200 contain
no warnings and match the prior successful values bitwise. No clipping,
absolute-value correction or new normalization was introduced.

## Small interaction, formation and strong states

The separate fixed-state sweep uses zmax=2 and the same remaining baseline
settings. At sigma0=1e-8 cm^2/g, CDM-reference and SIDM total count and bound
mass fraction agree exactly; maximum relative profile difference is 4.69e-8
and core radius/r_s <= 4.52e-5. 294 of 336 nodes are formed; unformed nodes
retain identity with zero weights. At sigma0=147.1 the profiles evolve while
these integrated weights coincide; at sigma0=1e4, the SIDM survivor count is
2129.72 versus CDM-reference 8052.71. All three cases have finite arrays,
nonnegative weights, identical base factors across states and no warnings.
These tests exercise the specified states; they are not a new strong-SIDM
calibration.

An unadjusted C/SI comparison differs by up to 2.24% in density: besides
the rounded SI critical density, C fixes the median converted virial mass
across concentration nodes while SI fixes each node M200. In a diagnostic
view aligning critical density and NFW initial-mass normalization, radii
and densities agree within 2.3e-15 and tidal bound masses are bitwise equal.
This explicitly adjusted comparison does not change either product
calibration or claim an unadjusted full C/SI equivalence. Vmax was not part
of the matched-C comparison; the rounded gravitational constants remain
different. The small-interaction SIDM/CDM-reference velocity check is separate.

The total-cross-section one-power denominator was adopted after independent
review. Direct angular integration and the differential formula agree; it
does not change the separate viscosity/effective cross section used here.
