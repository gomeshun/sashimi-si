# Standard SIDM API migration

`from sashimi_si import SubhaloProperties` is the canonical entry point. The
ITAMAE-named modules are import aliases. `physics_mode` and dynamic legacy-class
adaptation are removed. The archived corrected SI reference is reproduced only
by the isolated A/B workflow in `sashimi-family`.

The primary calculation returns `cdm_reference` and `sidm` weighted catalogs.
The 27-field tuple is built from those catalogs. Its order is preserved; its four
velocity fields use the historical Mpc/s convention, whereas named catalogs use
km/s. `SIDM_cross_section`, `SIDM_parametric_model`, cosmology/halo primitives and
the tidal solver remain available through the standard module. Required
physical kernels have been extracted from the retired class instead of inherited
from a hidden legacy implementation.

The explicit backend propagates into the tidal solver. Interaction parameters,
formation/truncation validity, solver options, grids, cosmology, SI rounded-G
convention and source revisions are recorded under calculation specification
`sashimi-si:paired-catalog:2026-09-10:v1`. The defaults remain unchanged.

Validation used ITAMAE `5da8dbbbd88f3f45dd7d2fbe66f11203d9632fa7`:

- All 35 equation, migration, formation-boundary, weak-interaction, serialization
  and public-contract tests pass, with no runtime warnings.
- All 27 small-catalog fields match independent corrected reference B within
  8.77e-15 relative difference. The established 5e-12 tolerance is unchanged.
- Both the usage walkthrough and separate scientific notebook execute in fresh
  kernels. The scientific notebook includes the 168/164-node formation case.

The total-cross-section formula has a separate, pending scientific adoption
question. This API refactor does not change it. Further convergence, full
release artifact checks and publication are later units; these tests alone do
not establish a new physical calibration.
