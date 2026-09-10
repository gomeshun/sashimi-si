# Changelog

## 0.2.0rc1 — migration review candidate

The standard `sashimi_si` import exposes paired named CDM-reference/SIDM
catalogs with shared node identity, canonical units, independent weight factors
and state-specific survival. Product legacy execution is removed; frozen
references preserve earlier results outside the runtime.

Total cross section now uses the adopted one-power denominator; differential
cross section retains its squared denominator. Direct thermal integration,
formation-domain handling, finite EPS limits and canonical stage velocities are
validated separately. Tidal/gravothermal calibrations and profile prescription
remain SI-owned; small-interaction and strong-state diagnostics are documented.

This candidate supports Python 3.11–3.13 and depends on
`sashimi-itamae>=0.2.0rc1,<0.3`. Candidate wheels are supplied locally;
public index availability, main integration and publication remain separate
post-review operations. No tolerance enlargement or old-fixture overwrite is
part of release preparation.
