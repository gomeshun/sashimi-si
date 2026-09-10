# Formation and truncation boundaries in the shared SI executor

The archived SI model applies both `z_formation > z_accretion` and CDM
truncation to the final weights. The first pipeline adapter had dropped those
cross-state gates. It also evaluated a SIDM pre-accretion history backwards for
unformed candidates, producing a negative normalized time and a NaN core
radius. The small 16-node migration golden did not exercise this boundary.

This repair retains all node positions and the original pre-selection weight.
`valid_accretion` explicitly records the formation gate. SIDM quantities for
unformed nodes are uncomputed, represented by zero, and have zero final weight;
those zeroes are not physical profiles. Both states apply the shared formation
and `c_t > ct_th` gates. SIDM additionally applies its own profile validity.
A nonfinite result inside the valid physical domain remains an error.

The existing Yang EPS Heaviside support is evaluated before division, rather
than computing discarded 0/0 expressions and applying `nan_to_num`. An audit on
the existing 30-mass, 200-host-quadrature test found 17,580 nonfinite trial values,
all outside `m_acc < m_max`; none was inside the physical support. The single
host-quadrature case also preserves the full redshift-dependent mass shape.

## Evidence

- Two new formation/truncation regressions fail before the repair and pass after.
- Frozen A is `e17d3664dac677b604fd4ff02fb2af105a6937fa`, never imported from the
  current product. Independent B adds the 50-digit principal Lambert-W NFW
  inverse and stable selected-branch evaluation. Exact environment, inputs,
  patches and hashes are in `tests/formation_reference/B-all-formation.json`
  and the `sashimi-family/validation/references/sashimi-si` workflow.
- On the 168-node comparison, 164 nodes are formed. B has four nonfinite initial
  SIDM core radii only in the excluded nodes. Every formed-node array agrees at
  relative difference <= 1.28e-14; both states' full final weights agree within
  2.05e-15. The established `5e-12` regression tolerance is unchanged.
- An intentionally high truncation threshold removes both populations. The
  paired weak-interaction test uses the existing scientific CDM-limit tolerance.
- These are domain and migration consistency checks. Larger grid/solver
  convergence and the separate total-cross-section formula review remain release
  requirements; no new SIDM calibration or default is introduced here.

Local validation used ITAMAE `5da8dbbbd88f3f45dd7d2fbe66f11203d9632fa7`:
20 migration tests pass, and the 20 established equation/physics tests also
pass. The still-present old public loop produces two historical square-root
warnings in its excluded nodes; it is removed in the subsequent standard-API
review unit. The shared executor's new formation tests complete without those
warnings. The independent EPS-only B patch leaves all arrays bitwise identical
to B-all, including its historical excluded-node NaNs.
