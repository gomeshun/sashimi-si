# Scientific test review: sashimi-si — 2026-10-14

Reviewed **56 test definitions** individually. This count is not the number of parametrized cases. The review proposes future changes; all test bodies, assertions and tolerances are retained.

Executable-code baseline before this documentation change: `37757b3a783a8ed86e43c1f433d467e10cf31bb2`.
Comparison baseline for added definitions: author upstream/main, `e17d3664dac677b604fd4ff02fb2af105a6937fa`.

## Decisions

| Decision | Definitions |
| --- | ---: |
| Retain | 41 |
| Consolidate without losing conditions | 9 |
| Relax only incidental assertions | 3 |
| Retire or replace, conditionally | 3 |

A consolidation retains every independent scenario and positive/negative control. A relaxation removes an implementation constraint only after preserving the physical or provenance invariant. A retirement is conditional on the replacement/surviving coverage stated below; no candidate is an instruction to delete a test.

Exact equality remains appropriate for lossless serialization, unchanged input identity, deterministic weight factorization and algebraic contracts. A near-roundoff comparison with a frozen fixture protects reproduction of a specified calculation; it is not an independent physical accuracy requirement. Analytic solutions, independent quadrature, unit identities and controlled refinement have a separate scientific role.

## Validation of this documentation change

- NumPy docstrings parsed for 175 in-scope definitions; all explicit parameters are documented.
- All 26 tracked Python files have identical executable ASTs after removing only module/class/function docstrings. The test bodies, decorators and assertions are unchanged.
- Preserved reference and historical analysis files remain byte-for-byte unchanged.
- Python 3.11.15: 72 existing pytest cases passed, 0 failures/errors and 0 skips.
- Ruff checks pass; ITAMAE additionally passes its full format and mypy checks.
- The saved numerical sweeps/catalogues retain their original calculation revisions. This audit does not relabel them as runs of a later documentation commit.

## Candidate index

| Test | Proposal | Reason and retained coverage |
| --- | --- | --- |
| [test_archive_rejects_conflicting_explicit_revision](../tests/test_build_source_identity.py#L213) | Consolidate without losing conditions | Consolidate with matching-override and malformed-override cases into one parametrized provenance test; preserve all conflict scenarios. |
| [test_exact_checkout_rejects_conflicting_explicit_revision](../tests/test_build_source_identity.py#L242) | Consolidate without losing conditions | Consolidate override cases while retaining both checkout and archive origins; they protect different identity sources. |
| [test_malformed_explicit_revision_does_not_silently_fall_back](../tests/test_build_source_identity.py#L270) | Consolidate without losing conditions | Consolidate malformed values into the common override matrix; rejecting ambiguous provenance remains required. |
| [test_matching_explicit_archive_revision_is_allowed](../tests/test_build_source_identity.py#L298) | Consolidate without losing conditions | Merge this positive control with conflicting-override coverage so the rejection tests cannot pass by rejecting every build. |
| [test_small_interaction_limit_on_the_shared_executor](../tests/test_formation_validity.py#L114) | Consolidate without losing conditions | Consolidate with TestEndToEnd.test_sidm_reduces_to_cdm_when_the_cross_section_vanishes, preserving one catalogue-level limit and both named/tuple access paths if supported. |
| [test_standard_api_aliases_have_one_calculation_path](../tests/test_itamae_migration.py#L71) | Relax only incidental assertions | Keep old-mode rejection; exact class identity can be consolidated with actual named/tuple population equivalence. |
| [test_backend_is_passed_to_tidal_solver](../tests/test_itamae_migration.py#L221) | Relax only incidental assertions | Keep backend use and physical mass checks; object-identity comparison can be relaxed to behavior under a distinguishing supported backend. |
| [TestIntegralApproach.test_normalizing_at_t_f_would_give_a_different_answer](../tests/test_physics_equations.py#L522) | Consolidate without losing conditions | Consolidate as a sensitivity assertion inside the adjacent master-function test; alone it does not invoke production code. |
| [TestCrossSection.test_subhalo_properties_shares_sigma_eff_with_its_parametric_model](../tests/test_physics_equations.py#L618) | Relax only incidental assertions | Replace or supplement object identity with equality of evaluated cross sections and collapse times; retire the identity-only assertion only after that behavioral coverage exists. |
| [TestCosmology.test_growthD_is_normalized_to_unity_today](../tests/test_physics_equations.py#L645) | Consolidate without losing conditions | Consolidate with the growth-derivative/background test while retaining the normalization point. |
| [TestCosmology.test_dark_energy_density_parameter_is_recovered_today](../tests/test_physics_equations.py#L689) | Retire or replace, conditionally | Retirement/consolidation candidate: this tests constants and their algebra, not the redshift-dependent production dark-energy expression. Keep growth-derivative and direct background-equation checks. |
| [TestNFWConstants.test_rmax_over_rs](../tests/test_physics_equations.py#L758) | Retire or replace, conditionally | Candidate to replace: no production function or stored coefficient is read, so changing the implementation constant would not fail this test. Retain the derivation as reference and test actual stage outputs instead. |
| [TestNFWConstants.test_vmax_and_rhos_normalizations](../tests/test_physics_equations.py#L774) | Retire or replace, conditionally | Candidate to replace: the asserted literals are in the test itself. Production stage-unit identities already add stronger coverage; verify actual collapse-time coefficients before removing this derivation-only test. |
| [TestEndToEnd.test_catalog_is_finite_and_positive_where_subhaloes_survive](../tests/test_physics_equations.py#L873) | Consolidate without losing conditions | Consolidate with catalogue invariants/formation tests if the larger fixture adds no unique support range; preserve positive surviving structure checks. |
| [TestEndToEnd.test_sidm_reduces_to_cdm_when_the_cross_section_vanishes](../tests/test_physics_equations.py#L895) | Consolidate without losing conditions | Consolidate with the named-catalogue small-interaction test, keeping at least one full physical limit and the required API conversion checks. |

## Individual assessment of every test

### tests/test_build_source_identity.py

#### 1. `test_archive_in_unrelated_repository_keeps_embedded_revision`

[Source](../tests/test_build_source_identity.py#L141) · **Retain** · Role: reproducibility.

**Purpose.** Keep a published catalogue attributable to the code inside its source archive.
**Pass.** An exported package inside an unrelated checkout resolves to its embedded source SHA rather than the enclosing repository SHA.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one integration case per distinct build hook; otherwise a valid numerical result may cite the enclosing project's revision.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 2. `test_unidentified_export_rejects_enclosing_repository`

[Source](../tests/test_build_source_identity.py#L165) · **Retain** · Role: reproducibility.

**Purpose.** Reject an export whose scientific source revision cannot be established.
**Pass.** A source export lacking embedded identity raises RuntimeError instead of using the surrounding repository's SHA.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This prevents false reproducibility claims; it is independent of physical accuracy.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 3. `test_exact_source_checkout_uses_own_revision`

[Source](../tests/test_build_source_identity.py#L191) · **Retain** · Role: reproducibility.

**Purpose.** Identify the actual source checkout used to produce a catalogue.
**Pass.** A package at its own Git root resolves to that checkout's exact revision.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an end-to-end provenance case for each packaging system.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 4. `test_archive_rejects_conflicting_explicit_revision`

[Source](../tests/test_build_source_identity.py#L213) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Prevent a build override from relabelling a preserved scientific source archive.
**Pass.** A supplied revision that conflicts with the archive's embedded SHA raises RuntimeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with matching-override and malformed-override cases into one parametrized provenance test; preserve all conflict scenarios.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 5. `test_exact_checkout_rejects_conflicting_explicit_revision`

[Source](../tests/test_build_source_identity.py#L242) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Prevent an environment override from relabelling the checked-out calculation.
**Pass.** A supplied revision different from the checkout's SHA raises RuntimeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate override cases while retaining both checkout and archive origins; they protect different identity sources.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 6. `test_malformed_explicit_revision_does_not_silently_fall_back`

[Source](../tests/test_build_source_identity.py#L270) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Reject an unusable source identifier before a scientific package is built.
**Pass.** Malformed explicit revision values raise RuntimeError even when another source identity is available.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate malformed values into the common override matrix; rejecting ambiguous provenance remains required.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

#### 7. `test_matching_explicit_archive_revision_is_allowed`

[Source](../tests/test_build_source_identity.py#L298) · **Consolidate without losing conditions** · Role: reproducibility.

**Purpose.** Permit an explicitly confirmed archive identity without changing its provenance.
**Pass.** An explicit SHA equal to the embedded archive SHA is accepted and recorded unchanged.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Merge this positive control with conflicting-override coverage so the rejection tests cannot pass by rejecting every build.
**Limit.** These checks establish source attribution, not the numerical or physical correctness of the packaged model.

### tests/test_cross_section_precision.py

#### 8. `test_effective_cross_section_nodes_match_high_precision_definition`

[Source](../tests/test_cross_section_precision.py#L9) · **Retain** · Role: numerical.

**Purpose.** Resolve the effective SIDM cross section where the analytic expression suffers cancellation.
**Pass.** Tabulated interpolation nodes match the saved high-precision definition to rtol=5e-13 with no absolute floor.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain this targeted numerical check; the separate angular/velocity quadrature test checks the physical integral at a looser tolerance.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_eps_normalized_limit.py

#### 9. `test_normalized_yang_kernel_against_independent_definite_integral`

[Source](../tests/test_eps_normalized_limit.py#L11) · **Retain** · Role: numerical.

**Purpose.** Recover the finite normalized EPS limit when the collapse-barrier gap vanishes.
**Pass.** For barrier gaps from zero to ten, the normalized Yang kernel agrees with an independent definite integral at the specified floating-point tolerance.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** The zero-gap limit contributes to host-history quadrature and must not be dropped as a numerical singularity.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 10. `test_host_quadrature_refinement_includes_finite_zero_gap_nodes`

[Source](../tests/test_eps_normalized_limit.py#L39) · **Retain** · Role: numerical.

**Purpose.** Keep accretion rates physical as host-history quadrature samples near-degenerate nodes.
**Pass.** The refined host quadrature returns the expected mass-grid shape with finite nonnegative rates under strict floating-point handling.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because increasing numerical resolution must not introduce NaNs or negative expected counts.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 11. `test_zero_width_normalization_support_has_no_runtime_warning`

[Source](../tests/test_eps_normalized_limit.py#L66) · **Retain** · Role: numerical.

**Purpose.** Handle genuine zero-support EPS normalization cells without undefined arithmetic.
**Pass.** The tested prescriptions and host-quadrature orders return finite nonnegative rates with correct shape under strict floating-point handling.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** A physically empty support and an invalid active variance interval need different treatment.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_formation_validity.py

#### 12. `test_population_does_not_evolve_nodes_before_formation`

[Source](../tests/test_formation_validity.py#L11) · **Retain** · Role: numerical.

**Purpose.** Exclude accretion nodes that precede halo formation without contaminating valid outputs.
**Pass.** The case includes both formed and unformed nodes; every field is finite, invalid nodes have zero final weight, and their accretion core radii are zero.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because evolving an unformed SIDM halo can produce undefined collapse times and spurious population weight.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 13. `test_sidm_weight_includes_the_shared_cdm_truncation_gate`

[Source](../tests/test_formation_validity.py#L36) · **Retain** · Role: numerical.

**Purpose.** Apply tidal disruption to both the CDM reference and SIDM population.
**Pass.** An intentionally excessive c_t threshold makes both final weight sums zero.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain a whole-pipeline check in addition to the component-level survival-mask test.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 14. `test_formed_nodes_and_both_weights_match_independent_reference`

[Source](../tests/test_formation_validity.py#L55) · **Retain** · Role: numerical.

**Purpose.** Preserve valid SIDM evolution while correcting pre-formation nodes.
**Pass.** Of 168 nodes, 164 are valid; all formed physical fields and both full weight arrays match the independent reference to rtol=5e-12; the reference's four invalid core values are recognized.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain the targeted formation regression; near-machine agreement is for this fixed prescription and reference, not a universal physical error requirement.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 15. `test_single_host_quadrature_node_accepts_redshift_dependent_mass_grid`

[Source](../tests/test_formation_validity.py#L90) · **Retain** · Role: numerical.

**Purpose.** Keep a one-node host-history approximation compatible with epoch-dependent accretion masses.
**Pass.** All three EPS prescriptions return finite nonnegative values with the two-by-three input mass-grid shape.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because the collapsed quadrature axis is a distinct edge case from multi-node broadcasting.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 16. `test_small_interaction_limit_on_the_shared_executor`

[Source](../tests/test_formation_validity.py#L114) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Recover the CDM structure as self-interaction strength tends to zero.
**Pass.** At sigma0/m=1e-8, surviving SIDM and CDM velocity, radius, scale radius and density agree to 1e-3 and the core is below 1e-3 of r_s.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with TestEndToEnd.test_sidm_reduces_to_cdm_when_the_cross_section_vanishes, preserving one catalogue-level limit and both named/tuple access paths if supported.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_components.py

#### 17. `test_survival_combines_shared_cdm_gate_with_sidm_profile_validity`

[Source](../tests/test_itamae_components.py#L6) · **Retain** · Role: numerical.

**Purpose.** Combine tidal disruption with physically admissible SIDM structure.
**Pass.** The toy c_t array gives CDM mask [False,True,True,True]; negative SIDM core, radius or velocity reduces it to [False,True,False,False].
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an isolated survival-rule test so end-to-end failures can be separated from evolution or quadrature errors.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_migration.py

#### 18. `test_historical_provenance_is_preserved`

[Source](../tests/test_itamae_migration.py#L45) · **Retain** · Role: numerical.

**Purpose.** Keep historical SI modes and independently generated corrected references identifiable.
**Pass.** Old mode labels remain; corrected-reference metadata identify role B, an independent process, source SHA and worker/source hashes.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain provenance of reference data; metadata presence alone does not prove physical correctness.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 19. `test_standard_api_aliases_have_one_calculation_path`

[Source](../tests/test_itamae_migration.py#L71) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Avoid silently selecting a retired SIDM physical mode through compatibility imports.
**Pass.** Standard and historical aliases identify the same classes and obsolete physics_mode arguments raise TypeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep old-mode rejection; exact class identity can be consolidated with actual named/tuple population equivalence.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 20. `test_all_27_arrays_match_independent_corrected_reference`

[Source](../tests/test_itamae_migration.py#L93) · **Retain** · Role: numerical.

**Purpose.** Reproduce the corrected SIDM catalogue on its preserved reference calculation.
**Pass.** All 27 tuple arrays match the independently produced reference, with exact final masks and rtol=5e-12 for physical values.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one full-array regression; this strict tolerance measures fixed-algorithm reproducibility and must not block intentional physics corrections with new independent evidence.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 21. `test_weights_observables_units_and_roundtrip`

[Source](../tests/test_itamae_migration.py#L124) · **Retain** · Role: numerical.

**Purpose.** Keep CDM/SIDM effective counts, canonical velocities and saved data consistent.
**Pass.** Weight factors multiply to final weights, velocities convert correctly, histogram/threshold counts match the reference to 5e-12, and catalogue arrays/metadata round-trip exactly.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain these scientifically meaningful output contracts; exact conversion/storage checks are justified algebraic identities.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 22. `test_paired_states_keep_initial_node_identity`

[Source](../tests/test_itamae_migration.py#L190) · **Retain** · Role: numerical.

**Purpose.** Make CDM-reference and SIDM views refer to the same accretion nodes.
**Pass.** Both views contain identical physical column arrays and equal base/concentration weights, differing through survival weighting.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain pairing to isolate interaction effects; array equality here is shared-state identity, not a claim that CDM and SIDM structure columns equal each other.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 23. `test_backend_is_passed_to_tidal_solver`

[Source](../tests/test_itamae_migration.py#L221) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Use the selected cosmological/unit configuration for SI tidal mass loss.
**Pass.** The supplied backend configuration reaches the solver and produces positive bound masses no larger than accretion masses.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Keep backend use and physical mass checks; object-identity comparison can be relaxed to behavior under a distinguishing supported backend.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 24. `test_public_state_and_cosmology_validation`

[Source](../tests/test_itamae_migration.py#L243) · **Retain** · Role: numerical.

**Purpose.** Reject incompatible SI backgrounds and ambiguous population-view names.
**Pass.** An incompatible Omega_m0 and an unknown state fail; the supported cdm alias records cdm_reference.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because the requested physical state and background determine interpretation of every returned weight.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_itamae_variance.py

#### 25. `test_si_variance_adapter_matches_legacy_model`

[Source](../tests/test_itamae_variance.py#L9) · **Retain** · Role: numerical.

**Purpose.** Preserve the CDM initial density variance that seeds the SIDM population.
**Pass.** At three mass/redshift pairs, sigma and dS/dM are unchanged and variance equals sigma squared to rtol=2e-15.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain adapter consistency without treating agreement with the same fitted model as independent calibration.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_physics_equations.py

#### 26. `TestParametricModelDerivatives.test_dVmax_matches_finite_difference_of_eq24`

[Source](../tests/test_physics_equations.py#L184) · **Retain** · Role: numerical.

**Purpose.** Differentiate the SIDM maximum-velocity fit consistently with its evolution equation.
**Pass.** Over the supplied dimensionless times, the implemented derivative agrees with a central finite difference of the independently written fit to absolute tolerance 1e-6.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an independent equation check that permits intended changes in historical outputs.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 27. `TestParametricModelDerivatives.test_drmax_matches_finite_difference_of_eq24`

[Source](../tests/test_physics_equations.py#L211) · **Retain** · Role: numerical.

**Purpose.** Differentiate the SIDM maximum-velocity radius fit consistently.
**Pass.** The radius derivative agrees with a central finite difference of the independent polynomial to absolute tolerance 1e-6.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because a coefficient error directly changes core-collapse structure.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 28. `TestParametricModelDerivatives.test_numexpr_variants_agree_with_the_numpy_variants`

[Source](../tests/test_physics_equations.py#L237) · **Retain** · Role: numerical.

**Purpose.** Keep optimized SIDM evolution algebraically equivalent to the readable formula.
**Pass.** Both optimized derivatives agree with their NumPy counterparts to rtol=1e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain while both execution paths are used; this is implementation equivalence, complemented by independent finite-difference tests.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 29. `TestParametricModelDerivatives.test_derivatives_are_linear_in_the_cdm_normalization`

[Source](../tests/test_physics_equations.py#L273) · **Retain** · Role: numerical.

**Purpose.** Scale SIDM velocity evolution with the instantaneous CDM velocity amplitude.
**Pass.** Multiplying the CDM normalization by the tested scale array multiplies dVmax/dtau by that array.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain a dimensional scaling invariant separate from polynomial coefficient checks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 30. `TestParametricModelDerivatives.test_evolution_is_frozen_beyond_the_threshold`

[Source](../tests/test_physics_equations.py#L300) · **Retain** · Role: numerical.

**Purpose.** Respect the declared cutoff of the SIDM structural fit.
**Pass.** Both velocity and radius derivatives, including optimized forms, are exactly zero above tt_th.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This verifies the chosen extrapolation policy; it does not establish that freezing is physically correct beyond the fit domain.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 31. `TestParametricModelInversion.test_get_Vmax0_inverts_eq24`

[Source](../tests/test_physics_equations.py#L336) · **Retain** · Role: numerical.

**Purpose.** Recover the initial velocity normalization from the SIDM velocity fit.
**Pass.** Applying the inverse to the independent unit-normalized fit returns one to rtol=1e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an algebraic inverse needed for reconstructing halo structure.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 32. `TestParametricModelInversion.test_get_rmax0_inverts_eq24`

[Source](../tests/test_physics_equations.py#L360) · **Retain** · Role: numerical.

**Purpose.** Recover the initial radius normalization from the SIDM radius fit.
**Pass.** Applying the inverse to the independent unit-normalized radius fit returns one to rtol=1e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain alongside the velocity inverse because the two fitting polynomials differ.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 33. `TestParametricModelInversion.test_eq23_reduces_to_the_initial_nfw_at_tau_zero`

[Source](../tests/test_physics_equations.py#L384) · **Retain** · Role: numerical.

**Purpose.** Start SIDM evolution from the stated NFW halo with no interaction-generated core.
**Pass.** At tau=0, density and scale-radius factors equal one and core radius equals zero within 1e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This is an independent physical boundary condition for the parametric model.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 34. `TestIntegralApproach.test_master_function_matches_an_independent_eq33_reference`

[Source](../tests/test_physics_equations.py#L497) · **Retain** · Role: numerical.

**Purpose.** Normalize the SIDM correction along the evolving CDM history.
**Pass.** For a deliberately declining CDM track, final Vmax and rmax match independent integration of the stated equation to rtol=1e-10.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because freezing normalization at formation gives a different physical evolution.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 35. `TestIntegralApproach.test_normalizing_at_t_f_would_give_a_different_answer`

[Source](../tests/test_physics_equations.py#L522) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Ensure the synthetic history can detect the wrong SIDM normalization epoch.
**Pass.** The independent running-time and frozen-at-formation formulas differ by more than 1e-3 in both outputs.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate as a sensitivity assertion inside the adjacent master-function test; alone it does not invoke production code.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 36. `TestCrossSection.test_analytic_sigma_eff_matches_brute_force_quadrature`

[Source](../tests/test_physics_equations.py#L557) · **Retain** · Role: numerical.

**Purpose.** Recover the effective scattering rate from its angular and velocity integral.
**Pass.** Across six velocities 0.1..300 km/s, the analytic/interpolated effective cross section agrees with direct two-dimensional quadrature to one percent.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain the independent physical integral; if runtime is excessive, put the costly grid in a slower validation tier rather than replacing it with self-comparison.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 37. `TestCrossSection.test_sigma_eff_approaches_sigma0_in_the_contact_limit`

[Source](../tests/test_physics_equations.py#L595) · **Retain** · Role: numerical.

**Purpose.** Recover isotropic contact scattering at velocities much below the interaction scale.
**Pass.** At Vmax=1e-3 km/s, sigma_eff/m approaches sigma0/m to relative tolerance 1e-3.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an analytic physical limit not supplied by a frozen catalogue.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 38. `TestCrossSection.test_subhalo_properties_shares_sigma_eff_with_its_parametric_model`

[Source](../tests/test_physics_equations.py#L618) · **Relax only incidental assertions** · Role: numerical.

**Purpose.** Keep collapse-time diagnostics tied to the interaction law used for evolution.
**Pass.** The population model and its parametric model hold the identical sigma_eff callable.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Replace or supplement object identity with equality of evaluated cross sections and collapse times; retire the identity-only assertion only after that behavioral coverage exists.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 39. `TestCosmology.test_growthD_is_normalized_to_unity_today`

[Source](../tests/test_physics_equations.py#L645) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Normalize the SI linear growth factor at the present epoch.
**Pass.** D(0) equals one within absolute tolerance 1e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with the growth-derivative/background test while retaining the normalization point.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 40. `TestCosmology.test_dDdz_is_the_derivative_of_growthD`

[Source](../tests/test_physics_equations.py#L665) · **Retain** · Role: numerical.

**Purpose.** Use a growth derivative consistent with the halo accretion history.
**Pass.** At z=0,0.5,1,2,4,7,10, dD/dz agrees with a central difference of growthD to relative tolerance 1e-5.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because an erroneous h factor in the background can change the host-growth exponent.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 41. `TestCosmology.test_dark_energy_density_parameter_is_recovered_today`

[Source](../tests/test_physics_equations.py#L689) · **Retire or replace, conditionally** · Role: numerical.

**Purpose.** Check the stored flat-background density-parameter normalization.
**Pass.** Omega_L/(Omega_L+Omega_m) equals Omega_L to relative tolerance 1e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retirement/consolidation candidate: this tests constants and their algebra, not the redshift-dependent production dark-energy expression. Keep growth-derivative and direct background-equation checks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 42. `TestNFWConstants.test_rmax_over_rs`

[Source](../tests/test_physics_equations.py#L758) · **Retire or replace, conditionally** · Role: numerical.

**Purpose.** Check the numerical NFW peak-radius constant quoted by the SI model.
**Pass.** An independent root of the NFW circular-speed stationarity condition agrees with literal 2.1626 to absolute tolerance 1e-4.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Candidate to replace: no production function or stored coefficient is read, so changing the implementation constant would not fail this test. Retain the derivation as reference and test actual stage outputs instead.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 43. `TestNFWConstants.test_vmax_and_rhos_normalizations`

[Source](../tests/test_physics_equations.py#L774) · **Retire or replace, conditionally** · Role: numerical.

**Purpose.** Check the rounded NFW structure constants quoted in the SIDM prescription.
**Pass.** Independent NFW algebra agrees with literal coefficients 1.648 and 4.625 to absolute tolerance 1e-3.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Candidate to replace: the asserted literals are in the test itself. Production stage-unit identities already add stronger coverage; verify actual collapse-time coefficients before removing this derivation-only test.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 44. `TestEndToEnd.test_catalog_is_finite_and_positive_where_subhaloes_survive`

[Source](../tests/test_physics_equations.py#L873) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Require physically usable CDM reference structure in the full SI catalogue.
**Pass.** At least one node survives; surviving CDM masses, velocities, radii and densities are finite and positive.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with catalogue invariants/formation tests if the larger fixture adds no unique support range; preserve positive surviving structure checks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 45. `TestEndToEnd.test_sidm_reduces_to_cdm_when_the_cross_section_vanishes`

[Source](../tests/test_physics_equations.py#L895) · **Consolidate without losing conditions** · Role: numerical.

**Purpose.** Recover collisionless structure in the vanishing-interaction limit.
**Pass.** At sigma0/m=1e-8, surviving SIDM and CDM structure agrees to rtol=1e-3 and the SIDM core is less than 1e-3 of r_s.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Consolidate with the named-catalogue small-interaction test, keeping at least one full physical limit and the required API conversion checks.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_shared_picard.py

#### 46. `test_full_history_default_matches_direct_reference_and_invalidates_state`

[Source](../tests/test_shared_picard.py#L10) · **Retain** · Role: numerical.

**Purpose.** Preserve the entire tidal history needed for SIDM structural evolution.
**Pass.** For 55 masses and 100 epochs from z=5 to 0, default histories match DOP853 to rtol=1e-3; doubling host mass invalidates state and the updated histories still match.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain full-history accuracy because endpoint agreement alone cannot validate time-integrated SIDM corrections.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 47. `test_shared_picard_does_not_change_accretion_or_population_weights`

[Source](../tests/test_shared_picard.py#L42) · **Retain** · Role: numerical.

**Purpose.** Isolate the new SIDM tidal integrator from the initial population prescription.
**Pass.** Initial masses, epochs and base/concentration weights equal the historical solver; bound masses match direct DOP853 to rtol=1e-3 and metadata record Picard without fallback.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain paired-state integration coverage; survival weights may legitimately change with evolved structure and are not required equal.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 48. `test_fallback_and_invalid_options_are_not_silent`

[Source](../tests/test_shared_picard.py#L86) · **Retain** · Role: numerical.

**Purpose.** Expose failed history convergence and reject controls belonging to another solver.
**Pass.** A strict one-iteration history query warns and counts one fallback mass; unsupported mxstep raises TypeError.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain model-level propagation of the core's numerical safeguards.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_shared_tidal_ode.py

#### 49. `test_tidal_solver_preserves_direct_scipy_values`

[Source](../tests/test_shared_tidal_ode.py#L28) · **Retain** · Role: numerical.

**Purpose.** Preserve the variant's physical stripping equation when using the common ODE controller.
**Pass.** With default and explicit controls, the variant's returned history or endpoint is exactly equal to direct SciPy odeint on the same equation and grid.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain one integration test per distinct adapter; common-controller unit tests cannot detect variant-specific argument or unit errors.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 50. `test_zero_evolution_preserves_initial_masses`

[Source](../tests/test_shared_tidal_ode.py#L58) · **Retain** · Role: numerical.

**Purpose.** Keep the accreted mass unchanged when no cosmic evolution is requested.
**Pass.** Equal accretion and final redshifts return the input masses exactly.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** This identity catches output-index and zero-duration handling errors.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 51. `test_failed_integration_does_not_return_mass_values`

[Source](../tests/test_shared_tidal_ode.py#L81) · **Retain** · Role: numerical.

**Purpose.** Prevent failed physical stripping from being reported as a successful mass calculation.
**Pass.** Restricting odeint to mxstep=1 raises RuntimeError rather than returning masses.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain propagation through the variant boundary even though ITAMAE also tests the numerical controller.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 52. `test_sidm_multiple_and_repeated_output_times`

[Source](../tests/test_shared_tidal_ode.py#L105) · **Retain** · Role: numerical.

**Purpose.** Keep every requested SIDM history epoch, including duplicate redshifts.
**Pass.** The repeated descending grid returns exactly the same history as direct SciPy odeint.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain because SIDM evolution consumes trajectories rather than just a final mass.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 53. `test_state_first_jacobian_adapter_preserves_stiff_solution`

[Source](../tests/test_shared_tidal_ode.py#L131) · **Retain** · Role: numerical.

**Purpose.** Adapt the historical state-first Jacobian without changing stiff evolution.
**Pass.** For two decay rates 1 and 1000, the supplied Jacobian is actually called with correct argument shapes and the result equals direct odeint.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain while the variant exposes historical Jacobian arguments; it exercises a signature conversion absent from the generic core tests.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_stage_velocity_units.py

#### 54. `test_real_stage_velocities_follow_canonical_nfw_identity`

[Source](../tests/test_stage_velocity_units.py#L12) · **Retain** · Role: numerical.

**Purpose.** Use km/s consistently at every SI population-stage boundary.
**Pass.** Recorded initial and evolved CDM velocities satisfy the canonical NFW relation to rtol=5e-15 and are transported unchanged into all four catalogue velocity columns.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain this physical unit identity; it can detect unit mistakes even when tuple and catalogue code share the same error.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

### tests/test_total_cross_section.py

#### 55. `test_total_matches_independent_angular_integration`

[Source](../tests/test_total_cross_section.py#L11) · **Retain** · Role: numerical.

**Purpose.** Integrate the differential scattering law to the total cross section.
**Pass.** For velocity/scale ratios 0..100, angular quadrature meets its error target and agrees with sigma_total to rtol=5e-12.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain an independent physical definition distinct from the effective transport cross section.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.

#### 56. `test_zero_and_array_velocity_limits`

[Source](../tests/test_total_cross_section.py#L44) · **Retain** · Role: numerical.

**Purpose.** Recover analytic total-scattering limits for vectorized velocity inputs.
**Pass.** The output keeps shape, equals sigma0 at zero velocity and sigma0/2 at v=w, and follows the high-velocity inverse-square limit to 2e-10.
**Fail.** Any stated comparison/identity mismatch, missing expected exception/warning, or unexpected calculation error fails the case.
**Necessity / proposal.** Retain these physical limits and vectorization because the total and effective cross sections have different meanings.
**Limit.** This controlled calculation does not calibrate a subhalo population against simulations or observations.
