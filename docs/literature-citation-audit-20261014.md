# Literature citation audit: sashimi-si — started 2026-10-14 UTC

Inventoried **195 active definitions**, including inherited/moved functions, private/nested helpers, constructors and tests. Updated **95 docstrings**, of which **91** receive or complete local paper references. The remaining edits clarify units, assumptions or NumPy structure.

The executable baseline before this citation edit is `120f9d27e082776287819468ed73c677f89169ca`. Comparison baseline: author upstream/main `e17d3664dac677b604fd4ff02fb2af105a6937fa`. Moving a definition does not make it new; this citation pass also includes inherited definitions.

The audit screens numerical literals and existing attributions, then matches adopted values to primary equations/tables. Generic algorithms, numerical controls, unit conversions and frozen regression outputs do not receive invented physical calibrations. The full [definition index and hash evidence](literature-citation-audit-20261014.json) distinguish edited definitions, unchanged references and archived evidence. An inventory entry is not proof of physical calibration.

## Findings and limits

- The fiducial sigma0/m=147.1 cm^2/g and w=24.33 km/s are Ando et al. Model I. The gravothermal cutoff 1.1 follows their Section 2.2; it is a model boundary, not a numerical accuracy guarantee.
- The profile/collapse coefficients are linked to Yang et al. PDF Eqs. (2.2)-(2.4), and integral evolution to Eq. (3.3). Derivatives use exact differentiation of the retained polynomial; separately rounded printed derivative coefficients are not substituted.
- Low-level scattering functions retain supplied cross-section units and use matching velocity units; effective interpolators use internal Mpc/s. Public model constructors accept cm^2/g and km/s. The beta argument is stored but is not used in the current evolution kernels.
- The adopted Planck-2018-like background does not recalibrate earlier fits. SI defaults to ct_th=0; the literature 0.77 threshold is documented as a separate selectable rule.

## Verification

The dependency/CI lock advances from ITAMAE `c452c5bbbbf8276870a8ff1f61f77d8bff0d8d7c` to its citation-only revision `4aeb13ed4e3e3e87171fe800e08f74b95c1244a2`. Its executable AST is unchanged.

- All 95 changed docstrings parse as NumPy style, cover explicit arguments and contain their assigned paper identifiers.
- All 26 inventoried tracked Python files have identical executable ASTs after removing only docstrings; assertions, tolerances and equations are unchanged. 0 historical Python files remain byte-identical.
- 72 distinct existing local pytest cases pass on Python 3.11.15. This is source-checkout testing, not a new scientific catalogue or a new calibration.
- The repository CI Ruff scope passes.
- Historical numerical results keep their original calculation revisions. The docstring edit changes source bytes and build identity; executable equivalence does not permit relabeling old outputs. No tests are deleted or weakened.

## Changed definitions and paper mapping

| Definition | References / change |
| --- | --- |
| [`HaloModel`](../src/sashimi_si/__init__.py#L37) | Units, calibration wording, or NumPy structure clarified. |
| [`HaloModel.__init__`](../src/sashimi_si/__init__.py#L60) | [planck](https://arxiv.org/abs/1807.06209) |
| [`HaloModel.growthD`](../src/sashimi_si/__init__.py#L156) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`TidalStrippingSolver`](../src/sashimi_si/__init__.py#L190) | Units, calibration wording, or NumPy structure clarified. |
| [`TidalStrippingSolver.__init__`](../src/sashimi_si/__init__.py#L221) | Units, calibration wording, or NumPy structure clarified. |
| [`SubhaloProperties`](../src/sashimi_si/__init__.py#L264) | [sashimi-sidm](https://arxiv.org/abs/2403.16633), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SubhaloProperties.__init__`](../src/sashimi_si/__init__.py#L311) | [sashimi-sidm](https://arxiv.org/abs/2403.16633), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SubhaloProperties.Ffunc`](../src/sashimi_si/__init__.py#L380) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.Gfunc`](../src/sashimi_si/__init__.py#L420) | [parkinson](https://arxiv.org/abs/0708.1382), [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.Ffunc_Yang`](../src/sashimi_si/__init__.py#L467) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties._normalized_yang_kernel`](../src/sashimi_si/__init__.py#L516) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.Na_calc`](../src/sashimi_si/__init__.py#L586) | [yang](https://arxiv.org/abs/1104.1757) |
| [`SubhaloProperties.subhalo_catalogs_calc`](../src/sashimi_si/__init__.py#L741) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`SubhaloProperties.subhalo_catalog_calc`](../src/sashimi_si/__init__.py#L1032) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`SubhaloProperties.subhalo_properties_calc`](../src/sashimi_si/__init__.py#L1085) | [scatter](https://arxiv.org/abs/1101.2020), [structure](https://arxiv.org/abs/1803.07691), [survival](https://arxiv.org/abs/astro-ph/0203004) |
| [`SIDMAccretionSlices.build`](../src/sashimi_si/_itamae_components.py#L69) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |
| [`SIDMInitialStructure.initialize`](../src/sashimi_si/_itamae_components.py#L165) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |
| [`SIDMProfileEvolution.evolve`](../src/sashimi_si/_itamae_components.py#L260) | [tracks](https://arxiv.org/abs/1002.3376), [structure](https://arxiv.org/abs/1803.07691), [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`SIDMSurvival`](../src/sashimi_si/_itamae_components.py#L411) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`SIDMSurvival.select`](../src/sashimi_si/_itamae_components.py#L444) | [survival](https://arxiv.org/abs/astro-ph/0203004), [structure](https://arxiv.org/abs/1803.07691) |
| [`SIUnits`](../src/sashimi_si/_physics.py#L15) | [constants](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf) |
| [`SIUnits.__init__`](../src/sashimi_si/_physics.py#L37) | [constants](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf) |
| [`SIBackgroundKernels`](../src/sashimi_si/_physics.py#L72) | [planck](https://arxiv.org/abs/1807.06209) |
| [`SIBackgroundKernels.__init__`](../src/sashimi_si/_physics.py#L93) | [planck](https://arxiv.org/abs/1807.06209) |
| [`SIBackgroundKernels.growthD`](../src/sashimi_si/_physics.py#L192) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`SIBackgroundKernels.dDdz`](../src/sashimi_si/_physics.py#L233) | [growth](https://doi.org/10.1146/annurev.aa.30.090192.002435) |
| [`SIHaloKernels.__init__`](../src/sashimi_si/_physics.py#L325) | Units, calibration wording, or NumPy structure clarified. |
| [`SIHaloKernels.xi`](../src/sashimi_si/_physics.py#L337) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.sigmaMz`](../src/sashimi_si/_physics.py#L370) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.deltac_func`](../src/sashimi_si/_physics.py#L411) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.fc`](../src/sashimi_si/_physics.py#L467) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`SIHaloKernels.Delc`](../src/sashimi_si/_physics.py#L498) | [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`SIHaloKernels.conc200`](../src/sashimi_si/_physics.py#L529) | [correa](https://arxiv.org/abs/1502.00391) |
| [`SIHaloKernels.Mvir_from_M200`](../src/sashimi_si/_physics.py#L579) | [nfw](https://arxiv.org/abs/astro-ph/9611107) |
| [`SIHaloKernels.Mvir_from_M200_fit`](../src/sashimi_si/_physics.py#L626) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`SIHaloKernels.Mvir_from_M200_fit.ffunc`](../src/sashimi_si/_physics.py#L666) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`SIHaloKernels.Mvir_from_M200_fit.xfunc`](../src/sashimi_si/_physics.py#L698) | [conversion](https://arxiv.org/abs/astro-ph/0203169) |
| [`SIHaloKernels.Mzi`](../src/sashimi_si/_physics.py#L742) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.Mzzi`](../src/sashimi_si/_physics.py#L785) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.dMdz`](../src/sashimi_si/_physics.py#L833) | [correa](https://arxiv.org/abs/1502.00391), [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.dsdm`](../src/sashimi_si/_physics.py#L884) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIHaloKernels.dlogSdlogM`](../src/sashimi_si/_physics.py#L927) | [ludlow](https://arxiv.org/abs/1601.02624) |
| [`SIDM_cross_section`](../src/sashimi_si/_physics.py#L972) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_cross_section.__init__`](../src/sashimi_si/_physics.py#L997) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_cross_section.dsigmadcostheta`](../src/sashimi_si/_physics.py#L1020) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_cross_section.sigma_total`](../src/sashimi_si/_physics.py#L1063) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_cross_section.sigma_viscosity`](../src/sashimi_si/_physics.py#L1103) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_cross_section.sigma_eff_m_interpolate`](../src/sashimi_si/_physics.py#L1144) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_cross_section.sigma_eff_m_interpolate_analytical`](../src/sashimi_si/_physics.py#L1202) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model`](../src/sashimi_si/_physics.py#L1285) | [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SIDM_parametric_model.__init__`](../src/sashimi_si/_physics.py#L1312) | [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SIDM_parametric_model.t_collapse`](../src/sashimi_si/_physics.py#L1352) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691), [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model.dVmaxSIDMdtt`](../src/sashimi_si/_physics.py#L1402) | [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SIDM_parametric_model.dVmaxSIDMdtt_numexpr_optimized`](../src/sashimi_si/_physics.py#L1454) | [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SIDM_parametric_model.drmaxSIDMdtt`](../src/sashimi_si/_physics.py#L1501) | [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SIDM_parametric_model.drmaxSIDMdtt_numexpr_optimized`](../src/sashimi_si/_physics.py#L1547) | [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SIDM_parametric_model.get_Vmax0`](../src/sashimi_si/_physics.py#L1593) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model.get_rmax0`](../src/sashimi_si/_physics.py#L1634) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model.get_rhos`](../src/sashimi_si/_physics.py#L1667) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model.get_rs`](../src/sashimi_si/_physics.py#L1707) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model.get_rc`](../src/sashimi_si/_physics.py#L1746) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`SIDM_parametric_model.master_function`](../src/sashimi_si/_physics.py#L1781) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691), [sidm](https://arxiv.org/abs/2305.16176), [sashimi-sidm](https://arxiv.org/abs/2403.16633) |
| [`SITidalKernels.AMz`](../src/sashimi_si/_physics.py#L2029) | [tidal](https://arxiv.org/abs/1803.07691) |
| [`SITidalKernels.zetaMz`](../src/sashimi_si/_physics.py#L2067) | [tidal](https://arxiv.org/abs/1803.07691) |
| [`SITidalKernels.tdynz`](../src/sashimi_si/_physics.py#L2103) | [time](https://arxiv.org/abs/1403.6827), [virial](https://arxiv.org/abs/astro-ph/9710107) |
| [`test_normalized_yang_kernel_against_independent_definite_integral`](../tests/test_eps_normalized_limit.py#L11) | [yang](https://arxiv.org/abs/1104.1757) |
| [`test_backend_is_passed_to_tidal_solver`](../tests/test_itamae_migration.py#L221) | [planck](https://arxiv.org/abs/1807.06209) |
| [`test_public_state_and_cosmology_validation`](../tests/test_itamae_migration.py#L252) | [planck](https://arxiv.org/abs/1807.06209) |
| [`Vmax_fit`](../tests/test_physics_equations.py#L81) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`rmax_fit`](../tests/test_physics_equations.py#L114) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`dVmax_fit`](../tests/test_physics_equations.py#L145) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`drmax_fit`](../tests/test_physics_equations.py#L183) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelDerivatives`](../tests/test_physics_equations.py#L214) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelDerivatives.test_dVmax_matches_finite_difference_of_eq24`](../tests/test_physics_equations.py#L234) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelDerivatives.test_drmax_matches_finite_difference_of_eq24`](../tests/test_physics_equations.py#L271) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelDerivatives.test_numexpr_variants_agree_with_the_numpy_variants`](../tests/test_physics_equations.py#L307) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelDerivatives.test_derivatives_are_linear_in_the_cdm_normalization`](../tests/test_physics_equations.py#L353) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelDerivatives.test_evolution_is_frozen_beyond_the_threshold`](../tests/test_physics_equations.py#L390) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelInversion`](../tests/test_physics_equations.py#L427) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelInversion.test_get_Vmax0_inverts_eq24`](../tests/test_physics_equations.py#L446) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelInversion.test_get_rmax0_inverts_eq24`](../tests/test_physics_equations.py#L480) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestParametricModelInversion.test_eq23_reduces_to_the_initial_nfw_at_tau_zero`](../tests/test_physics_equations.py#L514) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestIntegralApproach`](../tests/test_physics_equations.py#L548) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestIntegralApproach._synthetic_cdm_history`](../tests/test_physics_equations.py#L569) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestIntegralApproach._eq33_reference`](../tests/test_physics_equations.py#L612) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestIntegralApproach.test_master_function_matches_an_independent_eq33_reference`](../tests/test_physics_equations.py#L667) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestIntegralApproach.test_normalizing_at_t_f_would_give_a_different_answer`](../tests/test_physics_equations.py#L702) | [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestCrossSection`](../tests/test_physics_equations.py#L738) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestCrossSection.test_analytic_sigma_eff_matches_brute_force_quadrature`](../tests/test_physics_equations.py#L759) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestCrossSection.test_sigma_eff_approaches_sigma0_in_the_contact_limit`](../tests/test_physics_equations.py#L809) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestCrossSection.test_subhalo_properties_shares_sigma_eff_with_its_parametric_model`](../tests/test_physics_equations.py#L844) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`TestNFWConstants.test_rmax_over_rs`](../tests/test_physics_equations.py#L996) | [sidm](https://arxiv.org/abs/2305.16176), [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |
| [`TestNFWConstants.test_vmax_and_rhos_normalizations`](../tests/test_physics_equations.py#L1032) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |
| [`TestEndToEnd.test_sidm_reduces_to_cdm_when_the_cross_section_vanishes`](../tests/test_physics_equations.py#L1166) | [cross](https://arxiv.org/abs/2205.03392), [sidm](https://arxiv.org/abs/2305.16176) |
| [`test_real_stage_velocities_follow_canonical_nfw_identity`](../tests/test_stage_velocity_units.py#L12) | [nfw](https://arxiv.org/abs/astro-ph/9611107), [structure](https://arxiv.org/abs/1803.07691) |

## Bibliography

- **constants**: [Mohr & Taylor (2005), CODATA recommended values of the fundamental physical constants: 2002, Table XXVI.](https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf)
- **conversion**: [Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.](https://arxiv.org/abs/astro-ph/0203169)
- **correa**: [Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.](https://arxiv.org/abs/1502.00391)
- **cross**: [Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).](https://arxiv.org/abs/2205.03392)
- **growth**: [Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5.](https://doi.org/10.1146/annurev.aa.30.090192.002435)
- **ludlow**: [Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Appendix C.](https://arxiv.org/abs/1601.02624)
- **nfw**: [Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.](https://arxiv.org/abs/astro-ph/9611107)
- **parkinson**: [Parkinson, Cole & Helly (2008), Generating Dark Matter Halo Merger Trees, Section 3.1.](https://arxiv.org/abs/0708.1382)
- **planck**: [Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.](https://arxiv.org/abs/1807.06209)
- **sashimi-sidm**: [Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.](https://arxiv.org/abs/2403.16633)
- **scatter**: [Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.](https://arxiv.org/abs/1101.2020)
- **sidm**: [Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).](https://arxiv.org/abs/2305.16176)
- **structure**: [Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).](https://arxiv.org/abs/1803.07691)
- **survival**: [Hayashi et al. (2003), The Structural Evolution of Substructure.](https://arxiv.org/abs/astro-ph/0203004)
- **tidal**: [Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Eqs. (23)-(24).](https://arxiv.org/abs/1803.07691)
- **time**: [Jiang & van den Bosch (2016), Statistics of Dark Matter Substructure: I. Model and Universal Fitting Functions, Eq. (2).](https://arxiv.org/abs/1403.6827)
- **tracks**: [Penarrubia et al. (2010), The impact of dark matter cusps and cores on the satellite galaxy population around spiral galaxies, Eq. (8), Fig. 6.](https://arxiv.org/abs/1002.3376)
- **virial**: [Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).](https://arxiv.org/abs/astro-ph/9710107)
- **yang**: [Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).](https://arxiv.org/abs/1104.1757)
