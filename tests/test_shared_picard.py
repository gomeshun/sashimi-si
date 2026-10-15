"""Full SI trajectories and catalog stages use the common numerical backend."""

import numpy as np
import pytest
from itamae.evolution import PicardMassLossSolver, PicardFallbackWarning
from sashimi_si import SubhaloProperties, TidalStrippingSolver
from sashimi_si._picard import get_picard_solver


def test_full_history_default_matches_direct_reference_and_invalidates_state():
    """Preserve the entire tidal history needed for SIDM structural evolution.

    Notes
    -----
    Pass criteria: For 55 masses and 100 epochs from z=5 to 0, default histories match
    DOP853 to rtol=1e-3; doubling host mass invalidates state and the updated histories
    still match.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    solver = TidalStrippingSolver(1e12, z_max=5.0)
    mass = np.geomspace(1e2, 1e10, 55)
    z = np.linspace(5.0, 0.0, 100)
    actual = solver.subhalo_mass_stripped(mass, 5.0, z)
    reference = solver.subhalo_mass_stripped(mass, 5.0, z, method="dop853")
    np.testing.assert_allclose(actual, reference, rtol=1e-3)
    backend = get_picard_solver(solver)
    assert isinstance(backend, PicardMassLossSolver)
    assert backend.diagnostics()["histories"][0]["n_iterations"] == 3
    old_key = backend.state_key()
    solver.M0 = 2e12
    assert backend.state_key() != old_key
    updated = solver.subhalo_mass_stripped(mass, 5.0, z)
    assert not np.array_equal(actual, updated)
    np.testing.assert_allclose(updated, backend.direct(mass, 5.0, z), rtol=1e-3)


def test_shared_picard_does_not_change_accretion_or_population_weights():
    """Isolate the new SIDM tidal integrator from the initial population prescription.

    Notes
    -----
    Pass criteria: Initial masses, epochs and base/concentration weights equal the
    historical solver; bound masses match direct DOP853 to rtol=1e-3 and metadata record
    Picard without fallback.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    options = dict(
        M0=1e10,
        redshift=0.0,
        dz=0.5,
        zmax=1.0,
        N_ma=4,
        N_herm=2,
        logmamin=5.0,
        logmamax=7.0,
        N_hermNa=2,
    )
    model = SubhaloProperties()
    default = model.subhalo_catalogs_calc(**options)
    reference = SubhaloProperties().subhalo_catalogs_calc(**options, method="dop853")
    old = SubhaloProperties().subhalo_catalogs_calc(**options, method="pert2_shanks")
    for state, catalog in default.items():
        metadata = catalog.metadata
        assert metadata["calculation_parameters"]["method"] == "picard"
        assert metadata["picard_diagnostics"]["backend"] == "itamae:picard-log-mass:v1"
        assert metadata["picard_diagnostics"]["fallback_calls"] == 0
        for name in ["weight_base", "weight_concentration"]:
            np.testing.assert_array_equal(catalog.weights[name], old[state].weights[name])
        for name in ["m200_acc", "z_acc"]:
            np.testing.assert_array_equal(catalog.columns[name], old[state].columns[name])
        np.testing.assert_allclose(
            catalog.columns["m_bound"], reference[state].columns["m_bound"], rtol=1e-3
        )


def test_fallback_and_invalid_options_are_not_silent():
    """Expose failed history convergence and reject controls belonging to another solver.

    Notes
    -----
    Pass criteria: A strict one-iteration history query warns and counts one fallback mass;
    unsupported mxstep raises TypeError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    solver = TidalStrippingSolver(1e12)
    with pytest.warns(PicardFallbackWarning, match="convergence"):
        solver.subhalo_mass_stripped(
            [1e6], 5.0, [5.0, 0.0], n_iterations=1, convergence_tolerance=1e-15
        )
    assert get_picard_solver(solver).diagnostics()["fallback_mass_count"] == 1
    with pytest.raises(TypeError):
        solver.subhalo_mass_stripped([1e6], 2.0, [2.0, 0.0], mxstep=1)
