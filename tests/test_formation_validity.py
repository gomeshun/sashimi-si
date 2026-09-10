"""Formation and shared truncation boundaries in the SI population executor."""

import json
from pathlib import Path
import numpy as np
from sashimi_si_itamae_migration import create_itamae_model

PARAMETERS = dict(M0=1e12, dz=0.2, zmax=2.0, logmamin=9.0, N_ma=12, N_herm=2, N_hermNa=4)


def test_population_does_not_evolve_nodes_before_formation():
    model = create_itamae_model()
    catalogs = model.subhalo_catalogs_calc(**PARAMETERS)
    cdm, sidm = catalogs["cdm_reference"], catalogs["sidm"]
    valid = cdm.columns["valid_accretion"]
    assert np.any(valid) and np.any(~valid)
    for catalog in (cdm, sidm):
        assert all(np.all(np.isfinite(v)) for v in catalog.columns.values())
        assert np.all(catalog.weight_final[~valid] == 0)
    assert np.all(sidm.columns["r_c_sidm_acc"][~valid] == 0)


def test_sidm_weight_includes_the_shared_cdm_truncation_gate():
    catalogs = create_itamae_model().subhalo_catalogs_calc(**PARAMETERS, ct_th=1e6)
    assert np.sum(catalogs["cdm_reference"].weight_final) == 0
    assert np.sum(catalogs["sidm"].weight_final) == 0


def test_formed_nodes_and_both_weights_match_independent_reference():
    reference_dir = Path(__file__).parent / "formation_reference"
    provenance = json.loads((reference_dir / "B-accurate-formation.json").read_text())
    model = create_itamae_model()
    parameters = provenance["calculation"]["parameters"]
    catalogs = model.subhalo_catalogs_calc(**parameters)
    valid = catalogs["sidm"].columns["valid_accretion"]
    result = model.subhalo_properties_calc(**parameters)
    assert valid.size == 168 and valid.sum() == 164
    with np.load(reference_dir / "B-accurate-formation.npz") as reference:
        for i, actual in enumerate(result):
            mask = np.ones_like(valid) if i in (23, 24) else valid
            np.testing.assert_allclose(
                np.asarray(actual)[mask],
                reference[f"tuple_{i}"][mask],
                rtol=5e-12,
                atol=1e-300,
            )
        assert np.count_nonzero(~np.isfinite(reference["tuple_8"])) == 4


def test_single_host_quadrature_node_accepts_redshift_dependent_mass_grid():
    model = create_itamae_model()
    redshift = np.array([0.5, 1.0])
    mass = np.array([[1e8, 1e9, 1e10], [5e7, 5e8, 5e9]])
    for prescription in (1, 2, 3):
        value = model.Na_calc(mass, redshift, 1e12, N_herm=1, Na_model=prescription)
        assert value.shape == mass.shape
        assert np.all(np.isfinite(value)) and np.all(value >= 0)


def test_small_interaction_limit_on_the_shared_executor():
    catalogs = create_itamae_model(sigma0_m=1e-8).subhalo_catalogs_calc(**PARAMETERS)
    catalog = catalogs["sidm"]
    alive = catalog.weight_final > 0
    assert alive.any()
    for sidm, cdm in [
        ("v_max_sidm", "v_max_cdm"),
        ("r_max_sidm", "r_max_cdm"),
        ("r_s_sidm", "r_s_cdm"),
        ("rho_s_sidm", "rho_s_cdm"),
    ]:
        np.testing.assert_allclose(
            catalog.columns[sidm][alive], catalog.columns[cdm][alive], rtol=1e-3, atol=0
        )
    assert np.all(catalog.columns["r_c_sidm"][alive] < 1e-3 * catalog.columns["r_s_sidm"][alive])
