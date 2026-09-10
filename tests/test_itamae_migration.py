"""The standard API is checked against an independent frozen/patch reference."""

import json
from pathlib import Path
import warnings
import numpy as np
import pytest
from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.provenance import CALCULATION_METADATA_KEYS
from itamae.types import WeightedSubhaloCatalog
from itamae.units import NativeUnits
from sashimi_si import HaloModel, SubhaloProperties, TidalStrippingSolver, CALCULATION_SPECIFICATION
import sashimi_si_itamae as aliases

REF = Path(__file__).parent / "references"
PARAMETERS = json.loads((REF / "B-all.json").read_text())["calculation"]["parameters"]


@pytest.fixture(scope="module")
def products():
    model = SubhaloProperties()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        result = model.subhalo_properties_calc(**PARAMETERS)
        catalogs = model.subhalo_catalogs_calc(**PARAMETERS)
    assert not captured
    return model, result, catalogs


def test_historical_provenance_is_preserved():
    original = json.loads(
        (Path(__file__).parent / "golden/sidm_small_catalog_provenance.json").read_text()
    )
    assert set(original["physics_modes"]) == {"legacy", "consistent"}
    reference = json.loads((REF / "B-all.json").read_text())
    assert reference["role"] == "B"
    assert reference["independent_process"]
    assert len(reference["source_revision"]) == 40
    assert reference["worker_sha256"]
    assert reference["source_export_sha256"]


def test_standard_api_aliases_have_one_calculation_path():
    assert aliases.subhalo_properties is SubhaloProperties
    assert aliases.halo_model is HaloModel
    assert aliases.TidalStrippingSolver is TidalStrippingSolver
    for mode in ("legacy", "consistent", "unknown"):
        with pytest.raises(TypeError, match="physics_mode"):
            SubhaloProperties(physics_mode=mode)


def test_all_27_arrays_match_independent_corrected_reference(products):
    _, result, _ = products
    with np.load(REF / "B-accurate.npz") as reference:
        for i, value in enumerate(result):
            expected = reference[f"tuple_{i}"]
            if i in (25, 26):
                np.testing.assert_array_equal(value, expected)
            else:
                np.testing.assert_allclose(value, expected, rtol=5e-12, atol=1e-300)


@pytest.mark.parametrize(("state", "wi"), [("cdm_reference", 23), ("sidm", 24)])
def test_weights_observables_units_and_roundtrip(products, tmp_path, state, wi):
    model, values, catalogs = products
    catalog = catalogs[state]
    assert set(catalog.weights) == {"weight_base", "weight_concentration", "weight_survival"}
    np.testing.assert_array_equal(catalog.weight_final, values[wi])
    np.testing.assert_array_equal(
        catalog.weight_final, np.prod(list(catalog.weights.values()), axis=0)
    )
    assert all(np.all(np.isfinite(v)) for v in catalog.columns.values())
    assert all(np.all(v >= 0) for v in catalog.weights.values())
    np.testing.assert_allclose(
        catalog.columns["v_max_cdm_acc"], values[5] / (model.km / model.s), rtol=3e-16
    )
    assert set(CALCULATION_METADATA_KEYS) <= set(catalog.metadata)
    assert "physics_mode" not in catalog.metadata
    assert catalog.metadata["calculation_specification"] == CALCULATION_SPECIFICATION
    assert catalog.metadata["state"] == state
    assert catalog.metadata["calculation_parameters"]["method"] == "pert2_shanks"
    with np.load(REF / "B-accurate.npz") as reference:
        mass, weight = reference["tuple_11"], reference[f"tuple_{wi}"]
        bins = np.geomspace(1e4, 1e7, 10)
        np.testing.assert_allclose(
            np.histogram(catalog.columns["m_bound"], bins, weights=catalog.weight_final)[0],
            np.histogram(mass, bins, weights=weight)[0],
            rtol=5e-12,
        )
        for threshold in (1e4, 1e5, 1e6, 1e7):
            np.testing.assert_allclose(
                catalog.weight_final[catalog.columns["m_bound"] >= threshold].sum(),
                weight[mass >= threshold].sum(),
                rtol=5e-12,
            )
    archive = tmp_path / f"{state}.npz"
    catalog.to_npz(archive)
    restored = WeightedSubhaloCatalog.from_npz(archive)
    assert dict(restored.metadata) == dict(catalog.metadata)
    for key in catalog.columns:
        np.testing.assert_array_equal(catalog.columns[key], restored.columns[key])


def test_paired_states_keep_initial_node_identity(products):
    _, _, catalogs = products
    for name in catalogs["cdm_reference"].columns:
        np.testing.assert_array_equal(
            catalogs["cdm_reference"].columns[name], catalogs["sidm"].columns[name]
        )
    for name in ("weight_base", "weight_concentration"):
        np.testing.assert_array_equal(
            catalogs["cdm_reference"].weights[name], catalogs["sidm"].weights[name]
        )


def test_backend_is_passed_to_tidal_solver():
    config = BackendConfig(cosmology=NativeFlatLCDM(omega_m0=0.315, h=0.674), units=NativeUnits())
    solver = TidalStrippingSolver(1e10, z_max=1.5, n_z_interp=32, backend_config=config)
    assert solver.itamae_backend is config
    mass = np.array([1e6, 1e7, 1e8])
    bound = solver.subhalo_mass_stripped_pert2_shanks(mass, 1.0, 0.0)
    assert np.all(bound > 0) and np.all(bound <= mass)


def test_public_state_and_cosmology_validation():
    with pytest.raises(ValueError, match=r"OmegaM=0\.315"):
        HaloModel(cosmology_backend=NativeFlatLCDM(omega_m0=0.3, h=0.674))
    with pytest.raises(ValueError, match="state must"):
        SubhaloProperties().subhalo_catalog_calc(**PARAMETERS, state="invalid")
    assert (
        SubhaloProperties().subhalo_catalog_calc(**PARAMETERS, state="cdm").metadata["state"]
        == "cdm_reference"
    )
