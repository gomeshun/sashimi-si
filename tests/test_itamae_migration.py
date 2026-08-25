"""Regression tests for the SASHIMI-SI ITAMAE migration boundary."""

import json
from pathlib import Path

import numpy as np
import pytest

import sashimi_si
import sashimi_si_itamae
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import invert_nfw_mass_function
from itamae.provenance import MIGRATION_METADATA_KEYS
from sashimi_si_itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
    create_itamae_model,
)
from sashimi_si import TidalStrippingSolver, halo_model, subhalo_properties


_GOLDEN_PROVENANCE = Path(__file__).parent / "golden" / "sidm_small_catalog_provenance.json"
with _GOLDEN_PROVENANCE.open(encoding="utf-8") as input_file:
    _GOLDEN = json.load(input_file)

SMALL_CATALOG_PARAMETERS = dict(_GOLDEN["parameters"])
SMALL_CATALOG_GOLDEN_SUMS = np.asarray(_GOLDEN["golden_sums"], dtype=float)


def test_golden_sidecar_provenance_is_complete() -> None:
    """The SI sidecar documents the inline full-catalog regression."""
    assert _GOLDEN["fixture_schema"] == "sashimi-family:golden-provenance:v1"
    assert _GOLDEN["fixture_category"] == "full_small_catalog_golden"
    assert _GOLDEN["fixture_format"] == "sidecar-for-inline-regression"
    assert _GOLDEN["variant"] == "sashimi-si"
    assert len(_GOLDEN["generated_repository_revision"]) == 40
    assert len(_GOLDEN["itamae_source_revision"]) == 40
    assert set(_GOLDEN["physics_modes"]) == {"legacy", "consistent"}
    assert len(_GOLDEN["golden_sums"]) == 27
    assert _GOLDEN["parameters"] == SMALL_CATALOG_PARAMETERS


def _mass_function(mass, weight, bin_edges):
    """Return dN/dln(m) on fixed physical-mass bins."""

    log_edges = np.log(np.asarray(bin_edges, dtype=float))
    counts, _ = np.histogram(
        np.log(np.asarray(mass, dtype=float)),
        bins=log_edges,
        weights=np.asarray(weight, dtype=float),
    )
    return counts / np.diff(log_edges)


def _accumulated_satellite_number(mass, weight, thresholds):
    """Return the expected number of subhaloes above each mass threshold."""

    mass = np.asarray(mass, dtype=float)
    weight = np.asarray(weight, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    return np.asarray([np.sum(weight[mass >= threshold]) for threshold in thresholds])


@pytest.fixture(scope="module")
def legacy_mode_observable_products():
    """Evaluate true legacy and migrated-legacy products on one small grid."""

    legacy = subhalo_properties().subhalo_properties_calc(**SMALL_CATALOG_PARAMETERS)
    migrated_model = create_itamae_model(physics_mode="legacy")
    migrated = migrated_model.subhalo_properties_calc(**SMALL_CATALOG_PARAMETERS)
    catalogs = migrated_model.subhalo_catalogs_calc(**SMALL_CATALOG_PARAMETERS)
    return legacy, migrated, catalogs


def test_itamae_cosmology_matches_legacy_background() -> None:
    """The adapted background should reproduce the legacy implementation."""
    legacy = halo_model()
    migrated = ItamaeHaloModel()
    redshift = np.array([0.0, 0.5, 1.0, 3.0, 7.0])

    np.testing.assert_allclose(
        migrated.Hubble(redshift), legacy.Hubble(redshift), rtol=2.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        migrated.growthD(redshift), legacy.growthD(redshift), rtol=2.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        migrated.rhocrit(redshift), legacy.rhocrit(redshift), rtol=2.0e-12, atol=0.0
    )


def test_itamae_cosmology_preserves_shared_halo_calculations() -> None:
    """Shared halo primitives should remain unchanged before SIDM evolution."""
    legacy = halo_model()
    migrated = ItamaeHaloModel()
    mass = np.array([1.0e8, 1.0e10, 1.0e12]) * legacy.Msun
    redshift = np.array([0.0, 1.0, 3.0])

    np.testing.assert_allclose(
        migrated.sigmaMz(mass, redshift),
        legacy.sigmaMz(mass, redshift),
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        migrated.Mvir_from_M200_fit(mass, redshift),
        legacy.Mvir_from_M200_fit(mass, redshift),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_itamae_shanks_solver_matches_legacy() -> None:
    """Migrated perturbative stripping should reproduce the legacy solver."""
    host_mass = 1.0e10
    legacy = TidalStrippingSolver(host_mass, z_min=0.0, z_max=1.5, n_z_interp=32)
    migrated = ItamaeTidalStrippingSolver(
        host_mass, z_min=0.0, z_max=1.5, n_z_interp=32
    )
    mass = np.array([1.0e6, 1.0e7, 1.0e8])

    np.testing.assert_allclose(
        migrated.subhalo_mass_stripped_pert2_shanks(mass, 1.0, 0.0),
        legacy.subhalo_mass_stripped_pert2_shanks(mass, 1.0, 0.0),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_normal_constructor_uses_instance_local_dependencies() -> None:
    """The opt-in constructor must not mutate legacy module dependencies."""
    legacy_solver = sashimi_si.TidalStrippingSolver
    legacy = subhalo_properties()
    migrated = create_itamae_model()

    assert legacy.tidal_solver_factory is legacy_solver
    assert migrated.tidal_solver_factory is ItamaeTidalStrippingSolver
    assert migrated.ct_func is invert_nfw_mass_function
    assert sashimi_si.TidalStrippingSolver is legacy_solver
    assert callable(migrated.sigma_eff_m)


def test_public_opt_in_module_leaves_legacy_api_unchanged() -> None:
    """Changing the imported module should be the only opt-in action."""
    assert sashimi_si.subhalo_properties is subhalo_properties
    assert sashimi_si.TidalStrippingSolver is TidalStrippingSolver
    assert sashimi_si_itamae.subhalo_properties is ItamaeSubhaloProperties
    assert sashimi_si_itamae.TidalStrippingSolver is ItamaeTidalStrippingSolver
    assert sashimi_si_itamae.halo_model is ItamaeHaloModel


@pytest.mark.parametrize("physics_mode", ["consistent", "legacy"])
def test_small_full_catalog_golden_and_legacy_agreement(physics_mode: str) -> None:
    """Both SI mode labels should retain the canonical full-catalog result."""
    legacy = subhalo_properties().subhalo_properties_calc(**SMALL_CATALOG_PARAMETERS)
    migrated_model = create_itamae_model(physics_mode=physics_mode)
    migrated = migrated_model.subhalo_properties_calc(**SMALL_CATALOG_PARAMETERS)

    assert len(migrated) == 27
    assert all(np.asarray(value).shape == (16,) for value in migrated)
    sums = np.array([np.sum(np.asarray(value, dtype=float)) for value in migrated])
    np.testing.assert_allclose(
        sums,
        SMALL_CATALOG_GOLDEN_SUMS,
        rtol=_GOLDEN["comparison"]["sum_rtol"],
        atol=_GOLDEN["comparison"]["default_atol"],
    )

    for index, (actual, reference) in enumerate(
        zip(migrated, legacy, strict=True)
    ):
        if index in {25, 26}:
            np.testing.assert_array_equal(actual, reference)
        else:
            np.testing.assert_allclose(
                actual,
                reference,
                rtol=(
                    _GOLDEN["comparison"]["array_index_21_rtol"]
                    if index == 21
                    else _GOLDEN["comparison"]["default_rtol"]
                ),
                atol=(
                    _GOLDEN["comparison"]["array_index_21_atol"]
                    if index == 21
                    else _GOLDEN["comparison"]["default_atol"]
                ),
            )


@pytest.mark.parametrize(
    ("state", "weight_index"),
    [("cdm_reference", 23), ("sidm", 24)],
)
def test_legacy_mode_mass_function_and_accumulated_satellite_number_match(
    state: str,
    weight_index: int,
    legacy_mode_observable_products,
) -> None:
    """High-level population observables must match the true public legacy model."""

    legacy, migrated, catalogs = legacy_mode_observable_products
    catalog = catalogs[state]
    bin_edges = np.geomspace(1.0e4, 1.0e7, 10)
    thresholds = np.asarray([1.0e4, 1.0e5, 1.0e6, 1.0e7])

    legacy_mass = legacy[11]
    legacy_weight = legacy[weight_index]
    migrated_mass = migrated[11]
    migrated_weight = migrated[weight_index]

    np.testing.assert_allclose(
        migrated_mass,
        legacy_mass,
        rtol=5.0e-12,
        atol=1.0e-300,
    )
    np.testing.assert_allclose(
        migrated_weight,
        legacy_weight,
        rtol=5.0e-12,
        atol=1.0e-300,
    )
    np.testing.assert_allclose(
        catalog.columns["m_bound"],
        legacy_mass,
        rtol=5.0e-12,
        atol=1.0e-300,
    )
    np.testing.assert_allclose(
        catalog.weight_final,
        legacy_weight,
        rtol=5.0e-12,
        atol=1.0e-300,
    )

    legacy_mass_function = _mass_function(
        legacy_mass,
        legacy_weight,
        bin_edges,
    )
    legacy_accumulated = _accumulated_satellite_number(
        legacy_mass,
        legacy_weight,
        thresholds,
    )
    for mass, weight in (
        (migrated_mass, migrated_weight),
        (catalog.columns["m_bound"], catalog.weight_final),
    ):
        np.testing.assert_allclose(
            _mass_function(mass, weight, bin_edges),
            legacy_mass_function,
            rtol=5.0e-12,
            atol=1.0e-300,
        )
        np.testing.assert_allclose(
            _accumulated_satellite_number(mass, weight, thresholds),
            legacy_accumulated,
            rtol=5.0e-12,
            atol=1.0e-300,
        )


@pytest.mark.parametrize("physics_mode", ["consistent", "legacy"])
def test_generated_catalogs_factor_weights_and_metadata(
    physics_mode: str, tmp_path: Path
) -> None:
    """Catalog views should retain independent generation-stage factors."""
    model = create_itamae_model(physics_mode=physics_mode)
    legacy_result, factors = model.subhalo_properties_calc(
        **SMALL_CATALOG_PARAMETERS,
        return_weight_factors=True,
    )
    catalogs = model.catalogs_from_legacy(
        legacy_result,
        weight_factors=factors,
    )

    assert set(catalogs) == {"cdm_reference", "sidm"}
    for state, weight_index in (("cdm_reference", 23), ("sidm", 24)):
        catalog = catalogs[state]
        assert set(catalog.weights) == {
            "weight_base",
            "weight_concentration",
            "weight_survival",
        }
        np.testing.assert_allclose(
            catalog.weight_final,
            legacy_result[weight_index],
            rtol=2.0e-15,
            atol=0.0,
        )
        assert catalog.metadata["schema_version"] == "1.0"
        assert set(MIGRATION_METADATA_KEYS) <= set(catalog.metadata)
        assert catalog.metadata["sashimi_variant"] == "sashimi-si"
        assert len(catalog.metadata["itamae_source_revision"]) == 40
        assert len(catalog.metadata["sashimi_source_revision"]) == 40
        assert catalog.metadata["sashimi_version"] == "0.1.0a1"
        assert catalog.metadata["catalog_schema_version"] == "1.0"
        assert catalog.metadata["canonical_unit_schema"] == "1.0"
        assert catalog.metadata["variance_identifier"] == "sashimi-si:analytic-cdm-fit:v1"
        assert catalog.metadata["power_identifier"] == "sashimi-si:cdm-linear-power:v1"
        assert catalog.metadata["solver_identifier"] == (
            "sashimi-si:gravothermal-tidal-stripping:v1"
        )
        assert catalog.metadata["cosmology_parameters"] == {
            "omega_m0": 0.315,
            "h": 0.674,
        }
        assert catalog.metadata["backend_identifier"] == model.itamae_backend.identifier
        assert catalog.metadata["state"] == state
        assert catalog.metadata["physics_mode"] == physics_mode
        assert catalog.metadata["physics_mode_equivalence"] == "legacy=consistent"
        assert catalog.metadata["weight_factorization"] == "generation-stage"
        assert catalog.metadata["source_identifier"] == (
            "sashimi-si:upstream-physics:e17d3664dac677b604fd4ff02fb2af105a6937fa"
        )
        np.testing.assert_allclose(
            catalog.weight_final,
            catalog.weights["weight_base"]
            * catalog.weights["weight_concentration"]
            * catalog.weights["weight_survival"],
            rtol=0.0,
            atol=0.0,
        )
        assert all(np.all(value >= 0.0) for value in catalog.weights.values())
        archive = tmp_path / f"{state}.npz"
        catalog.to_npz(archive)
        restored = type(catalog).from_npz(archive)
        for name in catalog.columns:
            np.testing.assert_array_equal(restored.columns[name], catalog.columns[name])
        assert dict(restored.metadata) == dict(catalog.metadata)

    np.testing.assert_allclose(
        catalogs["cdm_reference"].columns["v_max_cdm_acc"],
        legacy_result[5] / (model.km / model.s),
        rtol=0.0,
        atol=0.0,
    )
    assert catalogs["cdm_reference"].metadata["model_identifier"] == (
        "sashimi-si:cdm-reference:v1"
    )
    assert catalogs["sidm"].metadata["model_identifier"] == (
        "sashimi-si:sidm-parametric:v1"
    )


def test_public_catalog_state_alias_and_validation() -> None:
    """The CDM alias should work and invalid modes or states should fail."""
    with pytest.raises(ValueError, match="physics_mode"):
        create_itamae_model(physics_mode="unknown")
    with pytest.raises(ValueError, match="OmegaM=0.315"):
        ItamaeHaloModel(
            cosmology_backend=NativeFlatLCDM(omega_m0=0.3, h=0.674)
        )

    model = create_itamae_model()
    cdm = model.subhalo_catalog_calc(
        **SMALL_CATALOG_PARAMETERS,
        state="cdm",
    )
    assert cdm.metadata["state"] == "cdm_reference"

    with pytest.raises(ValueError, match="state must"):
        model.subhalo_catalog_calc(
            **SMALL_CATALOG_PARAMETERS,
            state="invalid",
        )
