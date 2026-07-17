"""Regression tests for the SASHIMI-SI ITAMAE migration boundary."""

import numpy as np
import pytest

import sashimi_si
import sashimi_si_itamae
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import invert_nfw_mass_function
from itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
    create_itamae_model,
)
from sashimi_si import TidalStrippingSolver, halo_model, subhalo_properties


SMALL_CATALOG_PARAMETERS = {
    "M0": 1.0e10,
    "redshift": 0.0,
    "dz": 0.5,
    "zmax": 1.0,
    "N_ma": 4,
    "sigmalogc": 0.128,
    "N_herm": 2,
    "logmamin": 5.0,
    "logmamax": 7.0,
    "N_hermNa": 2,
    "Na_model": 3,
    "ct_th": 0.0,
    "method": "pert2_shanks",
}

# Frozen sums for all 27 arrays returned by the reduced, full catalog
# calculation above. Both SI physics-mode labels intentionally share this
# golden because there is no known SI-specific physical correction between
# them.
SMALL_CATALOG_GOLDEN_SUMS = np.array(
    [
        4.0649575730516836e7,
        1.2e1,
        2.108927609907079e-3,
        6.697411930147994e17,
        4.560766849185049e-3,
        1.2150085855607123e-18,
        1.7334409013415137e-3,
        1.2080086585330422e18,
        8.402593998967302e-4,
        4.511023230841443e-3,
        1.2323740742189587e-18,
        1.1523467608557079e7,
        1.0621956283726915e-3,
        1.7168240770127242e18,
        2.297104265918782e-3,
        9.602949288122435e-19,
        7.154873599267463e-4,
        8.557771877413161e18,
        3.784206525067025e-4,
        1.8857344992248613e-3,
        1.0555009737428389e-18,
        1.7140338095626925e2,
        9.678584903377276,
        2.897645066121662e2,
        2.897645066121662e2,
        1.6e1,
        1.6e1,
    ]
)


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
        rtol=5.0e-11,
        atol=1.0e-300,
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
                rtol=3.0e-5 if index == 21 else 5.0e-12,
                atol=2.0e-4 if index == 21 else 1.0e-300,
            )


@pytest.mark.parametrize("physics_mode", ["consistent", "legacy"])
def test_generated_catalogs_factor_weights_and_metadata(physics_mode: str) -> None:
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
        assert catalog.metadata["backend_identifier"] == model.itamae_backend.identifier
        assert catalog.metadata["state"] == state
        assert catalog.metadata["physics_mode"] == physics_mode
        assert catalog.metadata["physics_mode_equivalence"] == "legacy=consistent"
        assert catalog.metadata["weight_factorization"] == "generation-stage"
        assert catalog.metadata["source_identifier"] == (
            "sashimi-si:upstream-physics:e17d366"
        )

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
