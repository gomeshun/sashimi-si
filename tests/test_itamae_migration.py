"""Regression tests for the SASHIMI-SI ITAMAE migration boundary."""

import numpy as np

from itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
)
from sashimi_si import TidalStrippingSolver, halo_model, subhalo_properties


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


def test_itamae_nfw_inverse_is_injected_and_restored(monkeypatch) -> None:
    """The catalog call should use exact NFW inversion without global leakage."""

    instance = object.__new__(ItamaeSubhaloProperties)
    original_ct = lambda value: np.asarray(value) + 100.0
    instance.ct_func = original_ct
    expected_result = tuple(np.arange(3.0) for _ in range(27))

    def fake_legacy_calculation(self, *args, **kwargs):
        enclosed_fraction = np.array([1.0e-4, 0.1, 1.0, 3.0])
        concentration = self.ct_func(enclosed_fraction)
        np.testing.assert_allclose(
            self.fc(concentration), enclosed_fraction, rtol=2.0e-11, atol=1.0e-13
        )
        return expected_result

    monkeypatch.setattr(
        subhalo_properties,
        "subhalo_properties_calc",
        fake_legacy_calculation,
    )
    result = instance.subhalo_properties_calc()

    assert result is expected_result
    assert instance.ct_func is original_ct


def test_sidm_tuple_is_converted_to_named_state_catalogs() -> None:
    """CDM and SIDM state weights should remain explicit after conversion."""

    size = 4
    values = [np.full(size, float(index)) for index in range(27)]
    values[23] = np.array([0.5, 1.0, 1.5, 2.0])
    values[24] = np.array([0.25, 0.0, 1.5, 1.0])
    values[25] = np.array([True, True, True, False])
    values[26] = np.array([True, False, True, True])

    catalogs = ItamaeSubhaloProperties.catalogs_from_legacy(tuple(values))

    assert set(catalogs) == {"cdm", "sidm"}
    np.testing.assert_array_equal(catalogs["cdm"].weight_final, values[23])
    np.testing.assert_array_equal(catalogs["sidm"].weight_final, values[24])
    np.testing.assert_array_equal(catalogs["cdm"].columns["m200_acc"], values[0])
    np.testing.assert_array_equal(catalogs["sidm"].columns["r_c_sidm"], values[18])
    assert set(catalogs["cdm"].weights) == {"weight_base"}
    assert set(catalogs["sidm"].weights) == {"weight_base"}
    assert catalogs["cdm"].metadata["schema_version"] == "1.0"
    assert catalogs["sidm"].metadata["model_identifier"] == (
        "sashimi-si:sidm:itamae-migration:v1"
    )
    assert catalogs["cdm"].metadata["state"] == "cdm"
    assert catalogs["sidm"].metadata["state"] == "sidm"
    assert catalogs["sidm"].metadata["legacy_survival_folded"] is True
