"""Independent regression checks for the standalone main maintenance branch."""
import json
from pathlib import Path
import warnings

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad

import sashimi_si as si


@pytest.mark.parametrize("ratio", [0., .1, 1., 3., 10., 30., 100.])
def test_total_is_angular_integral(ratio):
    model = si.SIDM_cross_section()
    expected, _ = quad(lambda mu: model.dsigmadcostheta(1., 1., ratio, mu),
                       -1., 1., epsabs=1e-13, epsrel=1e-12)
    np.testing.assert_allclose(model.sigma_total(1., 1., ratio), expected, rtol=5e-12)


def test_effective_cross_section_has_no_discarded_overflow():
    model = si.SIDM_cross_section()
    with np.errstate(over="raise", invalid="raise"):
        f = model.sigma_eff_m_interpolate_analytical(1., 24.33 * model.km/model.s)
    assert np.all(np.isfinite(f.y))


def test_effective_cross_section_cancellation_against_65_digits():
    model = si.SIDM_cross_section()
    w = 24.33 * model.km/model.s
    f = model.sigma_eff_m_interpolate_analytical(1., w)
    a = w*w/(4*(.64*f.x)**2)
    selected = np.flatnonzero((a >= 20) & (a <= 703))
    with mp.workdps(65):
        expected = np.array([float(-mp.mpf(float(a[i]))**2 *
            (mp.exp(float(a[i]))*(1+mp.mpf(float(a[i])))*mp.ei(-float(a[i]))+1))
            for i in selected])
    np.testing.assert_allclose(f.y[selected], expected, rtol=5e-13)


@pytest.mark.parametrize("prescription", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 4, 64, 200])
def test_eps_redshift_mass_shape_and_support(prescription, order):
    model = si.subhalo_properties()
    z = np.array([.25, .5, 1.])
    ma = model.Mvir_from_M200_fit(np.logspace(7, 11, 5)[None, :]*model.Msun,
                                 z[:, None])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        rate = model.Na_calc(ma, z, 1e12*model.Msun, N_herm=order,
                             Na_model=prescription)
    assert rate.shape == (3, 5)
    assert np.all(np.isfinite(rate)) and np.all(rate >= 0)


def test_zero_barrier_gap_keeps_finite_normalized_kernel(monkeypatch):
    model = si.subhalo_properties()
    monkeypatch.setattr(model, "deltac_func", lambda z: np.ones_like(z)*1.686)
    z = np.array([.5, 1.])
    # At zero barrier gap the normalized kernel remains positive, rather than
    # being silently replaced by zero via nan_to_num.
    rate = model.Na_calc(np.array([1e7, 1e8])*model.Msun, z,
                         1e12*model.Msun, N_herm=4, Na_model=3)
    assert np.all(np.isfinite(rate)) and np.all(rate > 0)


def test_nfw_inverse_resolves_survival_boundary():
    model = si.subhalo_properties()
    radii = np.array([.77-1e-8, .77+1e-8, 1e-5, 500.])
    with mp.workdps(65):
        mass = np.array([float(mp.log1p(mp.mpf(float(x)))-mp.mpf(float(x))/(1+mp.mpf(float(x)))) for x in radii])
    actual = model.ct_func(mass)
    np.testing.assert_allclose(actual, radii, rtol=1e-11)
    np.testing.assert_array_equal(actual > .77, radii > .77)
    assert model.ct_func(0.) == 0.


def test_unformed_nodes_are_not_evolved_backwards():
    model = si.subhalo_properties()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        arrays = model.subhalo_properties_calc(
            M0=1e12, dz=.2, zmax=2, N_ma=12, N_herm=2,
            logmamin=9., N_hermNa=4, method="pert2_shanks")
    assert not [w for w in observed if issubclass(w.category, RuntimeWarning)]
    for value in arrays:
        assert np.all(np.isfinite(value))
    excluded = arrays[23] == 0
    assert np.any(excluded)
    assert np.all(arrays[24][excluded] == 0)
    # The fixed main contract keeps all nodes and represents uncomputed SIDM
    # profiles with zero. They are not physical zero-radius halos.
    assert np.all(arrays[8][excluded] == 0)


def test_total_correction_does_not_enter_catalog(monkeypatch):
    model = si.subhalo_properties()
    def forbidden(*args, **kwargs):
        raise AssertionError("total cross section is not the effective cross section")
    monkeypatch.setattr(si.SIDM_cross_section, "sigma_total", forbidden)
    config = json.loads((Path(__file__).parent/'validation/maintenance/A-config.json').read_text())
    arrays = model.subhalo_properties_calc(**config['parameters'])
    assert all(np.all(np.isfinite(x)) for x in arrays)


@pytest.mark.parametrize("name", ["B-accurate", "B-accurate-formation"])
def test_full_catalog_against_independently_patched_main(name):
    directory = Path(__file__).parent/'validation/maintenance'
    config = json.loads((directory/f'{name}.json').read_text())
    reference = np.load(directory/f'{name}-reference.npz')
    actual = si.subhalo_properties().subhalo_properties_calc(**config['parameters'])
    formed = reference['tuple_23'] > 0
    sidm_fields = {6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 22, 26}
    for i, value in enumerate(actual):
        expected = reference[f'tuple_{i}']
        assert np.shape(value) == expected.shape
        selected = formed if i in sidm_fields else np.ones_like(formed)
        if i in (25, 26):
            np.testing.assert_array_equal(value[selected], expected[selected])
        else:
            np.testing.assert_allclose(value[selected], expected[selected], rtol=5e-12, atol=0,
                                       err_msg=f'column {i}')
    for i in sidm_fields:
        assert np.all(actual[i][~formed] == 0)
