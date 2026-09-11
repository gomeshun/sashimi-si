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


