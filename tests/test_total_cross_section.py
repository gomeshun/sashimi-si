"""Total Rutherford scattering is the angular integral of the public differential API."""

import numpy as np
import pytest
from scipy.integrate import quad

from sashimi_si import SIDM_cross_section


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 100.0])
def test_total_matches_independent_angular_integration(ratio):
    model = SIDM_cross_section()
    sigma0, w = 3.7, 25.0
    integral, error = quad(
        lambda mu: model.dsigmadcostheta(sigma0, w, ratio * w, mu),
        -1.0, 1.0, epsabs=1e-13, epsrel=1e-12, limit=500,
    )
    assert error < 2e-12 * max(integral, 1.0)
    assert model.sigma_total(sigma0, w, ratio * w) == pytest.approx(
        integral, rel=5e-12, abs=0.0
    )


def test_zero_and_array_velocity_limits():
    model = SIDM_cross_section()
    sigma0, w = 4.0, 12.0
    velocity = np.array([0.0, w, 1e5 * w])
    result = model.sigma_total(sigma0, w, velocity)
    assert result.shape == velocity.shape
    assert result[0] == sigma0
    assert result[1] == sigma0 / 2.0
    assert result[-1] * (velocity[-1] / w) ** 2 == pytest.approx(sigma0, rel=2e-10)
