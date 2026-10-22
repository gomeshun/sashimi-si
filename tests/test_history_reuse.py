"""Exact regression checks for reuse of pointwise SIDM evolution rates."""

import numpy as np
import pytest

from sashimi_si import SIDM_parametric_model


@pytest.mark.parametrize("time_order", ["C", "F"])
@pytest.mark.parametrize("prefix_length", [99, 100])
@pytest.mark.parametrize("zero_scattering", [False, True])
def test_nested_histories_preserve_every_output_bit(prefix_length, zero_scattering, time_order):
    """Keep full and prefix SIDM endpoints identical to independent evaluations.

    Parameters
    ----------
    prefix_length : int
        Odd or even prefix sample count, exercising both Simpson endpoint rules.
    time_order : {"C", "F"}
        Memory order of the time grid, which affects reduction traversal.
    zero_scattering : bool
        Whether to exercise the infinite-collapse-time limit.

    Notes
    -----
    Synthetic, nonuniform 199-point histories have different formation times,
    concentrations, and mass tracks. The reference calls the public evolution
    method separately with the existing collapse-time input but without
    reused integrands, matching the pre-change Picard path. Passing requires byte equality
    of all five outputs for each endpoint, including signed zeros. A reused prefix
    must retain its own Simpson endpoint rule; a truncated full-history integral
    fails this comparison. This checks numerical equivalence, not physical
    calibration or equality across different dependency versions.

    References
    ----------
    Yang et al. (2024), integral prescription, Eq. (3.3).
    https://arxiv.org/abs/2305.16176
    """
    model = SIDM_parametric_model(sigma0_m=0.0 if zero_scattering else 100.0, w=30.0)
    fraction = np.linspace(0.0, 1.0, 199) ** 1.3
    formation = np.array([1.0, 1.5, 2.0, 3.0]) * model.Gyr
    time = formation + fraction[:, None] * np.array([10.0, 9.0, 8.0, 7.0]) * model.Gyr
    time = np.array(time, order=time_order)
    speed = ((1 + .2 * fraction[None, :, None]) * np.array([10., 20., 40., 60.])
             * np.array([.9, 1.1])[:, None, None] * model.km / model.s)
    radius = ((1 + .3 * fraction[None, :, None]) * np.array([.001, .002, .003, .004])
              * np.array([1.1, .9])[:, None, None])
    with np.errstate(divide="ignore", invalid="raise"):
        collapse = model.t_collapse(model.sigma_eff_m(speed), radius, speed)
        shared = model._history_integrands(speed, radius, time, formation, collapse)
        for length in (199, prefix_length):
            velocity, rmax = speed[:, :length], radius[:, :length]
            reference = model.master_function(
                velocity, rmax, time[:length], formation, collapse_time=collapse[:, :length])
            actual = model.master_function(
                velocity, rmax, time[:length], formation,
                collapse_time=collapse[:, :length],
                _integrands=tuple(a[:, :length] for a in shared),
            )
            for expected, result in zip(reference, actual, strict=True):
                assert result.dtype == expected.dtype
                assert result.shape == expected.shape
                assert result.tobytes() == expected.tobytes()


def test_reused_history_requires_matching_shapes():
    """Reject malformed reused history data before computing a SIDM endpoint.

    Notes
    -----
    The controlled arrays are numerical placeholders. Missing collapse times or
    truncated rate arrays must raise ValueError. Shape checks cannot establish
    that caller-supplied rates belong to the same physical model; reuse remains
    private to one population slice, and that equivalence is tested above.
    """
    model = SIDM_parametric_model(sigma0_m=100.0, w=30.0)
    history = np.ones((2, 5, 3))
    time = np.arange(5.0)[:, None] * np.ones((1, 3))
    with pytest.raises(ValueError, match="collapse_time"):
        model.master_function(history, history, time, np.zeros(3),
                              _integrands=(history,) * 3)
    with pytest.raises(ValueError, match="history shape"):
        model.master_function(history, history, time, np.zeros(3),
                              collapse_time=history, _integrands=(history[:, :4],) * 3)
