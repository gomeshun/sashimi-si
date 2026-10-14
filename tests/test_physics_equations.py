"""Physics unit tests for sashimi_si.

These tests pin the equations of Yang et al. (2023) [arXiv:2305.16176] and the
Carroll-Press-Turner growth factor directly, by comparing the implementation against
independently written references (finite differences, brute-force quadrature, exact NFW
constants). They deliberately do NOT compare against the output of an older commit: a
physics fix is *supposed* to change the numbers, so an equivalence-with-the-past test
can only ever stand in the way of correcting the code.

Run with:  pytest tests/test_physics_equations.py
"""

import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import brentq

import sashimi_si


SIGMA0_M = 147.1  # cm^2/g, Milky Way analog of Yang et al. (2023)
W = 24.33  # km/s


@pytest.fixture(scope="module")
def pm():
    """Construct the reference SIDM scattering/profile model for equation checks.

    Returns
    -------
    SIDM_parametric_model
        Model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return sashimi_si.SIDM_parametric_model(sigma0_m=SIGMA0_M, w=W)


@pytest.fixture(scope="module")
def cosmo():
    """Construct the SI background used for growth and density-parameter identity checks.

    Returns
    -------
    HaloModel
        SI calibrated flat background.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return sashimi_si.cosmology()


@pytest.fixture(scope="module")
def tau():
    """Normalized time tau = (t - t_f)/t_c, within the calibrated range.

    Returns
    -------
    numpy.ndarray
        501 dimensionless gravothermal times from 1e-4 to 1.1.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return np.linspace(1.0e-4, 1.1, 501)


# --------------------------------------------------------------------------------------
# Eq. (2.4) of Yang et al. (2023), written out independently of the implementation.
# --------------------------------------------------------------------------------------


def Vmax_fit(t):
    r"""Vmax(tau)/Vmax_0, Eq. (2.4).

    Parameters
    ----------
    t : array_like
        Dimensionless gravothermal time tau=(t_cosmic-t_f)/t_c, within the fit interval.

    Returns
    -------
    numpy.ndarray
        Vmax(tau)/Vmax_initial from the explicit reference polynomial.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """
    return (
        1.0 + 0.1777 * t - 4.399 * t**3 + 16.66 * t**4 - 18.87 * t**5 + 9.077 * t**7 - 2.436 * t**9
    )


def rmax_fit(t):
    r"""rmax(tau)/rmax_0, Eq. (2.4).

    Parameters
    ----------
    t : array_like
        Dimensionless gravothermal time tau=(t_cosmic-t_f)/t_c, within the fit interval.

    Returns
    -------
    numpy.ndarray
        rmax(tau)/rmax_initial from the explicit reference polynomial.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """
    return 1.0 + 0.007623 * t - 0.7200 * t**2 + 0.3376 * t**3 - 0.1375 * t**4


def dVmax_fit(t):
    r"""Differentiate the maximum circular speed polynomial in Yang et al. Eq. (2.4).

    Parameters
    ----------
    t : array_like
        Dimensionless gravothermal time tau=(t_cosmic-t_f)/t_c, within the fit interval.

    Returns
    -------
    numpy.ndarray
        d[Vmax/Vmax_initial]/d tau from the explicit reference polynomial.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    The polynomial is differentiated from the displayed coefficients of Yang et al. Eq.
    (2.4). For example its velocity derivative has 66.64*tau**3, whereas the separate
    derivative list below Eq. (3.3) prints 66.62*tau**3. The radius derivative likewise
    uses -0.55*tau**3, rather than the printed -0.5502*tau**3. These are distinct
    coefficient lists; agreement with an exact derivative test does not verify the
    separately printed list.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """
    return (
        0.1777
        - 3 * 4.399 * t**2
        + 4 * 16.66 * t**3
        - 5 * 18.87 * t**4
        + 7 * 9.077 * t**6
        - 9 * 2.436 * t**8
    )


def drmax_fit(t):
    r"""Differentiate the radius of maximum circular speed polynomial in Yang et al. Eq. (2.4).

    Parameters
    ----------
    t : array_like
        Dimensionless gravothermal time tau=(t_cosmic-t_f)/t_c, within the fit interval.

    Returns
    -------
    numpy.ndarray
        d[rmax/rmax_initial]/d tau from the explicit reference polynomial.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    The polynomial is differentiated from the displayed coefficients of Yang et al. Eq.
    (2.4). For example its velocity derivative has 66.64*tau**3, whereas the separate
    derivative list below Eq. (3.3) prints 66.62*tau**3. The radius derivative likewise
    uses -0.55*tau**3, rather than the printed -0.5502*tau**3. These are distinct
    coefficient lists; agreement with an exact derivative test does not verify the
    separately printed list.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """
    return 0.007623 - 2 * 0.7200 * t + 3 * 0.3376 * t**2 - 4 * 0.1375 * t**3


class TestParametricModelDerivatives:
    r"""Compare SIDM fitted-profile derivatives with independent derivatives of their stated
    functions.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """

    def test_dVmax_matches_finite_difference_of_eq24(self, pm, tau):
        r"""Differentiate the SIDM maximum-velocity fit consistently with its evolution equation.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.
        tau : numpy.ndarray
            Dimensionless SIDM evolution times (t-t_formation)/t_collapse within the chosen fit
            interval.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: Over the supplied dimensionless times, the implemented derivative agrees
        with a central finite difference of the independently written fit to absolute tolerance
        1e-6.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        h = 1.0e-6
        expected = (Vmax_fit(tau + h) - Vmax_fit(tau - h)) / (2 * h)
        assert np.allclose(pm.dVmaxSIDMdtt(tau, np.ones_like(tau)), expected, rtol=0.0, atol=1.0e-6)

    def test_drmax_matches_finite_difference_of_eq24(self, pm, tau):
        r"""Differentiate the SIDM maximum-velocity radius fit consistently.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.
        tau : numpy.ndarray
            Dimensionless SIDM evolution times (t-t_formation)/t_collapse within the chosen fit
            interval.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: The radius derivative agrees with a central finite difference of the
        independent polynomial to absolute tolerance 1e-6.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        h = 1.0e-6
        expected = (rmax_fit(tau + h) - rmax_fit(tau - h)) / (2 * h)
        assert np.allclose(pm.drmaxSIDMdtt(tau, np.ones_like(tau)), expected, rtol=0.0, atol=1.0e-6)

    def test_numexpr_variants_agree_with_the_numpy_variants(self, pm, tau):
        r"""Keep optimized SIDM evolution algebraically equivalent to the readable formula.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.
        tau : numpy.ndarray
            Dimensionless SIDM evolution times (t-t_formation)/t_collapse within the chosen fit
            interval.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: Both optimized derivatives agree with their NumPy counterparts to
        rtol=1e-12.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        ones = np.ones_like(tau)
        assert np.allclose(
            pm.dVmaxSIDMdtt_numexpr_optimized(tau, ones),
            pm.dVmaxSIDMdtt(tau, ones),
            rtol=1.0e-12,
            atol=0.0,
        )
        assert np.allclose(
            pm.drmaxSIDMdtt_numexpr_optimized(tau, ones),
            pm.drmaxSIDMdtt(tau, ones),
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_derivatives_are_linear_in_the_cdm_normalization(self, pm, tau):
        r"""Scale SIDM velocity evolution with the instantaneous CDM velocity amplitude.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.
        tau : numpy.ndarray
            Dimensionless SIDM evolution times (t-t_formation)/t_collapse within the chosen fit
            interval.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: Multiplying the CDM normalization by the tested scale array multiplies
        dVmax/dtau by that array.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        scale = np.linspace(2.0, 50.0, tau.size)
        assert np.allclose(
            pm.dVmaxSIDMdtt(tau, scale), scale * pm.dVmaxSIDMdtt(tau, np.ones_like(tau))
        )

    def test_evolution_is_frozen_beyond_the_threshold(self, pm):
        r"""Respect the declared cutoff of the SIDM structural fit.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: Both velocity and radius derivatives, including optimized forms, are
        exactly zero above tt_th.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        beyond = np.array([pm.tt_th + 1.0e-6, 2.0, 10.0])
        ones = np.ones_like(beyond)
        assert np.all(pm.dVmaxSIDMdtt(beyond, ones) == 0.0)
        assert np.all(pm.drmaxSIDMdtt(beyond, ones) == 0.0)
        assert np.all(pm.dVmaxSIDMdtt_numexpr_optimized(beyond, ones) == 0.0)
        assert np.all(pm.drmaxSIDMdtt_numexpr_optimized(beyond, ones) == 0.0)


class TestParametricModelInversion:
    r"""Check that the initial-profile reconstruction inverts the retained SIDM fit.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """

    def test_get_Vmax0_inverts_eq24(self, pm, tau):
        r"""Recover the initial velocity normalization from the SIDM velocity fit.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.
        tau : numpy.ndarray
            Dimensionless SIDM evolution times (t-t_formation)/t_collapse within the chosen fit
            interval.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: Applying the inverse to the independent unit-normalized fit returns one
        to rtol=1e-12.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        assert np.allclose(pm.get_Vmax0(Vmax_fit(tau), tau), 1.0, rtol=1.0e-12, atol=0.0)

    def test_get_rmax0_inverts_eq24(self, pm, tau):
        r"""Recover the initial radius normalization from the SIDM radius fit.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.
        tau : numpy.ndarray
            Dimensionless SIDM evolution times (t-t_formation)/t_collapse within the chosen fit
            interval.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: Applying the inverse to the independent unit-normalized radius fit
        returns one to rtol=1e-12.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        assert np.allclose(pm.get_rmax0(rmax_fit(tau), tau), 1.0, rtol=1.0e-12, atol=0.0)

    def test_eq23_reduces_to_the_initial_nfw_at_tau_zero(self, pm):
        r"""Start SIDM evolution from the stated NFW halo with no interaction-generated core.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: At tau=0, density and scale-radius factors equal one and core radius
        equals zero within 1e-12.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        assert pm.get_rhos(1.0, 0.0) == pytest.approx(1.0, abs=1.0e-12)
        assert pm.get_rs(1.0, 0.0) == pytest.approx(1.0, abs=1.0e-12)
        assert pm.get_rc(1.0, 0.0) == pytest.approx(0.0, abs=1.0e-12)


class TestIntegralApproach:
    r"""Compare the gravothermal history integral with explicit instantaneous-normalization
    reference polynomials.

    Notes
    -----
    The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
    (3.3). Agreement checks the implemented equations at the controlled inputs, not
    astrophysical calibration.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
    with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
    As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
    The implemented combination is therefore not a literal transcription of either
    displayed integral prescription. Here t_f is the formation-time integration origin,
    which need not equal the accretion time.

    The reference independently evaluates the chosen elapsed-time and running-
    normalization convention. Passing this test verifies that convention, not literal
    equivalence to both papers or a choice between their printed time and normalization
    prescriptions.

    References
    ----------

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176

    .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Section 2.2, Eqs. (2.12)-(2.17), Table 1.
        https://arxiv.org/abs/2403.16633
    """

    @staticmethod
    def _synthetic_cdm_history(pm, n_t=801):
        r"""Construct a deterministic declining CDM reference track with explicit SI unit
        conversions.

        Parameters
        ----------
        pm : SIDM_parametric_model
            Controlled SIDM profile model providing its internal units, cross section and
            collapse-time prescription.
        n_t : int
            Number of cosmic-time samples in the synthetic CDM history used for
            integration-refinement checks.

        Returns
        -------
        tuple
            Vmax and rmax arrays (1,n_t,2), times (n_t,2), and formation times (2,), in internal
            Mpc/s, Mpc and seconds.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        The reference independently evaluates the chosen elapsed-time and running-
        normalization convention. Passing this test verifies that convention, not literal
        equivalence to both papers or a choice between their printed time and normalization
        prescriptions.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Section 2.2, Eqs. (2.12)-(2.17), Table 1.
            https://arxiv.org/abs/2403.16633
        """
        Gyr, km, s, kpc = pm.Gyr, pm.km, pm.s, pm.kpc
        t_f = np.array([2.0, 2.0]) * Gyr
        t = np.linspace(2.0, 13.8, n_t)[:, None] * Gyr * np.ones((1, 2))
        frac = (t - t_f) / (13.8 * Gyr - t_f)  # 0 at accretion -> 1 today
        Vmax = (28.0 - 14.0 * frac) * km / s  # 28 -> 14 km/s
        rmax = (3.0 - 1.5 * frac) * kpc  # 3.0 -> 1.5 kpc
        return Vmax[None, ...], rmax[None, ...], t, t_f

    @staticmethod
    def _eq33_reference(pm, Vmax_CDM, rmax_CDM, t, t_f, freeze_at_tf=False):
        r"""Evaluate an independent reference for the adopted SIDM time convention.

        The polynomial coefficients are defined in this test module.

        Parameters
        ----------
        pm : SIDM_parametric_model
            Controlled SIDM profile model providing its internal units, cross section and
            collapse-time prescription.
        Vmax_CDM : numpy.ndarray
            CDM reference maximum circular speeds in internal Mpc/s, shaped
            (n_concentration,n_time,n_mass).
        rmax_CDM : numpy.ndarray
            CDM radii of maximum circular speed in Mpc, with the same history shape as Vmax_CDM.
        t : numpy.ndarray
            Cosmic-time grid in internal seconds, shape (n_time,n_mass).
        t_f : numpy.ndarray
            Formation cosmic time in SI internal seconds, aligned with each halo-mass track.
        freeze_at_tf : bool
            Use the formation-time normalization as a deliberately alternative reference; false
            uses the instantaneous CDM history.

        Returns
        -------
        tuple of numpy.ndarray
            Final Vmax and rmax from the explicit reference time integral, in the input physical
            units.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        The reference independently evaluates the chosen elapsed-time and running-
        normalization convention. Passing this test verifies that convention, not literal
        equivalence to both papers or a choice between their printed time and normalization
        prescriptions.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Section 2.2, Eqs. (2.12)-(2.17), Table 1.
            https://arxiv.org/abs/2403.16633
        """
        t_c = pm.t_collapse(pm.sigma_eff_m(Vmax_CDM), rmax_CDM, Vmax_CDM)
        tau = (t - t_f) / t_c
        norm_V = Vmax_CDM[:, :1] if freeze_at_tf else Vmax_CDM
        norm_r = rmax_CDM[:, :1] if freeze_at_tf else rmax_CDM
        dV = np.where(tau <= pm.tt_th, dVmax_fit(tau), 0.0) * norm_V / t_c
        dr = np.where(tau <= pm.tt_th, drmax_fit(tau), 0.0) * norm_r / t_c
        x = t * np.ones((len(Vmax_CDM), 1, 1))
        return (
            Vmax_CDM[:, -1] + integrate.simpson(dV, x=x, axis=1),
            rmax_CDM[:, -1] + integrate.simpson(dr, x=x, axis=1),
        )

    def test_master_function_matches_an_independent_eq33_reference(self, pm):
        r"""Normalize the SIDM correction along the evolving CDM history.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: For a deliberately declining CDM track, final Vmax and rmax match
        independent integration with the stated elapsed-time convention to rtol=1e-10.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        The reference independently evaluates the chosen elapsed-time and running-
        normalization convention. Passing this test verifies that convention, not literal
        equivalence to both papers or a choice between their printed time and normalization
        prescriptions.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Section 2.2, Eqs. (2.12)-(2.17), Table 1.
            https://arxiv.org/abs/2403.16633
        """
        args = self._synthetic_cdm_history(pm)
        Vmax_SIDM, rmax_SIDM, _, _, _ = pm.master_function(*args)
        V_ref, r_ref = self._eq33_reference(pm, *args)
        assert np.allclose(Vmax_SIDM, V_ref, rtol=1.0e-10, atol=0.0)
        assert np.allclose(rmax_SIDM, r_ref, rtol=1.0e-10, atol=0.0)

    def test_normalizing_at_t_f_would_give_a_different_answer(self, pm):
        r"""Distinguish running and formation-time SIDM normalization on a controlled track.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        Pass criteria: The independent running-time and frozen-at-formation formulas differ by
        more than 1e-3 in both outputs.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        The reference independently evaluates the chosen elapsed-time and running-
        normalization convention. Passing this test verifies that convention, not literal
        equivalence to both papers or a choice between their printed time and normalization
        prescriptions.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Section 2.2, Eqs. (2.12)-(2.17), Table 1.
            https://arxiv.org/abs/2403.16633
        """
        args = self._synthetic_cdm_history(pm)
        V_running, r_running = self._eq33_reference(pm, *args, freeze_at_tf=False)
        V_frozen, r_frozen = self._eq33_reference(pm, *args, freeze_at_tf=True)
        assert np.all(np.abs(V_frozen / V_running - 1.0) > 1.0e-3)
        assert np.all(np.abs(r_frozen / r_running - 1.0) > 1.0e-3)


class TestCrossSection:
    r"""Check low-velocity limits and relative-velocity conventions of the SIDM cross section.

    Notes
    -----
    The reference scattering and effective-velocity laws are Yang & Yu and Yang et al. These
    comparisons test numerical evaluation of those adopted laws.

    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.

    References
    ----------

    .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
        https://arxiv.org/abs/2205.03392

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """

    def test_analytic_sigma_eff_matches_brute_force_quadrature(self, pm):
        r"""Recover the effective scattering rate from its angular and velocity integral.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

        Notes
        -----
        The reference scattering and effective-velocity laws are Yang & Yu and Yang et al. These
        comparisons test numerical evaluation of those adopted laws.

        Pass criteria: Across six velocities 0.1..300 km/s, the analytic/interpolated effective
        cross section agrees with direct two-dimensional quadrature to one percent.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        km, s, cm, gram = pm.km, pm.s, pm.cm, pm.gram
        sigma0_m, w = SIGMA0_M * cm**2 / gram, W * km / s
        f_ana = pm.sigma_eff_m_interpolate_analytical(sigma0_m, w)

        for V in [0.1, 1.0, 10.0, 24.33, 100.0, 300.0]:
            Vmax = V * km / s
            veff = 0.64 * Vmax
            v = np.linspace(1.0e-8 * veff, 40.0 * veff, 20000)
            costheta = np.linspace(-1.0, 1.0, 2000)
            integrand = (
                pm.dsigmadcostheta(sigma0_m, w, v[:, None], costheta[None, :])
                * v[:, None] ** 7
                * (1.0 - costheta[None, :] ** 2)
                * np.exp(-(v[:, None] ** 2) / (4.0 * veff**2))
            )
            inner = integrate.simpson(integrand, x=costheta, axis=-1)
            reference = integrate.simpson(inner, x=v) / (512.0 * veff**8)
            assert f_ana(Vmax) == pytest.approx(reference, rel=1.0e-2)

    def test_sigma_eff_approaches_sigma0_in_the_contact_limit(self, pm):
        r"""Recover isotropic contact scattering at velocities much below the interaction scale.

        Parameters
        ----------
        pm : SIDM_parametric_model
            SIDM structural model with sigma0/m=147.1 cm^2/g and w=24.33 km/s.

        Notes
        -----
        The reference scattering and effective-velocity laws are Yang & Yu and Yang et al. These
        comparisons test numerical evaluation of those adopted laws.

        Pass criteria: At Vmax=1e-3 km/s, sigma_eff/m approaches sigma0/m to relative tolerance
        1e-3.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        km, s, cm, gram = pm.km, pm.s, pm.cm, pm.gram
        f_ana = pm.sigma_eff_m_interpolate_analytical(SIGMA0_M * cm**2 / gram, W * km / s)
        assert f_ana(1.0e-3 * km / s) / (cm**2 / gram) == pytest.approx(SIGMA0_M, rel=1.0e-3)

    def test_subhalo_properties_shares_sigma_eff_with_its_parametric_model(self):
        r"""Keep collapse-time diagnostics tied to the interaction law used for evolution.

        Notes
        -----
        The reference scattering and effective-velocity laws are Yang & Yu and Yang et al. These
        comparisons test numerical evaluation of those adopted laws.

        Pass criteria: The population model and its parametric model hold the identical
        sigma_eff callable.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        sh = sashimi_si.subhalo_properties(sigma0_m=SIGMA0_M, w=W)
        assert sh.sigma_eff_m is sh.param_model.sigma_eff_m


class TestCosmology:
    """Check growth normalization/derivatives and retained flat-background relations.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """

    def test_growthD_is_normalized_to_unity_today(self, cosmo):
        """Normalize the SI linear growth factor at the present epoch.

        Parameters
        ----------
        cosmo : object
            SI background instance used to evaluate growth and its redshift derivative.

        Notes
        -----
        Pass criteria: D(0) equals one within absolute tolerance 1e-12.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.
        """
        assert cosmo.growthD(0.0) == pytest.approx(1.0, abs=1.0e-12)

    def test_dDdz_is_the_derivative_of_growthD(self, cosmo):
        """Use a growth derivative consistent with the halo accretion history.

        Parameters
        ----------
        cosmo : object
            SI background instance used to evaluate growth and its redshift derivative.

        Notes
        -----
        Pass criteria: At z=0,0.5,1,2,4,7,10, dD/dz agrees with a central difference of growthD
        to relative tolerance 1e-5.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.
        """
        h = 1.0e-6
        for z in [0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]:
            expected = (cosmo.growthD(z + h) - cosmo.growthD(z - h)) / (2 * h)
            assert cosmo.dDdz(z) == pytest.approx(expected, rel=1.0e-5)

    def test_dark_energy_density_parameter_is_recovered_today(self, cosmo):
        """Check the stored flat-background density-parameter normalization.

        Parameters
        ----------
        cosmo : object
            SI background instance used to evaluate growth and its redshift derivative.

        Notes
        -----
        Pass criteria: Omega_L/(Omega_L+Omega_m) equals Omega_L to relative tolerance 1e-12.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.
        """
        assert cosmo.OmegaL / (cosmo.OmegaL + cosmo.OmegaM) == pytest.approx(
            cosmo.OmegaL, rel=1.0e-12
        )


class TestNFWConstants:
    """Derive NFW peak-radius/velocity constants independently of their rounded literal values.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """

    @staticmethod
    def _c_max():
        """rmax/rs, i.e. the c maximizing [ln(1+c) - c/(1+c)]/c.

        Returns
        -------
        float
            Dimensionless NFW radius maximizing circular speed, obtained by root finding.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """

        def stationarity(c):
            """Evaluate the stationary condition for the NFW circular-speed maximum.

            Parameters
            ----------
            c : float
                Positive dimensionless NFW radius r/r_s.

            Returns
            -------
            float
                c**2/(1+c)**2-f(c), whose root gives rmax/r_s.

            Notes
            -----
            This fixture/helper supports the controlled checks in this test module; its synthetic
            values are not a calibrated halo model.
            """
            return c**2 / (1 + c) ** 2 - (np.log(1 + c) - c / (1 + c))

        return brentq(stationarity, 1.0, 5.0)

    def test_rmax_over_rs(self):
        r"""Check the numerical NFW peak-radius constant quoted by the SI model.

        Notes
        -----
        The independent polynomial/reference values are from Yang et al., PDF Eqs. (2.2)-(2.4),
        (3.3). Agreement checks the implemented equations at the controlled inputs, not
        astrophysical calibration.

        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Pass criteria: An independent root of the NFW circular-speed stationarity condition
        agrees with literal 2.1626 to absolute tolerance 1e-4.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691
        """
        assert self._c_max() == pytest.approx(2.1626, abs=1.0e-4)

    def test_vmax_and_rhos_normalizations(self):
        r"""Check the rounded NFW structure constants quoted in the SIDM prescription.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Pass criteria: Independent NFW algebra agrees with literal coefficients 1.648 and 4.625
        to absolute tolerance 1e-3.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691
        """
        c = self._c_max()
        mu = (np.log(1 + c) - c / (1 + c)) / c
        # Vmax / (rs sqrt(G rho_s)), used in t_collapse
        assert np.sqrt(4 * np.pi * mu) == pytest.approx(1.648, abs=1.0e-3)
        # 4 pi G rs^2 rho_s / Vmax^2, used to rebuild rho_s from Vmax and rs
        assert 1.0 / mu == pytest.approx(4.625, abs=1.0e-3)


class TestEndToEnd:
    """Check SI population finiteness, collapse-time distribution and the weak-interaction CDM
    limit.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """

    CALC_INPUT = dict(
        M0=1.0e12,
        redshift=0.0,
        M0_at_redshift=True,
        dz=0.2,
        N_herm=3,
        zmax=4.0,
        logmamin=9,
        N_ma=30,
    )

    OUT_NAMES = [
        "ma200",
        "z_acc",
        "rsCDM_acc",
        "rhosCDM_acc",
        "rmaxCDM_acc",
        "VmaxCDM_acc",
        "rsSIDM_acc",
        "rhosSIDM_acc",
        "rcSIDM_acc",
        "rmaxSIDM_acc",
        "VmaxSIDM_acc",
        "m_z0",
        "rsCDM_z0",
        "rhosCDM_z0",
        "rmaxCDM_z0",
        "VmaxCDM_z0",
        "rsSIDM_z0",
        "rhosSIDM_z0",
        "rcSIDM_z0",
        "rmaxSIDM_z0",
        "VmaxSIDM_z0",
        "ctCDM_z0",
        "tt_ratio",
        "weightCDM",
        "weightSIDM",
        "surviveCDM",
        "surviveSIDM",
    ]

    def _run(self, **overrides):
        """Compute a controlled SI catalogue and label every historical tuple field.

        Parameters
        ----------
        **overrides : dict
            Deliberate changes to the otherwise valid controlled model, stage or numerical
            settings.

        Returns
        -------
        dict[str, numpy.ndarray]
            Paired CDM/SIDM quantities with named physical meaning; tuple-length changes fail
            immediately.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        kwargs = dict(sigma0_m=SIGMA0_M, w=W)
        kwargs.update(overrides)
        out = sashimi_si.subhalo_properties(**kwargs).subhalo_properties_calc(**self.CALC_INPUT)
        assert len(out) == len(self.OUT_NAMES)
        return dict(zip(self.OUT_NAMES, out))

    def test_catalog_is_finite_and_positive_where_subhaloes_survive(self):
        """Require physically usable CDM reference structure in the full SI catalogue.

        Notes
        -----
        Pass criteria: At least one node survives; surviving CDM masses, velocities, radii and
        densities are finite and positive.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.
        """
        cat = self._run()
        alive = cat["weightCDM"] > 0.0
        assert alive.any()
        for key in ["m_z0", "VmaxCDM_z0", "rmaxCDM_z0", "rsCDM_z0", "rhosCDM_z0"]:
            values = np.asarray(cat[key])[alive]
            assert np.all(np.isfinite(values))
            assert np.all(values > 0.0)

    def test_sidm_reduces_to_cdm_when_the_cross_section_vanishes(self):
        r"""Recover collisionless structure in the vanishing-interaction limit.

        Notes
        -----
        The reference scattering and effective-velocity laws are Yang & Yu and Yang et al. These
        comparisons test numerical evaluation of those adopted laws.

        Pass criteria: At sigma0/m=1e-8, surviving SIDM and CDM structure agrees to rtol=1e-3
        and the SIDM core is less than 1e-3 of r_s.

        Failure criteria: any mismatch with the stated value, shape or identity requirements
        fails the test.

        Scope: This controlled calculation does not calibrate a subhalo population against
        simulations or observations.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        cat = self._run(sigma0_m=1.0e-8)
        alive = cat["weightSIDM"] > 0.0
        assert alive.any()
        for sidm, cdm in [
            ("VmaxSIDM_z0", "VmaxCDM_z0"),
            ("rmaxSIDM_z0", "rmaxCDM_z0"),
            ("rsSIDM_z0", "rsCDM_z0"),
            ("rhosSIDM_z0", "rhosCDM_z0"),
        ]:
            assert np.allclose(
                np.asarray(cat[sidm])[alive], np.asarray(cat[cdm])[alive], rtol=1.0e-3, atol=0.0
            )
        rs = np.asarray(cat["rsSIDM_z0"])[alive]
        assert np.all(np.abs(np.asarray(cat["rcSIDM_z0"])[alive]) < 1.0e-3 * rs)
