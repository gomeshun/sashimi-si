"""SIDM cross sections, calibrated profiles, host and tidal physical kernels."""

from ._picard import get_picard_solver

from itamae.evolution import shanks_transform, solve_evolution
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import numexpr as ne


class SIUnits:
    r"""Define the dimensional scales used by SI cross sections, collapse times and halo
    evolution.

    Notes
    -----
    The inherited G=6.6742e-8 cm**3/g/s**2 equals the CODATA-2002 recommended value in cgs
    units. Rounded parsec, solar-mass and speed-of-light conversions are retained
    implementation conventions; this audit does not update their precision.

    SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
    prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
    Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
    rounded SI gravitational constant and its critical-density normalization are retained.

    References
    ----------

    .. [cite-constants] Mohr & Taylor (2005), CODATA recommended values of the fundamental physical constants: 2002, Table XXVI.
        https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf
    """

    def __init__(self):
        r"""Initialize Msun, Mpc and second scales, including the retained rounded SI gravitational
        constant.

        Notes
        -----
        The inherited G=6.6742e-8 cm**3/g/s**2 equals the CODATA-2002 recommended value in cgs
        units. Rounded parsec, solar-mass and speed-of-light conversions are retained
        implementation conventions; this audit does not update their precision.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-constants] Mohr & Taylor (2005), CODATA recommended values of the fundamental physical constants: 2002, Table XXVI.
            https://physics.nist.gov/cuu/pdf/CODATA_RMP2005.pdf
        """
        self.Mpc = 1.0
        self.kpc = self.Mpc / 1000.0
        self.pc = self.kpc / 1000.0
        self.cm = self.pc / 3.086e18
        self.km = 1.0e5 * self.cm
        self.s = 1.0
        self.yr = 3.15576e7 * self.s
        self.Gyr = 1.0e9 * self.yr
        self.Msun = 1.0
        self.gram = self.Msun / 1.988e33
        self.c = 2.9979e10 * self.cm / self.s
        self.G = 6.6742e-8 * self.cm**3 / self.gram / self.s**2


class SIBackgroundKernels(SIUnits):
    r"""Supply the retained flat-LambdaCDM background used by SI fitted halo relations.

    Notes
    -----
    The adopted background Omega_m0=0.315 and h=0.674 (and Omega_b0=0.049 where used) is a
    rounded Planck-2018 parameter choice. It is not evidence that every retained halo fit
    was calibrated in this background.

    SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
    prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
    Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
    rounded SI gravitational constant and its critical-density normalization are retained.

    References
    ----------

    .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
        https://arxiv.org/abs/1807.06209
    """

    def __init__(self):
        r"""Set the adopted SI density fractions, expansion rate and critical-density
        normalization.

        Notes
        -----
        The adopted background Omega_m0=0.315 and h=0.674 (and Omega_b0=0.049 where used) is a
        rounded Planck-2018 parameter choice. It is not evidence that every retained halo fit
        was calibrated in this background.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
            https://arxiv.org/abs/1807.06209
        """
        SIUnits.__init__(self)
        self.OmegaB = 0.049
        self.OmegaM = 0.315
        self.OmegaC = self.OmegaM - self.OmegaB
        self.OmegaL = 1.0 - self.OmegaM
        self.h = 0.674
        self.H0 = self.h * 100 * self.km / self.s / self.Mpc
        self.rhocrit0 = 3 * self.H0**2 / (8.0 * np.pi * self.G)

    def g(self, z):
        """Evaluate the flat matter-plus-Lambda squared expansion factor.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            OmegaM*(1+z)**3+OmegaL, dimensionless.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return self.OmegaM * (1.0 + z) ** 3 + self.OmegaL

    def Hubble(self, z):
        """Evaluate expansion in the SI inverse-second convention.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Expansion rate in inverse seconds, with the redshift shape.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return self.H0 * np.sqrt(self.OmegaM * (1.0 + z) ** 3 + self.OmegaL)

    def rhocrit(self, z):
        """Evaluate critical density using the retained rounded SI gravitational constant.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            3*H(z)**2/(8*pi*G) in Msun/Mpc^3.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return 3.0 * self.Hubble(z) ** 2 / (np.pi * 8.0 * self.G)

    def growthD(self, z):
        r"""Evaluate the retained normalized Carroll-Press-Turner growth approximation.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Dimensionless linear growth normalized at z=0.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """
        Omega_Lz = self.OmegaL / (self.OmegaL + self.OmegaM * (1.0 + z) ** 3)
        Omega_Mz = 1 - Omega_Lz
        phiz = Omega_Mz ** (4.0 / 7.0) - Omega_Lz + (1.0 + Omega_Mz / 2.0) * (1.0 + Omega_Lz / 70.0)
        phi0 = (
            self.OmegaM ** (4.0 / 7.0)
            - self.OmegaL
            + (1.0 + self.OmegaM / 2.0) * (1.0 + self.OmegaL / 70.0)
        )
        return (Omega_Mz / self.OmegaM) * (phi0 / phiz) / (1.0 + z)

    def dDdz(self, z):
        r"""Differentiate the retained Carroll-Press-Turner growth approximation with redshift.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless derivative dD/dz, in the adopted CDM background.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5, Eq. (29).
            https://doi.org/10.1146/annurev.aa.30.090192.002435
            Full text: https://ned.ipac.caltech.edu/level5/Carroll/Carroll3_5.html
        """

        def dOdz(z):
            """Evaluate d Omega_Lambda(z)/dz for the analytic growth derivative.

            Parameters
            ----------
            z : float or array_like
                Dimensionless redshift. The cosmological background requires z > -1; population
                nodes use nonnegative redshifts.

            Returns
            -------
            array_like
                Redshift derivative of the dimensionless dark-energy fraction.

            Notes
            -----
            SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
            prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
            Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
            rounded SI gravitational constant and its critical-density normalization are retained.
            """
            return (
                -self.OmegaL
                * 3
                * self.OmegaM
                * (1 + z) ** 2
                * (self.OmegaL + self.OmegaM * (1 + z) ** 3.0) ** -2
            )

        Omega_Lz = self.OmegaL / (self.OmegaL + self.OmegaM * (1 + z) ** 3)
        Omega_Mz = 1 - Omega_Lz
        phiz = Omega_Mz ** (4.0 / 7.0) - Omega_Lz + (1 + Omega_Mz / 2.0) * (1 + Omega_Lz / 70.0)
        phi0 = (
            self.OmegaM ** (4.0 / 7.0)
            - self.OmegaL
            + (1 + self.OmegaM / 2.0) * (1 + self.OmegaL / 70.0)
        )
        dphidz = dOdz(z) * (
            -4.0 / 7.0 * Omega_Mz ** (-3.0 / 7.0)
            + (Omega_Mz - Omega_Lz) / 140.0
            + 1.0 / 70.0
            - 3.0 / 2.0
        )
        return (phi0 / self.OmegaM) * (
            -dOdz(z) / (phiz * (1 + z))
            - Omega_Mz * (dphidz * (1 + z) + phiz) / phiz**2 / (1 + z) ** 2
        )


class SIHaloKernels(SIBackgroundKernels):
    """Supply SI-owned CDM linear variance, concentration and median host histories.

    Notes
    -----
    SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
    prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
    Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
    rounded SI gravitational constant and its critical-density normalization are retained.
    """

    def __init__(self):
        r"""Initialize the adopted SI background for the halo-history and EPS relations.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        SIBackgroundKernels.__init__(self)

    def xi(self, M):
        r"""Form the inverse halo-mass variable of the Ludlow linear-variance fit.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.

        Returns
        -------
        array_like
            xi=(1e10 Msun/h)/M, dimensionless.

        Notes
        -----
        The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
        fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
        Derivatives are analytic derivatives of this retained fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        return (M / ((1.0e10 * self.Msun) / self.h)) ** -1

    def sigmaMz(self, M, z):
        r"""Evaluate the retained Ludlow CDM rms fluctuation scaled by linear growth.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless sigma(M,z), not the variance S=sigma**2.

        Notes
        -----
        The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
        fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
        Derivatives are analytic derivatives of this retained fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        return (
            self.growthD(z)
            * 22.26
            * self.xi(M) ** 0.292
            / (1.0 + 1.53 * self.xi(M) ** 0.275 + 3.36 * self.xi(M) ** 0.198)
        )

    def deltac_func(self, z):
        r"""Evaluate the linearly extrapolated spherical-collapse barrier for EPS.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            delta_c(z)=1.686/D(z), dimensionless.

        Notes
        -----
        The adopted spherical-collapse barrier is delta_c(z)=1.686/D(z), as used in Ludlow et
        al., Appendix C. This approximation is separate from any mass-dependent WDM collapse
        correction.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        return 1.686 / self.growthD(z)

    def s_func(self, M):
        """Evaluate the present-day linear variance used throughout the EPS kernel.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.

        Returns
        -------
        array_like
            S(M)=sigma(M,0)**2, dimensionless.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return self.sigmaMz(M, 0) ** 2

    def fc(self, x):
        r"""Evaluate the dimensionless NFW enclosed-mass normalization.

        Parameters
        ----------
        x : array_like
            Positive NFW radius ratio r/r_s or concentration used to normalize enclosed mass.

        Returns
        -------
        array_like
            f(x)=ln(1+x)-x/(1+x), with x a radius ratio or concentration.

        Notes
        -----
        The enclosed-mass factor ln(1+c)-c/(1+c) follows the NFW density profile. Algebraic
        inversion tolerances concern this profile identity, not agreement with simulated halos.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107
        """
        return np.log(1 + x) - x * pow(1 + x, -1)

    def Delc(self, x):
        r"""Evaluate the retained virial overdensity relative to critical density.

        Parameters
        ----------
        x : array_like
            Dimensionless Omega_m(z)-1, not a radius ratio.

        Returns
        -------
        array_like
            Delta_c=18*pi**2+82*x-39*x**2 with x=Omega_m(z)-1.

        Notes
        -----
        The flat-universe expression 18*pi**2+82*x-39*x**2, x=Omega_m(z)-1, is Bryan & Norman,
        Eq. (6); it defines overdensity relative to the critical density.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-virial] Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).
            https://arxiv.org/abs/astro-ph/9710107
        """
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    def conc200(self, M200, z):
        r"""Evaluate the retained Correa et al. (2015) median concentration relation.

        Parameters
        ----------
        M200 : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless c200, using the low-z branch through z=4 and the high-z branch above
            it.

        Notes
        -----
        The low/high-redshift concentration coefficients, with the branch at z=4, are the
        Planck-fit formulae in Correa et al., Appendix B1. Evaluation in the adopted
        Planck-2018-like background does not refit these earlier coefficients.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391
        """
        alpha_cMz_1 = 1.7543 - 0.2766 * (1.0 + z) + 0.02039 * (1.0 + z) ** 2
        beta_cMz_1 = 0.2753 + 0.00351 * (1.0 + z) - 0.3038 * (1.0 + z) ** 0.0269
        gamma_cMz_1 = -0.01537 + 0.02102 * (1.0 + z) ** -0.1475
        c_Mz_1 = np.power(
            10.0,
            alpha_cMz_1
            + beta_cMz_1
            * np.log10(M200 / self.Msun)
            * (1 + gamma_cMz_1 * np.log10(M200 / self.Msun) ** 2),
        )
        alpha_cMz_2 = 1.3081 - 0.1078 * (1.0 + z) + 0.00398 * (1.0 + z) ** 2
        beta_cMz_2 = 0.0223 - 0.0944 * (1.0 + z) ** -0.3907
        c_Mz_2 = pow(10, alpha_cMz_2 + beta_cMz_2 * np.log10(M200 / self.Msun))
        return np.where(z <= 4.0, c_Mz_1, c_Mz_2)

    def Mvir_from_M200(self, M200, z):
        r"""Convert M200 to virial mass by solving the same NFW enclosed-density equation.

        Parameters
        ----------
        M200 : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Virial mass in the input mass unit; fsolve uses the M200 radius as its initial
            guess.

        Notes
        -----
        The enclosed-mass factor ln(1+c)-c/(1+c) follows the NFW density profile. Algebraic
        inversion tolerances concern this profile identity, not agreement with simulated halos.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107
        """
        gz = self.g(z)
        c200 = self.conc200(M200, z)
        r200 = (3.0 * M200 / (4 * np.pi * 200 * self.rhocrit0 * gz)) ** (1.0 / 3.0)
        rs = r200 / c200
        fc200 = self.fc(c200)
        rhos = M200 / (4 * np.pi * rs**3 * fc200)
        Dc = self.Delc(self.OmegaM * (1.0 + z) ** 3 / self.g(z) - 1.0)
        rvir = optimize.fsolve(
            lambda r: 3.0 * (rs / r) ** 3 * self.fc(r / rs) * rhos - Dc * self.rhocrit0 * gz, r200
        )
        Mvir = 4 * np.pi * rs**3 * rhos * self.fc(rvir / rs)
        return Mvir

    def Mvir_from_M200_fit(self, M200, z):
        r"""Convert M200 to virial mass with the retained fitted NFW inversion.

        Parameters
        ----------
        M200 : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Virial mass under Delta_c(z), in the input mass unit.

        Notes
        -----
        The inverse mass-conversion approximation, including coefficients 0.5116, -0.4283,
        -3.13e-3 and -3.52e-5 where used, is Hu & Kravtsov, Appendix C. It is an approximate NFW
        inversion, not the exact root-solved conversion.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-conversion] Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.
            https://arxiv.org/abs/astro-ph/0203169
        """
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.OmegaM * (1.0 + z) ** 3 / self.g(z)

        def ffunc(x):
            r"""Express NFW mean enclosed density as a function of inverse concentration.

            Parameters
            ----------
            x : array_like
                Positive inverse NFW concentration r_s/r_delta.

            Returns
            -------
            array_like
                x**3*[ln(1+1/x)-1/(1+x)], with x the inverse concentration.

            Notes
            -----
            The inverse mass-conversion approximation, including coefficients 0.5116, -0.4283,
            -3.13e-3 and -3.52e-5 where used, is Hu & Kravtsov, Appendix C. It is an approximate NFW
            inversion, not the exact root-solved conversion.

            SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
            prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
            Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
            rounded SI gravitational constant and its critical-density normalization are retained.

            References
            ----------

            .. [cite-conversion] Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.
                https://arxiv.org/abs/astro-ph/0203169
            """
            return np.power(x, 3.0) * (np.log(1.0 + 1.0 / x) - 1.0 / (1.0 + x))

        def xfunc(f):
            r"""Invert the dimensionless NFW mean-density mapping with the retained fit.

            Parameters
            ----------
            f : array_like
                Positive dimensionless mean-density mapping at the requested overdensity.

            Returns
            -------
            array_like
                Fitted inverse concentration corresponding to f.

            Notes
            -----
            The inverse mass-conversion approximation, including coefficients 0.5116, -0.4283,
            -3.13e-3 and -3.52e-5 where used, is Hu & Kravtsov, Appendix C. It is an approximate NFW
            inversion, not the exact root-solved conversion.

            SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
            prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
            Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
            rounded SI gravitational constant and its critical-density normalization are retained.

            References
            ----------

            .. [cite-conversion] Hu & Kravtsov (2003), Sample Variance Considerations for Cluster Surveys, Appendix C.
                https://arxiv.org/abs/astro-ph/0203169
            """
            p = a2 + a3 * np.log(f) + a4 * np.power(np.log(f), 2.0)
            return np.power(a1 * np.power(f, 2.0 * p) + (3.0 / 4.0) ** 2, -0.5) + 2.0 * f

        return (
            self.Delc(Oz - 1)
            / 200.0
            * M200
            * np.power(
                self.conc200(M200, z)
                * xfunc(self.Delc(Oz - 1) / 200.0 * ffunc(1.0 / self.conc200(M200, z))),
                -3.0,
            )
        )

    def Mzi(self, M0, z):
        r"""Evaluate the median M200 host history anchored by its z=0 mass.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            M200(z) in the same mass unit as M0.

        Notes
        -----
        The mass-history coefficients, including q = 4.137 / zf**0.9476 and the polynomial for
        zf, follow Correa et al., Section 3. Their original calibration is retained while the
        configured background is used; changing the background does not recalibrate the fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        a = 1.686 * np.sqrt(2.0 / np.pi) * self.dDdz(0) + 1.0
        zf = -0.0064 * np.log10(M0 / self.Msun) ** 2 + 0.0237 * np.log10(M0 / self.Msun) + 1.8837
        q = 4.137 / zf**0.9476
        fM0 = (self.sigmaMz(M0 / q, 0) ** 2 - self.sigmaMz(M0, 0) ** 2) ** -0.5
        return M0 * np.power(1.0 + z, a * fM0) * np.exp(-fM0 * z)

    def Mzzi(self, M0, z, zi):
        r"""Evaluate the re-anchored host history from the reference epoch zi.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        zi : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            M200(z), anchored to Mzi(M0,zi), in the internal mass unit.

        Notes
        -----
        The mass-history coefficients, including q = 4.137 / zf**0.9476 and the polynomial for
        zf, follow Correa et al., Section 3. Their original calibration is retained while the
        configured background is used; changing the background does not recalibrate the fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        Mzi0 = self.Mzi(M0, zi)
        zf = -0.0064 * np.log10(M0 / self.Msun) ** 2 + 0.0237 * np.log10(M0 / self.Msun) + 1.8837
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1.0)
        beta = -fMzi
        return Mzi0 * np.power(1.0 + z - zi, alpha) * np.exp(beta * (z - zi))

    def dMdz(self, M0, z, zi):
        r"""Evaluate the retained virial host-growth factor used by the EPS abundance.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        zi : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            [beta+alpha/(1+z-zi)]*Mvir(z), in mass per unit redshift. The formula retains the
            original fitted virial-growth convention.

        Notes
        -----
        The mass-history coefficients, including q = 4.137 / zf**0.9476 and the polynomial for
        zf, follow Correa et al., Section 3. Their original calibration is retained while the
        configured background is used; changing the background does not recalibrate the fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-correa] Correa et al. (2015), The accretion history of dark matter haloes - III. A physical model for the concentration-mass relation, Section 3 and Appendix B.
            https://arxiv.org/abs/1502.00391

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        Mzi0 = self.Mzi(M0, zi)
        zf = -0.0064 * np.log10(M0 / self.Msun) ** 2 + 0.0237 * np.log10(M0 / self.Msun) + 1.8837
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1)
        beta = -fMzi
        Mzzidef = Mzi0 * (1.0 + z - zi) ** alpha * np.exp(beta * (z - zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef, z)
        return (beta + alpha / (1.0 + z - zi)) * Mzzivir

    def dsdm(self, M, z):
        r"""Differentiate the squared Ludlow rms fluctuation, retaining D(z)**2.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            dS(M,z)/dM in inverse Msun.

        Notes
        -----
        The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
        fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
        Derivatives are analytic derivatives of this retained fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        dsdsigma = 2.0 * self.sigmaMz(M, z)
        dxidm = -1.0e10 * self.Msun / self.h / M**2
        dsigmadxi = self.sigmaMz(M, z) * (
            0.292 / self.xi(M)
            - (0.275 * 1.53 * self.xi(M) ** -0.725 + 0.198 * 3.36 * self.xi(M) ** -0.802)
            / (1.0 + 1.53 * self.xi(M) ** 0.275 + 3.36 * self.xi(M) ** 0.198)
        )
        return dsdsigma * dsigmadxi * dxidm

    def dlogSdlogM(self, M, z):
        r"""Evaluate the logarithmic slope of the linear variance.

        Parameters
        ----------
        M : float or array_like
            Halo M200 in the variant internal mass unit; divide by model.Msun for its numerical
            value in solar masses.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            (M/S)*dS/dM, dimensionless.

        Notes
        -----
        The coefficients 22.26, 1.53, 3.36 and exponents 0.292, 0.275, 0.198 are the variance
        fit in Ludlow et al., Appendix C. Its argument uses M in solar masses divided by h.
        Derivatives are analytic derivatives of this retained fit.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-ludlow] Ludlow et al. (2016), The mass-concentration-redshift relation of cold and warm dark matter haloes, Eq. (3) and Appendix C, Eqs. (C8)-(C12).
            https://arxiv.org/abs/1601.02624
        """
        s = self.sigmaMz(M, z) ** 2
        dsdsigma = 2.0 * self.sigmaMz(M, z)
        dxidm = -1.0e10 * self.Msun / self.h / M**2
        dsigmadxi = self.sigmaMz(M, z) * (
            0.292 * pow(self.xi(M), -1)
            - (0.275 * 1.53 * pow(self.xi(M), -0.725) + 0.198 * 3.36 * pow(self.xi(M), -0.802))
            * pow(1.0 + 1.53 * pow(self.xi(M), 0.275) + 3.36 * pow(self.xi(M), 0.198), -1)
        )
        return (M / s) * dsdsigma * dsigmadxi * dxidm


class SIDM_cross_section(SIUnits):
    r"""Calculate the effective cross section of SIDM.

    Notes
    -----
    The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
    Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
    Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
    switching thresholds are implementation controls.

    - v is the relative velocity between the two initial particles.
    - w is defined as w = m_phi * c / m_chi, where m_phi is the mass of the mediator, c is the speed of light, and m_chi is the mass of the dark matter particle.

    References
    ----------
        - Yang et al. (2023), https://arxiv.org/abs/2305.16176
        - Yang et al. (2022), https://arxiv.org/abs/2205.03392

    .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
        https://arxiv.org/abs/2205.03392

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """

    def __init__(self):
        r"""Set the internal mass, length and time units used by the SIDM scattering kernels.

        Notes
        -----
        The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
        Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
        Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
        switching thresholds are implementation controls.

        This constructor initializes unit conversions. Scattering strength and mediator velocity are supplied to the kernel calls; it does not fit them.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        SIUnits.__init__(self)

    def dsigmadcostheta(self, sigma0_m, w, v, costheta):
        r"""Return the Rutherford differential cross section divided by m.
        Yang et al. (2023), PDF Eq. (1.3) (HTML Eq. (3)), gives

        $$
        \frac{d\sigma}{d\cos\theta} = \frac{\sigma_0 w^4}{2(w^2+v^2\sin^2(\theta/2))^2}
        $$

        Parameters
        ----------
        sigma0_m : float
            Cross section per mass; internal calls use Mpc**2/Msun. The result retains the supplied cross-section units.
        w : float
            Characteristic speed in the same units as v; internal calls use Mpc/s.
        v : float
            Relative speed in the same units as w; internal calls use Mpc/s.

        costheta : array_like
            Dimensionless cosine of the scattering angle, in [-1,1].

        Returns
        -------
        dsigmadcostheta : float
            The value of d\sigma / d\cos\theta divided by m.

        Notes
        -----
        The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
        Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
        Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
        switching thresholds are implementation controls.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        return sigma0_m * w**4 / 2.0 / (w**2 + v**2 / 2.0 * (1.0 - costheta)) ** 2

    def sigma_total(self, sigma0_m, w, v):
        r"""Return the angular integral of the Rutherford differential cross section.

        sigma_total/m = (sigma0/m) / (1 + v**2/w**2). See the text
        following Eq. (2.1) of Yang & Yu (2022), arXiv:2205.03392
        (Eq. (1) in the arXiv HTML). No additional azimuth factor applies
        because dsigmadcostheta is already differential in cos(theta).

        Parameters
        ----------
        sigma0_m : float
            Cross section per mass; internal calls use Mpc**2/Msun. The result retains the supplied cross-section units.
        w : float
            Characteristic speed in the same units as v; internal calls use Mpc/s.
        v : float
            Relative speed in the same units as w; internal calls use Mpc/s.

        Returns
        -------
        sigma_total_m : float
            The value of the total cross section of SIDM divided by m.

        Notes
        -----
        The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
        Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
        Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
        switching thresholds are implementation controls.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        return sigma0_m / (1.0 + v**2 / w**2)

    def sigma_viscosity(self, sigma0_m, w, v):
        r"""Returns the viscosity cross section of SIDM divided by m, given by Eq. (2.7) of Yang et al. (2022) [arXiv:2205.03392].

        Parameters
        ----------
        sigma0_m : float
            Cross section per mass; internal calls use Mpc**2/Msun. The result retains the supplied cross-section units.
        w : float
            Characteristic speed in the same units as v; internal calls use Mpc/s.

        v : numpy.ndarray
            Relative speed in the same internal velocity units as w.

        Returns
        -------
        sigma_V_m : float
            The value of the viscosity cross section of SIDM divided by m.

        Notes
        -----
        The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
        Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
        Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
        switching thresholds are implementation controls.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        sigma_V_m = 6.0 * sigma0_m * w**4 / v**4
        sigma_V_m *= (2.0 * w**2 / v**2 + 1.0) * np.log(1.0 + v**2 / w**2) - 2.0
        v0 = 1.0e-2 * w
        sigma_V_m[v < v0] = 6.0 * sigma0_m * w**4 / v0**4
        sigma_V_m[v < v0] *= (2.0 * w**2 / v0**2 + 1.0) * np.log(1.0 + v0**2 / w**2) - 2.0
        return sigma_V_m

    def sigma_eff_m_interpolate(self, sigma0_m, w):
        r"""Returns the interpolation function of the effective cross section of SIDM divided by m.
        The effective cross section is defined by Eq. (1.1) of Yang et al. (2023) [arXiv:2305.16176]:

        $$
        \sigma_{eff} = \frac{1}{512 \nu_{eff}^8} \int v^2 dv \int d\cos\theta \frac{d\sigma}{d\cos\theta} v^5 \sin^2\theta \exp(-v^2/4\nu_{eff}^2)
        $$

        where $\nu_{eff} = 0.64 V_{max}$ is a characteristic velocity dispersion of dark matter particles in the halo.

        Parameters
        ----------
        sigma0_m : float
            Cross section per mass; internal calls use Mpc**2/Msun. The result retains the supplied cross-section units.
        w : float
            Characteristic speed in internal Mpc/s; convert a km/s value using self.km/self.s.

        Returns
        -------
        sigma_eff_m_interpolate : scipy.interpolate.interp1d
            Interpolator from internal Vmax (Mpc/s) to cross section per mass in the supplied sigma0_m units; its velocity domain is 1e-5 to 1e3 km/s.

        Notes
        -----
        The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
        Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
        Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
        switching thresholds are implementation controls.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        Vmax_dummy = np.logspace(-5.0, 3.0, 1000) * self.km / self.s

        v = np.linspace(0.0, 30.0 * Vmax_dummy, 1000)
        v2 = np.expand_dims(v, axis=-1)
        veff = 0.64 * Vmax_dummy
        veff2 = np.expand_dims(veff, axis=-1)
        costheta = np.linspace(-1.0, 1.0, 100)
        integrand = (
            self.dsigmadcostheta(sigma0_m, w, v2, costheta)
            * v2**7
            * (1.0 - costheta**2)
            * np.exp(-(v2**2) / (4.0 * veff2**2))
        )
        integrand2 = integrate.simpson(integrand, x=costheta, axis=-1)
        sigma_eff_m = integrate.simpson(integrand2, x=v, axis=0)
        sigma_eff_m = sigma_eff_m / (512.0 * veff**8)

        f_int = interp1d(Vmax_dummy, sigma_eff_m)
        return f_int

    def sigma_eff_m_interpolate_analytical(self, sigma0_m, w, a_threshold=703.0, degree=6):
        r"""Returns the analytic evaluation of the effective cross section of SIDM divided by m,
        simplified via the substitution
            a = w^2/(4*nu_eff^2)   with   nu_eff = 0.64 * Vmax.

        The expression is:

            sigma_eff = -sigma0_m * a^2 * [ exp(a) * (1+a) * Ei(-a) + 1 ]

        where Ei(-a) is the exponential integral function.

        For a > a_threshold, a degree-limited asymptotic expansion approaches
        sigma0_m. For 20 <= a <= a_threshold, a positive integral evaluates the
        same expression without subtractive cancellation.

        Parameters
        ----------
        sigma0_m : float
            Cross section per mass; internal calls use Mpc**2/Msun. The result retains the supplied cross-section units.
        w : float
            Characteristic speed in internal Mpc/s; convert a km/s value using self.km/self.s.
        a_threshold : float, optional
            The threshold for switching to the degree-limited asymptotic expansion.
            (Default is 703., based on that Ei(-a) returns NaN for such large a.)
        degree : int, optional
            The degree of the polynomial expansion used for large a.
            (Default is 6, which gives a good approximation for large a.)
        Returns
        -------
        sigma_eff : scipy.interpolate.interp1d
            Interpolator from internal Vmax to effective cross section per mass; its velocity domain is 1e-5 to 1e3 km/s.

        Notes
        -----
        The differential and viscosity cross sections are Yang & Yu, PDF Eqs. (2.1), (2.7).
        Effective averaging uses nu_eff=0.64*Vmax and the 1/(512*nu_eff**8) normalization in
        Yang et al., PDF Eq. (1.1). Finite velocity grids, the low-speed floor and asymptotic
        switching thresholds are implementation controls.

        The exponential-integral closed form is derived by integrating the Rutherford cross
        section in Yang et al. Eq. (1.1); the paper prints the defining integral, not this
        closed form. With a=w**2/(4*nu_eff**2), the ratio is a**2*((1+a)*exp(a)*E1(a)-1). Its
        positive-integral and asymptotic evaluations are numerical implementations of the same
        derived expression.

        References
        ----------

        .. [cite-cross] Yang & Yu (2022), Gravothermal evolution of dark matter halos with differential elastic scattering, PDF Eqs. (2.1), (2.7).
            https://arxiv.org/abs/2205.03392

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        # dummy Vmax for the interpolation
        Vmax = np.logspace(-5.0, 3.0, 1000) * self.km / self.s
        # Compute the effective velocity dispersion
        nu_eff = 0.64 * Vmax
        a = w**2 / (4 * nu_eff**2)
        # For a > a_threshold, return sigma0_m (i.e., effective cross section converges to sigma0_m)
        # sigma_eff_asymp = sigma0_m * (1 - 4/a + 18/a**2 - 96/a**3 + 600/a**4 - 4320/a**5)  # + (-1)^k(k+1)!(k+1)/a^k + ...
        # sigma_eff_asymp = sigma0_m  # Asymptotic behavior for large a
        # NOTE: instead, we use Polynomial expansion for large a
        coeff = [(-1) ** k * special.factorial(k + 1) * (k + 1) for k in range(degree + 1)]
        sigma_eff_asymp = sigma0_m * np.polynomial.Polynomial(coeff)(
            1 / a
        )  # Evaluate the polynomial at 1/a
        sigma_eff = np.array(sigma_eff_asymp, copy=True)
        # At large a the direct expression subtracts nearly equal numbers.
        # Its positive integral representation avoids that cancellation:
        # ratio = integral_0^inf u*exp(-u)/(1+u/a)^2 du.
        direct = (a <= a_threshold) & (a < 20.0)
        sigma_eff[direct] = (
            -sigma0_m
            * a[direct] ** 2
            * (np.exp(a[direct]) * (1 + a[direct]) * special.expi(-a[direct]) + 1)
        )
        stable = (a <= a_threshold) & ~direct
        if np.any(stable):
            nodes, weights = special.roots_genlaguerre(64, 1.0)
            sigma_eff[stable] = sigma0_m * np.sum(
                weights / (1.0 + nodes / a[stable, None]) ** 2,
                axis=-1,
            )
        # Interpolate the result to create a function that can be evaluated at any Vmax
        f_int = interp1d(Vmax, sigma_eff)
        return f_int


class SIDM_parametric_model(SIDM_cross_section):
    r"""A class to calculate the SIDM parametric model proposed by Yang et al. (2023) [arXiv:2305.16176].

    Notes
    -----
    The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
    PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
    in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
    differentiate Eq. (2.4) exactly and may differ from separately rounded printed
    derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
    guarantee.

    The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
    Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
    is not automatically covered by the published calibration.

    References
    ----------
    - Yang et al. (2023), https://arxiv.org/abs/2305.16176

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176

    .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
        https://arxiv.org/abs/2403.16633
    """

    def __init__(self, sigma0_m, w, tt_th=1.1):
        r"""Initialize the SIDM_parametric_model class.

        Parameters
        ----------
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.
        tt_th : float, optional
            Maximum dimensionless evolution time used by the derivative/integral implementation.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
        Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
        is not automatically covered by the published calibration.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633
        """
        SIDM_cross_section.__init__(self)
        self.sigma0_m = sigma0_m * self.cm**2 / self.gram
        self.w = w * self.km / self.s
        self.sigma_eff_m = self.sigma_eff_m_interpolate_analytical(self.sigma0_m, self.w)
        self.tt_th = tt_th

    def t_collapse(self, sigma_eff_m, rmax, Vmax):
        r"""Returns the collapse time of a subhalo according to Eq. (2.2) of Yang et al. (2023)

        Parameters
        ----------
        sigma_eff_m : float
            The effective cross section of SIDM divided by m.
        rmax : float
            The radius at which the maximum circular velocity is obtained.
        Vmax : float
            The maximum circular velocity.

        Returns
        -------
        t_c : float
            The collapse time of a subhalo.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        C = 0.75
        reff = rmax / 2.1626  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        rhoeff = (
            Vmax / (1.648 * reff)
        ) ** 2 / self.G  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        t_c = 150 / C / (sigma_eff_m * rhoeff * reff) / np.sqrt(4.0 * np.pi * self.G * rhoeff)
        return t_c

    def dVmaxSIDMdtt(self, tt, Vmax_CDM):
        r"""Differentiate the maximum circular speed polynomial in Yang et al. Eq. (2.4).

        Parameters
        ----------
        tt : float
            The time normalized by the collapse time.
        Vmax_CDM : float
            The maximum circular velocity of the CDM halo, evaluated at the same time as tt.

        Returns
        -------
        out : float
            The differential of Vmax.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
        Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
        is not automatically covered by the published calibration.

        - In the integral approach, dVmax_{Model}/dtt is normalized by Vmax_{CDM}(t), i.e. the CDM value
          at the running time t of the integral, not the one at the formation-time lower endpoint t_f.

        The polynomial is differentiated from the displayed coefficients of Yang et al. Eq.
        (2.4). For example its velocity derivative has 66.64*tau**3, whereas the separate
        derivative list below Eq. (3.3) prints 66.62*tau**3. The radius derivative likewise
        uses -0.55*tau**3, rather than the printed -0.5502*tau**3. These are distinct
        coefficient lists; agreement with an exact derivative test does not verify the
        separately printed list.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633
        """
        out = (
            0.1777
            - 4.399 * (3.0 * tt**2)
            + 16.66 * (4.0 * tt**3)
            - 18.87 * (5.0 * tt**4)
            + 9.077 * (7.0 * tt**6)
            - 2.436 * (9.0 * tt**8)
        )
        out = np.where(tt > self.tt_th, 0.0, out)
        out = Vmax_CDM * out
        return out

    def dVmaxSIDMdtt_numexpr_optimized(self, tt, Vmax_CDM):
        r"""Differentiate the maximum circular speed polynomial in Yang et al. Eq. (2.4).
        using NumExpr for optimization.
        This optimized version performs the entire calculation in a single evaluation for maximum speed.

        Parameters
        ----------
        tt : np.ndarray
            The time normalized by the collapse time.
        Vmax_CDM : np.ndarray or float
            The maximum circular velocity of the CDM halo, evaluated at the same time as tt.

        Returns
        -------
        out : np.ndarray
            The differential of Vmax.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
        Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
        is not automatically covered by the published calibration.

        The polynomial is differentiated from the displayed coefficients of Yang et al. Eq.
        (2.4). For example its velocity derivative has 66.64*tau**3, whereas the separate
        derivative list below Eq. (3.3) prints 66.62*tau**3. The radius derivative likewise
        uses -0.55*tau**3, rather than the printed -0.5502*tau**3. These are distinct
        coefficient lists; agreement with an exact derivative test does not verify the
        separately printed list.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633
        """
        out = ne.evaluate(
            "where(tt <= tt_th, (0.1777 - 13.197*tt*tt + 66.64*tt*tt*tt"
            " - 94.35*tt*tt*tt*tt + 63.539*tt*tt*tt*tt*tt*tt"
            " - 21.924*tt*tt*tt*tt*tt*tt*tt*tt) * Vmax_CDM, 0)",
            local_dict={"tt": tt, "tt_th": self.tt_th, "Vmax_CDM": Vmax_CDM},
        )
        return out

    def drmaxSIDMdtt(self, tt, rmax_CDM):
        r"""Differentiate the radius of maximum circular speed polynomial in Yang et al. Eq. (2.4).

        Parameters
        ----------
        tt : float
            The time normalized by the collapse time.
        rmax_CDM : float
            The radius at which the maximum circular velocity of the CDM halo is obtained,
            evaluated at the same time as tt.

        Returns
        -------
        out : float
            The differential of rmax.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
        Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
        is not automatically covered by the published calibration.

        - In the integral approach, drmax_{Model}/dtt is normalized by rmax_{CDM}(t), i.e. the CDM value
          at the running time t of the integral, not the one at the formation-time lower endpoint t_f.

        The polynomial is differentiated from the displayed coefficients of Yang et al. Eq.
        (2.4). For example its velocity derivative has 66.64*tau**3, whereas the separate
        derivative list below Eq. (3.3) prints 66.62*tau**3. The radius derivative likewise
        uses -0.55*tau**3, rather than the printed -0.5502*tau**3. These are distinct
        coefficient lists; agreement with an exact derivative test does not verify the
        separately printed list.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633
        """
        out = 0.007623 - 0.7200 * (2.0 * tt) + 0.3376 * (3.0 * tt**2) - 0.1375 * (4.0 * tt**3)
        out = np.where(tt > self.tt_th, 0.0, out)
        out = rmax_CDM * out
        return out

    def drmaxSIDMdtt_numexpr_optimized(self, tt, rmax_CDM):
        r"""Differentiate the radius of maximum circular speed polynomial in Yang et al. Eq. (2.4).
        using NumExpr for optimization.
        This optimized version performs the entire calculation in a single evaluation for maximum speed.

        Parameters
        ----------
        tt : np.ndarray
            The time normalized by the collapse time.
        rmax_CDM : np.ndarray or float
            The radius at which the maximum circular velocity of the CDM halo is obtained,
            evaluated at the same time as tt.

        Returns
        -------
        out : np.ndarray
            The differential of rmax.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
        Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
        is not automatically covered by the published calibration.

        The polynomial is differentiated from the displayed coefficients of Yang et al. Eq.
        (2.4). For example its velocity derivative has 66.64*tau**3, whereas the separate
        derivative list below Eq. (3.3) prints 66.62*tau**3. The radius derivative likewise
        uses -0.55*tau**3, rather than the printed -0.5502*tau**3. These are distinct
        coefficient lists; agreement with an exact derivative test does not verify the
        separately printed list.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633
        """
        out = ne.evaluate(
            "where(tt <= tt_th, (0.007623 - 1.44*tt + 1.0128*tt*tt - 0.55*tt*tt*tt) * rmax_CDM, 0)",
            local_dict={"tt": tt, "tt_th": self.tt_th, "rmax_CDM": rmax_CDM},
        )
        return out

    def get_Vmax0(self, Vmax, tt):
        r"""Returns the maximum circular velocity of the initial NFW profile, given by Eq. (2.4) of Yang et al. (2023)

        Parameters
        ----------
        Vmax : float
            The maximum circular velocity.
        tt : float
            The time normalized by the collapse time.

        Returns
        -------
        Vmax0 : float
            The value of Vmax of the initial NFW profile.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        Vmax0 = Vmax / (
            1.0
            + 0.1777 * tt
            - 4.399 * tt**3
            + 16.66 * tt**4
            - 18.87 * tt**5
            + 9.077 * tt**7
            - 2.436 * tt**9
        )
        return Vmax0

    def get_rmax0(self, rmax, tt):
        r"""Returns the radius at which the maximum circular velocity is obtained, given by Eq. (2.4) of Yang et al. (2023)

        Parameters
        ----------
        rmax : float
            The radius at which the maximum circular velocity is obtained.
        tt : float
            The time normalized by the collapse time.

        Returns
        -------
        rmax0 : float
            The value of rmax of the initial NFW profile.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        rmax0 = rmax / (1.0 + 0.007623 * tt - 0.7200 * tt**2 + 0.3376 * tt**3 - 0.1375 * tt**4)
        return rmax0

    def get_rhos(self, rhos0, tt):
        r"""Returns the density parameter of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ----------
        rhos0 : float
            The density parameter of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        -------
        rhos : float
            The density parameter of a SIDM halo.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        rhos = (
            2.033
            + 0.7381 * tt
            + 7.264 * tt**5
            - 12.73 * tt**7
            + 9.915 * tt**9
            + (1.0 - 2.033) / np.log(0.001) * np.log(tt + 0.001)
        ) * rhos0
        return rhos

    def get_rs(self, rs0, tt):
        r"""Returns the scale radius of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ----------
        rs0 : float
            The scale radius of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        -------
        rs : float
            The scale radius of a SIDM halo.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        rs = (
            0.7178
            - 0.1026 * tt
            + 0.2474 * tt**2
            - 0.4079 * tt**3
            + (1.0 - 0.7178) / np.log(0.001) * np.log(tt + 0.001)
        ) * rs0
        return rs

    def get_rc(self, rs0, tt):
        r"""Returns the core radius of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ----------
        rs0 : float
            The scale radius of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        -------
        rc : float
            The core radius of a SIDM halo.

        Notes
        -----
        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        References
        ----------

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        rc = (
            2.555 * np.sqrt(tt) - 3.632 * tt + 2.131 * tt**2 - 1.415 * tt**3 + 0.4683 * tt**4
        ) * rs0
        return rc

    def _history_integrands(self, Vmax_CDM, rmax_CDM, t, t_f, collapse_time):
        """Evaluate the pointwise SIDM rates shared by nested time histories.

        Parameters
        ----------
        Vmax_CDM, rmax_CDM : numpy.ndarray
            CDM histories, shape (n_concentration, n_time, n_mass), in internal
            Mpc/s and Mpc respectively.
        t : numpy.ndarray
            Cosmic time in seconds, shape (n_time, n_mass).
        t_f : numpy.ndarray
            Formation time in seconds, broadcastable over the mass tracks.
        collapse_time : numpy.ndarray
            Positive collapse time in seconds with the history shape. Infinity
            denotes zero scattering. It must belong to these same histories.

        Returns
        -------
        tuple of numpy.ndarray
            Elapsed time divided by collapse time (dimensionless), the Vmax
            integrand (Mpc/s**2), and the rmax integrand (Mpc/s), all with the
            history shape. Prefix views can be reused only for the same tracks,
            time samples, formation times, and physical parameters.

        Notes
        -----
        These are the existing pointwise rates of the running-CDM integral
        prescription. No quadrature is performed here: full and prefix histories
        still require separate Simpson integrals with their original endpoints.
        The arrays are local to one population slice, not a persistent cache.

        References
        ----------
        Yang et al. (2024), Eqs. (2.4) and (3.3).
        https://arxiv.org/abs/2305.16176
        Ando et al. (2025), Eqs. (2.12)-(2.15).
        https://arxiv.org/abs/2403.16633
        """
        scaled_time = (t - t_f) / collapse_time
        velocity = self.dVmaxSIDMdtt_numexpr_optimized(scaled_time, Vmax_CDM) / collapse_time
        radius = self.drmaxSIDMdtt_numexpr_optimized(scaled_time, rmax_CDM) / collapse_time
        return scaled_time, velocity, radius

    def master_function(self, Vmax_CDM, rmax_CDM, t, t_f, *, collapse_time=None, _integrands=None):
        r"""Evolve SIDM structure with the adopted formation-time integral prescription.

        Parameters
        ----------
        Vmax_CDM : numpy.ndarray
            CDM maximum circular speed in internal Mpc/s, shape (n_concentration, n_time, n_mass).
        rmax_CDM : numpy.ndarray
            CDM radius of maximum circular speed in Mpc, with the same shape as Vmax_CDM.
        t : numpy.ndarray
            Increasing cosmic time in seconds, shape (n_time, n_mass), from formation to observation.
        t_f : array_like
            Formation cosmic time in seconds, broadcastable over the mass tracks; this is the
            lower integration endpoint and the origin of elapsed time, not generally accretion.

        collapse_time : numpy.ndarray, optional
            Precomputed collapse time in seconds, matching the CDM history shape.
            Values must be positive; infinity denotes zero scattering. The caller
            must use the same histories and physical parameters.
        _integrands : tuple of numpy.ndarray, optional
            Internal per-call reuse of ``_history_integrands`` results for these
            exact histories or their prefix. Requires ``collapse_time``. Every
            array must match the CDM history shape; no persistent cache is used.

        Returns
        -------
        VmaxSIDM_z0 : numpy.ndarray
            Final SIDM VmaxSIDM in Mpc/s, shape (n_concentration, n_mass).
        rmaxSIDM_z0 : numpy.ndarray
            Final SIDM rmaxSIDM in Mpc, shape (n_concentration, n_mass).
        rhosSIDM_z0 : numpy.ndarray
            Final SIDM rhosSIDM in Msun/Mpc**3, shape (n_concentration, n_mass).
        rsSIDM_z0 : numpy.ndarray
            Final SIDM rsSIDM in Mpc, shape (n_concentration, n_mass).
        rcSIDM_z0 : numpy.ndarray
            Final SIDM rcSIDM in Mpc, shape (n_concentration, n_mass).

        Raises
        ------
        ValueError
            If a supplied collapse time has the wrong shape, is nonpositive or
            NaN, or if reused integrands lack matching collapse times or shapes.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        The gravothermal calibration is Yang et al.: collapse normalization C=0.75 and 150 in
        PDF Eq. (2.2); density/scale/core-radius fits in Eq. (2.3); velocity/radius polynomials
        in Eq. (2.4); integral evolution in Eq. (3.3). Derivative coefficients here
        differentiate Eq. (2.4) exactly and may differ from separately rounded printed
        derivatives. The tt_th cap is an explicit implementation limit, not an accuracy
        guarantee.

        The default gravothermal cutoff tt_th=1.1 follows the stopped-evolution prescription of
        Ando et al., Section 2.2. A different caller-supplied cap changes that prescription and
        is not automatically covered by the published calibration.

        This implementation combines the running CDM normalization in Yang et al. Eq. (3.3)
        with elapsed time (t-t_f)/t_c(t) and the 1.1 cutoff of Ando et al. Eqs. (2.12)-(2.15).
        As printed, Yang uses tau'=t'/t_c(t'), whereas Ando writes the normalization at t_f.
        The implemented combination is therefore not a literal transcription of either
        displayed integral prescription. Here t_f is the formation-time integration origin,
        which need not equal the accretion time.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (4)-(9), (28), (34).
            https://arxiv.org/abs/1803.07691

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633
        """
        # NOTE: In the equation just below Eq. (3.3) of Yang et al. (2023), dVmax_{Model}/dtt and
        # drmax_{Model}/dtt are normalized by Vmax_{CDM}(t) and rmax_{CDM}(t) evaluated at the running
        # time t of the integral, not by their values at the formation-time lower endpoint t_f.
        if _integrands is not None and collapse_time is None:
            raise ValueError("Reused integrands require the matching collapse_time.")
        if collapse_time is None:
            t_c = self.t_collapse(self.sigma_eff_m(Vmax_CDM), rmax_CDM, Vmax_CDM)
        else:
            t_c = np.asarray(collapse_time)
            if t_c.shape != Vmax_CDM.shape or np.any(np.isnan(t_c)) or np.any(t_c <= 0):
                raise ValueError("collapse_time must match the history and be positive (infinity denotes zero scattering).")
        if collapse_time is not None:
            if _integrands is None:
                _integrands = self._history_integrands(Vmax_CDM, rmax_CDM, t, t_f, t_c)
            if len(_integrands) != 3 or any(np.shape(a) != Vmax_CDM.shape for a in _integrands):
                raise ValueError("Reused integrands must match the CDM history shape.")
            scaled_time, velocity_integrand, radius_integrand = _integrands
            # Preserve Simpson's arithmetic and endpoint corrections separately
            # for the full history and any prefix; only pointwise rates are reused.
            # Keep the legacy grid shape and memory order: a singleton broadcast
            # can change NumPy's reduction order inside Simpson for F-order t.
            time_grid = t * np.ones((len(Vmax_CDM), 1, 1))
            VmaxSIDM_z0 = Vmax_CDM[:, -1] + integrate.simpson(
                velocity_integrand, x=time_grid, axis=1
            )
            rmaxSIDM_z0 = rmax_CDM[:, -1] + integrate.simpson(
                radius_integrand, x=time_grid, axis=1
            )
            tt = np.minimum(scaled_time[:, -1], self.tt_th)
        else:
            integrand = self.dVmaxSIDMdtt_numexpr_optimized((t - t_f) / t_c, Vmax_CDM) / t_c
            VmaxSIDM_z0 = Vmax_CDM[:, -1] + integrate.simpson(
                integrand, x=t * np.ones((len(Vmax_CDM), 1, 1)), axis=1
            )
            integrand = self.drmaxSIDMdtt_numexpr_optimized((t - t_f) / t_c, rmax_CDM) / t_c
            rmaxSIDM_z0 = rmax_CDM[:, -1] + integrate.simpson(
                integrand, x=t * np.ones((len(rmax_CDM), 1, 1)), axis=1
            )

            tt = np.minimum(((t - t_f) / t_c)[:, -1], self.tt_th)
        Vmax0_CDM_fict = self.get_Vmax0(VmaxSIDM_z0, tt)
        rmax0_CDM_fict = self.get_rmax0(rmaxSIDM_z0, tt)
        rs0_CDM_fict = rmax0_CDM_fict / 2.1626
        rhos0_CDM_fict = (4.625 / (4.0 * np.pi * self.G)) * (Vmax0_CDM_fict / rs0_CDM_fict) ** 2

        rhosSIDM_z0 = self.get_rhos(rhos0_CDM_fict, tt)
        rsSIDM_z0 = self.get_rs(rs0_CDM_fict, tt)
        rcSIDM_z0 = self.get_rc(rs0_CDM_fict, tt)

        return VmaxSIDM_z0, rmaxSIDM_z0, rhosSIDM_z0, rsSIDM_z0, rcSIDM_z0


class SITidalKernels(SIHaloKernels):
    """Solve the tidal stripping equation for a given subhalo.

    Parameters
    ----------
    M0 : float
        Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
        explicit epoch flags are described by the caller.
    z_min : float
        Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
    z_max : float
        Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
    n_z_interp : int
        Redshift-node count for the retained perturbative coefficient
        integrals/interpolants; independent of Picard endpoint-table settings.

    Notes
    -----
    SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
    prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
    Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
    rounded SI gravitational constant and its critical-density normalization are retained.
    """

    def __init__(self, M0, z_min=0.0, z_max=7.0, n_z_interp=64):
        """Initialize the SI host-dependent tidal law and retained perturbative interpolants.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z_min : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        z_max : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        n_z_interp : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        SIHaloKernels.__init__(self)
        self.z_min = z_min
        self.z_max = z_max
        self.n_z_interp = n_z_interp
        self.M0 = M0

    @property
    def M0(self):
        """Return the z=0 host M200 anchoring the current tidal history.

        Returns
        -------
        float
            Reference host mass in the internal solar-mass unit.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return self._M0

    @M0.setter
    def M0(self, value):
        """Change the reference host M200 and rebuild the SI epsilon interpolants.

        Parameters
        ----------
        value : float
            New z=0 host M200 in internal solar-mass units.

        Notes
        -----
        The retained perturbative coefficients are recomputed immediately. Shared Picard caches
        additionally compare the complete physical-state key before reuse.
        """
        self._M0 = value
        self.reset_interpolation(z_max=self.z_max, z_min=self.z_min, n_z=self.n_z_interp)

    def reset_interpolation(self, z_max, z_min, n_z):
        """Rebuild the retained epsilon integrals and interpolation functions for the current host.

        This function is called when the mass of the host
        halo is changed.

        Parameters
        ----------
        z_max : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        z_min : float
            Lower or upper redshift extent declared by the caller; require -1 < z_min <= z_max.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        _z, _eps_0 = self._eps_0(z_max, z_min, n_z)
        _, _eps_10, _eps_11 = self._eps_1(z_max, z_min, n_z)
        _, _eps_20, _eps_21, _eps_22 = self._eps_2(z_max, z_min, n_z)
        _, _eps_30, _eps_31, _eps_32, _eps_33 = self._eps_3(z_max, z_min, n_z)
        # get the interpolation functions as indefinite integrals
        self._eps_0_interp = lambda z: np.interp(z, _z[::-1], _eps_0[::-1])
        self._eps_10_interp = lambda z: np.interp(z, _z[::-1], _eps_10[::-1])
        self._eps_11_interp = lambda z: np.interp(z, _z[::-1], _eps_11[::-1])
        self._eps_20_interp = lambda z: np.interp(z, _z[::-1], _eps_20[::-1])
        self._eps_21_interp = lambda z: np.interp(z, _z[::-1], _eps_21[::-1])
        self._eps_22_interp = lambda z: np.interp(z, _z[::-1], _eps_22[::-1])
        self._eps_30_interp = lambda z: np.interp(z, _z[::-1], _eps_30[::-1])
        self._eps_31_interp = lambda z: np.interp(z, _z[::-1], _eps_31[::-1])
        self._eps_32_interp = lambda z: np.interp(z, _z[::-1], _eps_32[::-1])
        self._eps_33_interp = lambda z: np.interp(z, _z[::-1], _eps_33[::-1])
        # define the epsilon functions as definite integrals from za to z
        self.eps_0 = lambda _za, _z: self._eps_0_interp(_z) - self._eps_0_interp(_za)
        self.eps_10 = lambda _za, _z: self._eps_10_interp(_z) - self._eps_10_interp(_za)
        self.eps_11 = lambda _za, _z: self._eps_11_interp(_z) - self._eps_11_interp(_za)
        self.eps_20 = lambda _za, _z: self._eps_20_interp(_z) - self._eps_20_interp(_za)
        self.eps_21 = lambda _za, _z: self._eps_21_interp(_z) - self._eps_21_interp(_za)
        self.eps_22 = lambda _za, _z: self._eps_22_interp(_z) - self._eps_22_interp(_za)
        self.eps_30 = lambda _za, _z: self._eps_30_interp(_z) - self._eps_30_interp(_za)
        self.eps_31 = lambda _za, _z: self._eps_31_interp(_z) - self._eps_31_interp(_za)
        self.eps_32 = lambda _za, _z: self._eps_32_interp(_z) - self._eps_32_interp(_za)
        self.eps_33 = lambda _za, _z: self._eps_33_interp(_z) - self._eps_33_interp(_za)

    def Mzvir(self, z):
        """Evaluate the virial host history used in the subhalo-to-host mass ratio.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Virial host mass at z, in internal solar-mass units.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        Mz200 = self.Mzzi(self.M0, z, 0.0)
        Mvir = self.Mvir_from_M200_fit(Mz200, z)
        return Mvir

    def AMz(self, z):
        r"""Evaluate the retained SI tidal-loss amplitude as a function of virial host mass and
        redshift.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            A(M_host,z) from the retained host-mass/redshift fit.

        Notes
        -----
        The adopted A(M,z) and zeta(M,z) coefficients are Hiroshima et al., Eqs. (23)-(24),
        calibrated for host masses 1e-6 to 1e16 solar masses and redshifts 0 to 7. Numerical
        integration or iteration controls do not extend that calibration or bound total model
        error.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-tidal] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Eqs. (23)-(24).
            https://arxiv.org/abs/1803.07691
        """
        log10a = (-0.0003 * np.log10(self.Mzvir(z) / self.Msun) + 0.02) * z + (
            0.011 * np.log10(self.Mzvir(z) / self.Msun) - 0.354
        )
        return 10.0**log10a

    def zetaMz(self, z):
        r"""Evaluate the fitted exponent of the tidal subhalo-to-host mass ratio.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless zeta(M_host,z).

        Notes
        -----
        The adopted A(M,z) and zeta(M,z) coefficients are Hiroshima et al., Eqs. (23)-(24),
        calibrated for host masses 1e-6 to 1e16 solar masses and redshifts 0 to 7. Numerical
        integration or iteration controls do not extend that calibration or bound total model
        error.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-tidal] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Eqs. (23)-(24).
            https://arxiv.org/abs/1803.07691
        """
        return (0.00012 * np.log10(self.Mzvir(z) / self.Msun) - 0.0033) * z + (
            -0.0011 * np.log10(self.Mzvir(z) / self.Msun) + 0.026
        )

    def tdynz(self, z):
        r"""Evaluate the virial dynamical time entering the fitted tidal loss law.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dynamical time in internal seconds, scaled by Delta_c(z) and H(z)/H0.

        Notes
        -----
        The normalization 1.628/h Gyr and factors (Delta_vir/178)**(-1/2)*(H/H0)**(-1) are Jiang
        & van den Bosch, Eq. (2), with Bryan--Norman virial overdensity.

        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.

        References
        ----------

        .. [cite-time] Jiang & van den Bosch (2016), Statistics of Dark Matter Substructure: I. Model and Universal Fitting Functions, Eq. (2).
            https://arxiv.org/abs/1403.6827

        .. [cite-virial] Bryan & Norman (1998), Statistical Properties of X-ray Clusters: Analytic and Numerical Comparisons, Eq. (6).
            https://arxiv.org/abs/astro-ph/9710107
        """
        Oz_z = self.OmegaM * (1.0 + z) ** 3 / self.g(z)
        return (
            1.628
            / self.h
            * (self.Delc(Oz_z - 1.0) / 178.0) ** -0.5
            / (self.Hubble(z) / self.H0)
            * 1.0e9
            * self.yr
        )

    def msolve(self, m, z):
        """Evaluate the positive redshift derivative of the bound subhalo mass.

        Parameters
        ----------
        m : float or array_like
            Subhalo mass in the same internal unit as the host mass; masses must be positive for
            logarithmic tidal evolution.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            dm/dz=A*m*(m/M_host)**zeta/[t_dyn*H*(1+z)], in mass per unit redshift.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return (
            self.AMz(z)
            * (m / self.tdynz(z))
            * (m / self.Mzvir(z)) ** self.zetaMz(z)
            / (self.Hubble(z) * (1 + z))
        )

    def subhalo_mass_stripped_odeint(self, ma, za, z0, **kwargs):
        """Integrate the owned tidal RHS through ITAMAE's odeint controller.

        Numerical options retain SciPy meanings. A supplied Dfun keeps this
        method's historical state-first signature; ITAMAE receives time first.
        The controller owns args/tfirst/full_output, which are not accepted as
        duplicate overrides. Repeated times and zero evolution are preserved.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z0 : float or array_like
            Scalar final redshift or explicit output-redshift array. With an array the first
            entry supplies the odeint initial epoch; include za as the first coordinate.
        **kwargs : dict
            Controls for the selected Picard or ODE backend. Historical odeint Dfun callbacks
            use state-first signature; unsupported controller overrides are rejected.

        Returns
        -------
        float or numpy.ndarray
            Endpoint mass for a scalar target, or (n_redshift,n_mass) history for vector
            targets, in internal Msun.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        options = dict(kwargs)
        rtol, atol = options.pop("rtol", None), options.pop("atol", None)
        jacobian = options.get("Dfun")
        if callable(jacobian):
            options["Dfun"] = lambda time, state: jacobian(state, time)
        scalar_output = np.isscalar(z0)
        zcalc = np.linspace(za, z0, 100) if scalar_output else np.asarray(z0)
        solution = solve_evolution(
            lambda time, state: self.msolve(state, time),
            ma,
            zcalc,
            method="odeint",
            rtol=rtol,
            atol=atol,
            odeint_options=options,
            allow_repeated_times=True,
        )
        return solution[-1] if scalar_output else solution

    def Phi(self, z):
        """Evaluate the positive redshift coefficient A/[t_dyn*H*(1+z)] in the SI tidal law.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            A/[t_dyn*H*(1+z)], dimensionless per unit redshift.

        Notes
        -----
        Phi includes dt/dz in magnitude. Evolving toward decreasing redshift therefore decreases
        mass; the physical cosmic-time law is dm/dt=-A*m*(m/M_host)**zeta/t_dyn.
        """
        return self.AMz(z) / self.tdynz(z) / self.Hubble(z) / (1 + z)

    # @memoize_with_pickle()
    def _eps_0(self, za, z, n_z=64):
        """Integrate the order-0 coefficients of the retained log-mass perturbation expansion.

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps0 : array
            epsilon0 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z = np.linspace(za, z, n_z)
        Phi_z = self.Phi(_z)
        return _z, cumulative_trapezoid(Phi_z, x=_z, initial=0)

    # @memoize_with_pickle()
    def _eps_1(self, za, z, n_z=64):
        """Integrate the order-1 coefficients of the retained log-mass perturbation expansion.

        The first order correction epsilon_1 is given by the following equation:
            epsilon_1 = epsilon_10 + epsilon_11 * ln_ma

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps10 : array
            epsilon10 array.
        eps11 : array
            epsilon11 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_10 = Phi_z * (eps_0 - ln_Mvir_z) * zeta_z
        integrand_11 = Phi_z * zeta_z
        integral_10 = cumulative_trapezoid(integrand_10, x=_z, initial=0)
        integral_11 = cumulative_trapezoid(integrand_11, x=_z, initial=0)
        return _z, integral_10, integral_11

    # @memoize_with_pickle()
    def _eps_2(self, za, z, n_z=64):
        """Integrate the order-2 coefficients of the retained log-mass perturbation expansion.

        The second order correction epsilon_2 is given by the following equation:
            epsilon_2 = epsilon_20 + epsilon_21 * ln_ma + epsilon_22 * ln_ma^2

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps20 : array
            epsilon20 array.
        eps21 : array
            epsilon21 array.
        eps22 : array
            epsilon22 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_20 = Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) ** 2 / 2 + Phi_z * zeta_z * eps_10
        integrand_21 = Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) + Phi_z * zeta_z * eps_11
        integrand_22 = Phi_z * zeta_z**2 / 2
        integral_20 = cumulative_trapezoid(integrand_20, x=_z, initial=0)
        integral_21 = cumulative_trapezoid(integrand_21, x=_z, initial=0)
        integral_22 = cumulative_trapezoid(integrand_22, x=_z, initial=0)
        return _z, integral_20, integral_21, integral_22

    # @memoize_with_pickle()
    def _eps_3(self, za, z, n_z=64):
        """Integrate the order-3 coefficients of the retained log-mass perturbation expansion.

        The third order correction epsilon_3 is given by the following equation:
            epsilon_3 = epsilon_30 + epsilon_31 * ln_ma + epsilon_32 * ln_ma^2 + epsilon_33 * ln_ma^3

        Parameters
        ----------
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        n_z : int
            Redshift-node count for the retained perturbative coefficient
            integrals/interpolants; independent of Picard endpoint-table settings.

        Returns
        -------
        _z : array
            redshift array
        eps30 : array
            epsilon30 array.
        eps31 : array
            epsilon31 array.
        eps32 : array
            epsilon32 array.
        eps33 : array
            epsilon33 array.

        Notes
        -----
        The output redshift grid runs from za to z. Cumulative trapezoidal integrals start at
        zero at accretion. Polynomial coefficients multiply powers of ln(ma) in the fixed
        internal mass unit; they are coupled terms, not separately physical mass losses.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        _, eps_20, eps_21, eps_22 = self._eps_2(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_30 = (
            Phi_z * (eps_0 - ln_Mvir_z) ** 3 * zeta_z**3 / 6.0
            + Phi_z * (eps_0 - ln_Mvir_z) * eps_10 * (zeta_z**2)
            + Phi_z * eps_20 * zeta_z
        )
        integrand_31 = (
            Phi_z * (eps_0 - ln_Mvir_z) ** 2 * (zeta_z**3) / 2.0
            + Phi_z * eps_10 * (zeta_z**2)
            + Phi_z * eps_21 * zeta_z
            + Phi_z * (eps_0 - ln_Mvir_z) * eps_11 * (zeta_z**2)
        )
        integrand_32 = (
            Phi_z * (eps_0 - ln_Mvir_z) * (zeta_z**3) / 2.0
            + Phi_z * eps_11 * (zeta_z**2)
            + Phi_z * eps_22 * zeta_z
        )
        integrand_33 = Phi_z * (zeta_z**3) / 6.0
        integral_30 = cumulative_trapezoid(integrand_30, x=_z, initial=0)
        integral_31 = cumulative_trapezoid(integrand_31, x=_z, initial=0)
        integral_32 = cumulative_trapezoid(integrand_32, x=_z, initial=0)
        integral_33 = cumulative_trapezoid(integrand_33, x=_z, initial=0)
        return _z, integral_30, integral_31, integral_32, integral_33

    def subhalo_mass_stripped_pert0(self, ma, za, z):
        """Calculate subhalo mass stripping using zeroth-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-0 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        eps_0 = self.eps_0(za, z)
        return ma * np.exp(eps_0)

    def subhalo_mass_stripped_pert1(self, ma, za, z):
        """Calculate subhalo mass stripping using first-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-1 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        ln_ma = np.log(ma)
        eps = eps_0 + eps_10 + ln_ma * eps_11
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert2(self, ma, za, z):
        """Calculate subhalo mass stripping using second-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-2 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        eps = eps_0 + eps_10 + ln_ma * eps_11 + eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert2_shanks(self, ma, za, z):
        """Evaluate second-order stripping with ITAMAE Shanks acceleration.

        The SASHIMI-specific rule that disables acceleration for corrections
        smaller than two percent is preserved.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound masses from the retained second-order sequence with Shanks acceleration except
            for the prescribed small-correction branch.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """

        eps_0 = self.eps_0(za, z)
        ln_ma = np.log(ma)
        eps_1 = self.eps_10(za, z) + ln_ma * self.eps_11(za, z)
        eps_2 = self.eps_20(za, z) + ln_ma * self.eps_21(za, z) + ln_ma**2 * self.eps_22(za, z)
        partial_0 = eps_0
        partial_1 = eps_0 + eps_1
        partial_2 = partial_1 + eps_2
        accelerated = shanks_transform(partial_0, partial_1, partial_2)
        with np.errstate(divide="ignore", invalid="ignore"):
            small_correction = np.abs((eps_1 + eps_2) / eps_0) < 0.02
        eps = np.where(small_correction, partial_2, accelerated)
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert3(self, ma, za, z):
        """Calculate subhalo mass stripping using third-order perturbation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Bound mass from the truncated order-3 log-mass expansion, in the accretion mass
            unit.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        eps_30 = self.eps_30(za, z)
        eps_31 = self.eps_31(za, z)
        eps_32 = self.eps_32(za, z)
        eps_33 = self.eps_33(za, z)
        ln_ma = np.log(ma)
        eps = (
            eps_0
            + eps_10
            + ln_ma * eps_11
            + eps_20
            + ln_ma * eps_21
            + ln_ma**2 * eps_22
            + eps_30
            + ln_ma * eps_31
            + ln_ma**2 * eps_32
            + ln_ma**3 * eps_33
        )
        return ma * np.exp(eps)

    def subhalo_mass_stripped(self, ma, za, z, method="picard", **kwargs):
        """Evaluate an SI tidal endpoint or full mass history with the explicitly selected solver.

        Parameters
        ----------
        ma : float
            initial subhalo mass.
        za : float
            initial redshift.
        z : float or array_like
            Scalar final redshift or one-dimensional non-increasing history at or below za.
        method : str
            picard (default), dop853, odeint, or an explicitly selected retained perturbative
            approximation.
        **kwargs : dict
            Controls for the selected shared Picard/DOP853 backend or historical state-first
            odeint interface.

        Returns
        -------
        float or numpy.ndarray
            Endpoint mass for a scalar target, or (n_redshift,n_mass) history for vector
            targets, in internal Msun.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        if method == "picard":
            return get_picard_solver(self).history(ma, za, z, **kwargs)
        if method == "dop853":
            return get_picard_solver(self).direct(ma, za, z, **kwargs)
        if method == "odeint":
            # NOTE: odeint returns (len(z),len(ma)) array for array input.
            return self.subhalo_mass_stripped_odeint(ma, za, z, **kwargs)
        elif method[:4] == "pert":
            # When z and ma are given as 1d arrays, perturbative methods raise error.
            # To return the similar output as odeint, we broadcast the input arrays.
            if np.isscalar(z) and np.isscalar(ma):
                return getattr(self, f"subhalo_mass_stripped_{method}")(ma, za, z)
            else:
                ma = np.atleast_1d(ma)
                z = np.atleast_1d(z)
                ma = ma[np.newaxis, :]  # (1, len(ma))
                z = z[:, np.newaxis]  # (len(z), 1)
                return getattr(self, f"subhalo_mass_stripped_{method}")(ma, za, z)
        else:
            raise ValueError(f"Invalid method: {method}")
