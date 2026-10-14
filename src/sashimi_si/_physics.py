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
    """Define the dimensional scales used by SI cross sections, collapse times and halo
    evolution.

    Notes
    -----
    SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
    prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
    Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
    rounded SI gravitational constant and its critical-density normalization are retained.
    """

    def __init__(self):
        """Initialize Msun, Mpc and second scales, including the retained rounded SI gravitational
        constant.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
    """Supply the retained flat-LambdaCDM background used by SI fitted halo relations.

    Notes
    -----
    SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
    prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
    Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
    rounded SI gravitational constant and its critical-density normalization are retained.
    """

    def __init__(self):
        """Set the SI calibrated density fractions, expansion rate and critical-density
        normalization.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Evaluate the retained normalized Carroll-Press-Turner growth approximation.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Differentiate the retained Carroll-Press-Turner growth approximation with redshift.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        array_like
            Dimensionless derivative dD/dz, in the calibrated CDM background.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Initialize the calibrated SI background for the halo-history and EPS relations.

        Notes
        -----
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        SIBackgroundKernels.__init__(self)

    def xi(self, M):
        """Form the inverse halo-mass variable of the Ludlow linear-variance fit.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return (M / ((1.0e10 * self.Msun) / self.h)) ** -1

    def sigmaMz(self, M, z):
        """Evaluate the retained Ludlow CDM rms fluctuation scaled by linear growth.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return (
            self.growthD(z)
            * 22.26
            * self.xi(M) ** 0.292
            / (1.0 + 1.53 * self.xi(M) ** 0.275 + 3.36 * self.xi(M) ** 0.198)
        )

    def deltac_func(self, z):
        """Evaluate the linearly extrapolated spherical-collapse barrier for EPS.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Evaluate the dimensionless NFW enclosed-mass normalization.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return np.log(1 + x) - x * pow(1 + x, -1)

    def Delc(self, x):
        """Evaluate the retained virial overdensity relative to critical density.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    def conc200(self, M200, z):
        """Evaluate the retained Correa et al. (2015) median concentration relation.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Convert M200 to virial mass by solving the same NFW enclosed-density equation.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Convert M200 to virial mass with the retained fitted NFW inversion.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.OmegaM * (1.0 + z) ** 3 / self.g(z)

        def ffunc(x):
            """Express NFW mean enclosed density as a function of inverse concentration.

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
            SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
            prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
            Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
            rounded SI gravitational constant and its critical-density normalization are retained.
            """
            return np.power(x, 3.0) * (np.log(1.0 + 1.0 / x) - 1.0 / (1.0 + x))

        def xfunc(f):
            """Invert the dimensionless NFW mean-density mapping with the retained fit.

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
            SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
            prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
            Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
            rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Evaluate the median M200 host history anchored by its z=0 mass.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        a = 1.686 * np.sqrt(2.0 / np.pi) * self.dDdz(0) + 1.0
        zf = -0.0064 * np.log10(M0 / self.Msun) ** 2 + 0.0237 * np.log10(M0 / self.Msun) + 1.8837
        q = 4.137 / zf**0.9476
        fM0 = (self.sigmaMz(M0 / q, 0) ** 2 - self.sigmaMz(M0, 0) ** 2) ** -0.5
        return M0 * np.power(1.0 + z, a * fM0) * np.exp(-fM0 * z)

    def Mzzi(self, M0, z, zi):
        """Evaluate the re-anchored host history from the reference epoch zi.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        Mzi0 = self.Mzi(M0, zi)
        zf = -0.0064 * np.log10(M0 / self.Msun) ** 2 + 0.0237 * np.log10(M0 / self.Msun) + 1.8837
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1.0)
        beta = -fMzi
        return Mzi0 * np.power(1.0 + z - zi, alpha) * np.exp(beta * (z - zi))

    def dMdz(self, M0, z, zi):
        """Evaluate the retained virial host-growth factor used by the EPS abundance.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Differentiate the squared Ludlow rms fluctuation, retaining D(z)**2.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
        """Evaluate the logarithmic slope of the linear variance.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
    """Calculate the effective cross section of SIDM.

    Notes
    ---
    - v is the relative velocity between the two initial particles.
    - w is defined as w = m_phi * c / m_chi, where m_phi is the mass of the mediator, c is the speed of light, and m_chi is the mass of the dark matter particle.

    References
    ---
        - Yang et al. (2023), https://arxiv.org/abs/2305.16176
        - Yang et al. (2022), https://arxiv.org/abs/2205.03392
    """

    def __init__(self):
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
            The value of sigma_0 / m.
        w : float
            The value of w in unit of km/s.
        v : float
            The value of v in unit of km/s.

        Returns
        -------
        dsigmadcostheta : float
            The value of d\sigma / d\cos\theta divided by m.

        """
        return sigma0_m * w**4 / 2.0 / (w**2 + v**2 / 2.0 * (1.0 - costheta)) ** 2

    def sigma_total(self, sigma0_m, w, v):
        """Return the angular integral of the Rutherford differential cross section.

        sigma_total/m = (sigma0/m) / (1 + v**2/w**2). See the text
        following Eq. (2.1) of Yang & Yu (2022), arXiv:2205.03392
        (Eq. (1) in the arXiv HTML). No additional azimuth factor applies
        because dsigmadcostheta is already differential in cos(theta).

        Parameters
        ---
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.
        v : float
            The value of v in unit of km/s.

        Returns
        ---
        sigma_total_m : float
            The value of the total cross section of SIDM divided by m.
        """
        return sigma0_m / (1.0 + v**2 / w**2)

    def sigma_viscosity(self, sigma0_m, w, v):
        """Returns the viscosity cross section of SIDM divided by m, given by Eq. (2.7) of Yang et al. (2022) [arXiv:2205.03392].

        Parameters
        ---
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.

        Returns
        ---
        sigma_V_m : float
            The value of the viscosity cross section of SIDM divided by m.
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
            The value of sigma_0 / m.
        w : float
            The value of w.

        Returns
        -------
        sigma_eff_m_interpolate : scipy.interpolate.interp1d
            The interpolation function of the effective cross section of SIDM divided by m.
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
        """
        Returns the analytic evaluation of the effective cross section of SIDM divided by m,
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
            The value of sigma_0 / m.
        w : float
            The value of w (in the same units as used in the numerical methods).
        a_threshold : float, optional
            The threshold for switching to the degree-limited asymptotic expansion.
            (Default is 703., based on that Ei(-a) returns NaN for such large a.)
        degree : int, optional
            The degree of the polynomial expansion used for large a.
            (Default is 6, which gives a good approximation for large a.)
        Returns
        -------
        sigma_eff : float or np.ndarray
            The effective cross section of SIDM divided by m.
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
    """A class to calculate the SIDM parametric model proposed by Yang et al. (2023) [arXiv:2305.16176].

    Refereces
    ---
    - Yang et al. (2023), https://arxiv.org/abs/2305.16176
    """

    def __init__(self, sigma0_m, w, tt_th=1.1):
        """Initialize the SIDM_parametric_model class.

        Parameters
        ---
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.
        """
        SIDM_cross_section.__init__(self)
        self.sigma0_m = sigma0_m * self.cm**2 / self.gram
        self.w = w * self.km / self.s
        self.sigma_eff_m = self.sigma_eff_m_interpolate_analytical(self.sigma0_m, self.w)
        self.tt_th = tt_th

    def t_collapse(self, sigma_eff_m, rmax, Vmax):
        """Returns the collapse time of a subhalo according to Eq. (2.2) of Yang et al. (2023)

        Parameters
        ---
        sigma_eff_m : float
            The effective cross section of SIDM divided by m.
        rmax : float
            The radius at which the maximum circular velocity is obtained.
        Vmax : float
            The maximum circular velocity.

        Returns
        ---
        t_c : float
            The collapse time of a subhalo.
        """
        C = 0.75
        reff = rmax / 2.1626  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        rhoeff = (
            Vmax / (1.648 * reff)
        ) ** 2 / self.G  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        t_c = 150 / C / (sigma_eff_m * rhoeff * reff) / np.sqrt(4.0 * np.pi * self.G * rhoeff)
        return t_c

    def dVmaxSIDMdtt(self, tt, Vmax_CDM):
        """Returns differential of Vmax obtained by the equation just below Eq. (3.3) of Yang et al. (2023)

        Parameters
        ---
        tt : float
            The time normalized by the collapse time.
        Vmax_CDM : float
            The maximum circular velocity of the CDM halo, evaluated at the same time as tt.

        Returns
        ---
        out : float
            The differential of Vmax.

        Notes
        ---
        - In the integral approach, dVmax_{Model}/dtt is normalized by Vmax_{CDM}(t), i.e. the CDM value
          at the running time t of the integral, not the one at the accretion time t_f.
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
        """
        Returns differential of Vmax obtained by the equation just below Eq. (3.3) of Yang et al. (2023),
        using NumExpr for optimization.
        This optimized version performs the entire calculation in a single evaluation for maximum speed.

        Parameters
        ---
        tt : np.ndarray
            The time normalized by the collapse time.
        Vmax_CDM : np.ndarray or float
            The maximum circular velocity of the CDM halo, evaluated at the same time as tt.

        Returns
        ---
        out : np.ndarray
            The differential of Vmax.
        """
        out = ne.evaluate(
            "where(tt <= tt_th, (0.1777 - 13.197*tt*tt + 66.64*tt*tt*tt"
            " - 94.35*tt*tt*tt*tt + 63.539*tt*tt*tt*tt*tt*tt"
            " - 21.924*tt*tt*tt*tt*tt*tt*tt*tt) * Vmax_CDM, 0)",
            local_dict={"tt": tt, "tt_th": self.tt_th, "Vmax_CDM": Vmax_CDM},
        )
        return out

    def drmaxSIDMdtt(self, tt, rmax_CDM):
        """Returns differential of rmax obtained by the equation just below Eq. (3.3) of Yang et al. (2023)

        Parameters
        ---
        tt : float
            The time normalized by the collapse time.
        rmax_CDM : float
            The radius at which the maximum circular velocity of the CDM halo is obtained,
            evaluated at the same time as tt.

        Returns
        ---
        out : float
            The differential of rmax.

        Notes
        ---
        - In the integral approach, drmax_{Model}/dtt is normalized by rmax_{CDM}(t), i.e. the CDM value
          at the running time t of the integral, not the one at the accretion time t_f.
        """
        out = 0.007623 - 0.7200 * (2.0 * tt) + 0.3376 * (3.0 * tt**2) - 0.1375 * (4.0 * tt**3)
        out = np.where(tt > self.tt_th, 0.0, out)
        out = rmax_CDM * out
        return out

    def drmaxSIDMdtt_numexpr_optimized(self, tt, rmax_CDM):
        """
        Returns differential of rmax obtained by the equation just below Eq. (3.3) of Yang et al. (2023),
        using NumExpr for optimization.
        This optimized version performs the entire calculation in a single evaluation for maximum speed.

        Parameters
        ---
        tt : np.ndarray
            The time normalized by the collapse time.
        rmax_CDM : np.ndarray or float
            The radius at which the maximum circular velocity of the CDM halo is obtained,
            evaluated at the same time as tt.

        Returns
        ---
        out : np.ndarray
            The differential of rmax.
        """
        out = ne.evaluate(
            "where(tt <= tt_th, (0.007623 - 1.44*tt + 1.0128*tt*tt - 0.55*tt*tt*tt) * rmax_CDM, 0)",
            local_dict={"tt": tt, "tt_th": self.tt_th, "rmax_CDM": rmax_CDM},
        )
        return out

    def get_Vmax0(self, Vmax, tt):
        """Returns the maximum circular velocity of the initial NFW profile, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        Vmax : float
            The maximum circular velocity.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        Vmax0 : float
            The value of Vmax of the initial NFW profile.
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
        """Returns the radius at which the maximum circular velocity is obtained, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        rmax : float
            The radius at which the maximum circular velocity is obtained.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        rmax0 : float
            The value of rmax of the initial NFW profile.
        """
        rmax0 = rmax / (1.0 + 0.007623 * tt - 0.7200 * tt**2 + 0.3376 * tt**3 - 0.1375 * tt**4)
        return rmax0

    def get_rhos(self, rhos0, tt):
        """Returns the density parameter of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        rhos0 : float
            The density parameter of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        rhos : float
            The density parameter of a SIDM halo.
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
        """Returns the scale radius of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        rs0 : float
            The scale radius of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        rs : float
            The scale radius of a SIDM halo.
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
        """Returns the core radius of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        rs0 : float
            The scale radius of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        rc : float
            The core radius of a SIDM halo.
        """
        rc = (
            2.555 * np.sqrt(tt) - 3.632 * tt + 2.131 * tt**2 - 1.415 * tt**3 + 0.4683 * tt**4
        ) * rs0
        return rc

    def master_function(self, Vmax_CDM, rmax_CDM, t, t_f):
        """Calculate the properties of a SIDM halo at a given time t according to the integral approach proposed by Yang et al. (2023) [arXiv:2305.16176].

        Parameters
        ---
        Vmax_CDM : float
            The maximum circular velocity of the CDM halo, as a function of time (axis 1).
        rmax_CDM : float
            The radius at which the maximum circular velocity is obtained, as a function of time (axis 1).
        t : float
            The time, running from the accretion time t_f to the time of observation.
        t_f : float
            The accretion (formation) time of the subhalo, i.e. the lower end of the integral in Eq. (3.3).

        Returns
        ---
        VmaxSIDM_z0 : float
            The maximum circular velocity of the SIDM halo at z=0.
        rmaxSIDM_z0 : float
            The radius at which the maximum circular velocity is obtained at z=0.
        rhosSIDM_z0 : float
            The density parameter of the SIDM halo at z=0.
        rsSIDM_z0 : float
            The scale radius of the SIDM halo at z=0.
        rcSIDM_z0 : float
            The core radius of the SIDM halo at z=0.
        """
        # NOTE: In the equation just below Eq. (3.3) of Yang et al. (2023), dVmax_{Model}/dtt and
        # drmax_{Model}/dtt are normalized by Vmax_{CDM}(t) and rmax_{CDM}(t) evaluated at the running
        # time t of the integral, not by their values at the accretion time t_f.
        t_c = self.t_collapse(self.sigma_eff_m(Vmax_CDM), rmax_CDM, Vmax_CDM)
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
        """Evaluate the retained SI tidal-loss amplitude as a function of virial host mass and
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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        log10a = (-0.0003 * np.log10(self.Mzvir(z) / self.Msun) + 0.02) * z + (
            0.011 * np.log10(self.Mzvir(z) / self.Msun) - 0.354
        )
        return 10.0**log10a

    def zetaMz(self, z):
        """Evaluate the fitted exponent of the tidal subhalo-to-host mass ratio.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
        """
        return (0.00012 * np.log10(self.Mzvir(z) / self.Msun) - 0.0033) * z + (
            -0.0011 * np.log10(self.Mzvir(z) / self.Msun) + 0.026
        )

    def tdynz(self, z):
        """Evaluate the virial dynamical time entering the fitted tidal loss law.

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
        SI retains its fitted CDM background/host/tidal relations and SIDM gravothermal
        prescriptions. Internal base units are Msun, Mpc and seconds, so internal speeds are
        Mpc/s. The public stage/catalogue interface explicitly converts velocities to km/s. The
        rounded SI gravitational constant and its critical-density normalization are retained.
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
