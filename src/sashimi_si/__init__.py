"""Standard SIDM API with paired weighted catalogs and ITAMAE execution.

The archived reference is maintained independently in sashimi-family.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy import integrate, special
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss
from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.execution import PopulationComponents
from itamae.halo import invert_nfw_mass_function
from itamae.provenance import build_calculation_metadata
from itamae.types import WeightedSubhaloCatalog
from itamae.units import NativeUnits
from ._physics import (
    SIHaloKernels,
    SIUnits,
    SITidalKernels,
    SIDM_cross_section,
    SIDM_parametric_model,
)
from ._itamae_components import (
    SIDMAccretionSlices,
    SIDMInitialStructure,
    SIDMProfileEvolution,
    SIDMSurvival,
    SIDMCatalogColumns,
)

CALCULATION_SPECIFICATION = "sashimi-si:paired-catalog:2026-09-10:v2"


class HaloModel(SIHaloKernels):
    r"""Planck2018 SI calibration with an explicit common background backend.

    Parameters
    ----------
    backend_config : BackendConfig, optional
        Explicit background/unit/array configuration; mutually exclusive with
        cosmology_backend. SI requires NativeUnits and its adopted cosmological
        parameters.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.

    Notes
    -----
    CDM-reference and SIDM views share accretion nodes and generation weights. Formation
    before accretion is a domain requirement; invalid histories are not integrated backward
    and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
    CDM tidal-truncation selection.
    """

    calculation_specification = CALCULATION_SPECIFICATION

    def __init__(self, *, backend_config=None, cosmology_backend=None):
        r"""Bind the adopted SI background to one explicit native-unit backend configuration.

        Parameters
        ----------
        backend_config : BackendConfig, optional
            Explicit background/unit/array configuration; mutually exclusive with
            cosmology_backend. SI requires NativeUnits and its adopted cosmological
            parameters.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.

        Notes
        -----
        The backend compatibility requirement Omega_m0=0.315, H0=67.4 km/s/Mpc is the adopted
        rounded Planck-2018 background. It does not imply that the earlier concentration or
        tidal fits were recalibrated for this background.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------

        .. [cite-planck] Planck Collaboration (2020), Planck 2018 results. VI. Cosmological parameters, Table 2.
            https://arxiv.org/abs/1807.06209
        """
        if backend_config is not None and cosmology_backend is not None:
            raise ValueError("Pass either backend_config or cosmology_backend, not both.")
        config = backend_config or BackendConfig(
            cosmology=cosmology_backend or NativeFlatLCDM(), units=NativeUnits()
        )
        if not isinstance(config.units, NativeUnits):
            raise ValueError("SASHIMI-SI uses canonical floating arrays and requires NativeUnits.")
        backend = config.cosmology
        if not np.isclose(float(np.asarray(backend.omega_m(0))), 0.315, rtol=0, atol=1e-12):
            raise ValueError("SASHIMI-SI calibration requires OmegaM=0.315.")
        if not np.isclose(float(np.asarray(backend.H(0))), 67.4, rtol=0, atol=1e-10):
            raise ValueError("SASHIMI-SI calibration requires h=0.674.")
        self.itamae_backend = config
        self.itamae_cosmology = backend
        SIHaloKernels.__init__(self)
        self._rho_crit_scale = self.rhocrit0 / (
            float(np.asarray(backend.rho_crit(0))) * self.Msun / self.Mpc**3
        )

    def Hubble(self, z: Any) -> np.ndarray:
        """Return the Hubble rate in the inverse-second unit.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            H(z) in inverse seconds, with the redshift shape.

        Notes
        -----
        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.
        """
        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the published SI rounded-G convention.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Critical density in Msun/Mpc^3 after the explicit rounded-G normalization scale.

        Notes
        -----
        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.
        """
        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        r"""Return the normalized linear growth factor.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Dimensionless D(z) returned by the selected background.

        Notes
        -----
        The factors 5/2, 4/7 and 1/70 belong to the Carroll--Press--Turner growth approximation.
        Redshift derivatives differentiate that approximation; they are not a separate growth
        calibration.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------

        .. [cite-growth] Carroll, Press & Turner (1992), The Cosmological Constant, Section 3.5.
            https://doi.org/10.1146/annurev.aa.30.090192.002435
        """
        return np.asarray(self.itamae_cosmology.growth_factor(z))


class TidalStrippingSolver(HaloModel, SITidalKernels):
    r"""SI tidal evolution retaining the selected immutable backend.

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
    backend_config : BackendConfig, optional
        Explicit background/unit/array configuration; mutually exclusive with
        cosmology_backend. SI requires NativeUnits and its adopted cosmological
        parameters.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.

    Notes
    -----
    CDM-reference and SIDM views share accretion nodes and generation weights. Formation
    before accretion is a domain requirement; invalid histories are not integrated backward
    and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
    CDM tidal-truncation selection.
    """

    def __init__(
        self,
        M0,
        z_min=0.0,
        z_max=7.0,
        n_z_interp=64,
        *,
        backend_config=None,
        cosmology_backend=None,
    ):
        r"""Prepare SI tidal integration while retaining the selected background configuration.

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
        backend_config : BackendConfig, optional
            Explicit background/unit/array configuration; mutually exclusive with
            cosmology_backend. SI requires NativeUnits and its adopted cosmological
            parameters.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.

        Notes
        -----
        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.
        """
        HaloModel.__init__(self, backend_config=backend_config, cosmology_backend=cosmology_backend)
        SITidalKernels.__init__(self, M0, z_min=z_min, z_max=z_max, n_z_interp=n_z_interp)


class SubhaloProperties(HaloModel, SIDM_parametric_model):
    r"""Calculate aligned CDM-reference/SIDM states with independent weights.

    Parameters
    ----------
    sigma0_m : float
        Low-velocity scattering cross section per particle mass, numerical value in cm^2/g
        at the public constructor boundary.
    w : float
        Scattering velocity scale, numerical value in km/s at the public constructor
        boundary.
    beta : float
        Dimensionless retained SIDM structural parameter stored by the population model.
    tt_th : float
        Upper dimensionless gravothermal time (t-t_f)/t_c at which the fitted SIDM evolution
        contribution is stopped.
    backend_config : BackendConfig, optional
        Explicit background/unit/array configuration; mutually exclusive with
        cosmology_backend. SI requires NativeUnits and its adopted cosmological
        parameters.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.

    Notes
    -----
    The defaults sigma0/m=147.1 cm**2/g and w=24.33 km/s are the fiducial Model I in Ando et
    al., Table 1, also used by Yang et al. The default stop at gravothermal time 1.1 follows
    Ando et al., Section 2.2; it bounds the adopted evolution prescription, not numerical
    error. The beta argument is retained state and is not used by the current evolution
    kernels.

    CDM-reference and SIDM views share accretion nodes and generation weights. Formation
    before accretion is a domain requirement; invalid histories are not integrated backward
    and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
    CDM tidal-truncation selection.

    References
    ----------

    .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
        https://arxiv.org/abs/2403.16633

    .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
        https://arxiv.org/abs/2305.16176
    """

    def __init__(
        self,
        sigma0_m=147.1,
        w=24.33,
        beta=4,
        tt_th=1.1,
        *,
        backend_config=None,
        cosmology_backend=None,
    ):
        r"""Bind the scattering law and gravothermal profile fit to a paired CDM/SIDM population.

        Parameters
        ----------
        sigma0_m : float
            Low-velocity scattering cross section per particle mass, numerical value in cm^2/g
            at the public constructor boundary.
        w : float
            Scattering velocity scale, numerical value in km/s at the public constructor
            boundary.
        beta : float
            Dimensionless retained SIDM structural parameter stored by the population model.
        tt_th : float
            Upper dimensionless gravothermal time (t-t_f)/t_c at which the fitted SIDM evolution
            contribution is stopped.
        backend_config : BackendConfig, optional
            Explicit background/unit/array configuration; mutually exclusive with
            cosmology_backend. SI requires NativeUnits and its adopted cosmological
            parameters.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.

        Notes
        -----
        The defaults sigma0/m=147.1 cm**2/g and w=24.33 km/s are the fiducial Model I in Ando et
        al., Table 1, also used by Yang et al. The default stop at gravothermal time 1.1 follows
        Ando et al., Section 2.2; it bounds the adopted evolution prescription, not numerical
        error. The beta argument is retained state and is not used by the current evolution
        kernels.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------

        .. [cite-sashimi-sidm] Ando et al. (2025), SASHIMI-SIDM: Semi-analytical subhalo modelling for self-interacting dark matter at sub-galactic scales, Sections 2.2-2.3, Table 1.
            https://arxiv.org/abs/2403.16633

        .. [cite-sidm] Yang et al. (2024; preprint 2023), A parametric model for self-interacting dark matter halos, PDF Eqs. (1.1)-(1.3), (2.2)-(2.4), (3.3).
            https://arxiv.org/abs/2305.16176
        """
        HaloModel.__init__(self, backend_config=backend_config, cosmology_backend=cosmology_backend)
        self.param_model = SIDM_parametric_model(sigma0_m, w, tt_th)
        self.sigma0_m = sigma0_m * self.cm**2 / self.gram
        self.w = w * self.km / self.s
        self.sigma_eff_m = self.param_model.sigma_eff_m
        self.beta = beta
        self.ct_func = invert_nfw_mass_function
        self.physical_parameters = {
            "sigma0_m_cm2_g": sigma0_m,
            "w_km_s": w,
            "beta": beta,
            "collapse_time_threshold": tt_th,
        }

    def Ffunc(self, dela, s1, s2):
        r"""Evaluate the Yang et al. EPS kernel before support normalization.

        Parameters
        ----------
        dela : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        array_like
            delta_a/[sqrt(2*pi)*(s2-s1)**(3/2)], on positive variance support.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------
        - Yang et al. (2011), https://arxiv.org/abs/1104.1757

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        return 1 / np.sqrt(2.0 * np.pi) * dela / (s2 - s1) ** 1.5

    def Gfunc(self, dela, s1, s2):
        r"""Evaluate the retained empirical multiplicative correction to the EPS kernel.

        Parameters
        ----------
        dela : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        array_like
            0.57*(sqrt(s2/s1))**0.38*(delta_a/sqrt(s1))**(-0.01), dimensionless.

        Notes
        -----
        The empirical correction G0=0.57, gamma1=0.38, gamma2=-0.01 is from Parkinson et al.,
        Section 3.1, and is adopted in Yang et al., Eq. (13).

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------
        - Yang et al. (2011), https://arxiv.org/abs/1104.1757

        .. [cite-parkinson] Parkinson, Cole & Helly (2008), Generating Dark Matter Halo Merger Trees, Section 3.1.
            https://arxiv.org/abs/0708.1382

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        G0 = 0.57
        gamma1 = 0.38
        gamma2 = -0.01
        sig1 = np.sqrt(s1)
        sig2 = np.sqrt(s2)
        return G0 * pow(sig2 / sig1, gamma1) * pow(dela / sig1, gamma2)

    def Ffunc_Yang(self, delc1, delc2, s1, s2):
        r"""Evaluate the conditional EPS kernel with its collapse-barrier difference.

        Parameters
        ----------
        delc1 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        delc2 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        array_like
            Gaussian first-crossing density in variance, before support normalization.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------
        - Yang et al. (2011), https://arxiv.org/abs/1104.1757

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        return (
            1.0
            / np.sqrt(2.0 * np.pi)
            * (delc2 - delc1)
            / (s2 - s1) ** 1.5
            * np.exp(-((delc2 - delc1) ** 2) / (2.0 * (s2 - s1)))
        )

    def _normalized_yang_kernel(self, delc1, delc2, s1, s2, smin):
        r"""Yang Eq. (14) divided by its mass-support integral, including gap=0.

        For x=delta/sqrt(2*(smin-s1)), erf(x)/x tends to 2/sqrt(pi).
        The small-x series evaluates that ratio before the vanishing factors
        are divided. This is the same normalized kernel, with no weight cut.

        Parameters
        ----------
        delc1 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        delc2 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        smin : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        numpy.ndarray
            Conditional variance kernel divided by its finite-support integral, including the
            zero-barrier-gap limit.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        d1, d2, a, b, minimum = np.broadcast_arrays(delc1, delc2, s1, s2, smin)
        gap, ds, dsmin = d2 - d1, b - a, minimum - a
        if np.any(gap < 0) or np.any(ds <= 0) or np.any(dsmin <= 0):
            raise ValueError(
                "Normalized Yang kernel requires nonnegative barrier gap and positive variance gaps."
            )
        x2 = gap**2 / (2.0 * dsmin)
        small = x2 < 1e-8
        result = np.empty_like(x2, dtype=float)
        x = x2[small]
        integral = 1.0 - x / 3.0 + x**2 / 10.0 - x**3 / 42.0 + x**4 / 216.0
        result[small] = (
            np.sqrt(dsmin[small])
            / (2.0 * ds[small] ** 1.5)
            * np.exp(-(gap[small] ** 2) / (2.0 * ds[small]))
            / integral
        )
        regular = ~small
        norm = special.gamma(0.5) * special.gammainc(0.5, x2[regular]) / np.sqrt(np.pi)
        result[regular] = self.Ffunc_Yang(d1[regular], d2[regular], a[regular], b[regular]) / norm
        return result

    def Na_calc(self, ma, zacc, Mhost, z0=0.0, N_herm=200, Nrand=1000, Na_model=3):
        r"""Returns Na, Eq. (3) of Yang et al. (2011)

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        zacc : numpy.ndarray
            One-dimensional dimensionless accretion-redshift grid, paired with the population
            grid first axis.
        Mhost : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        z0 : float
            Target or reference redshift specified by the routine; the tidal ODE evolves from za
            down to this epoch.
        N_herm : int
            Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
            concentration in catalogue stages or host history in Na_calc.
        Nrand : int
            Number of points in the deterministic auxiliary redshift inversion grid; despite its
            name, this is not a Monte Carlo sample size.
        Na_model : {1, 2, 3}
            Yang EPS accretion/normalization prescription. This is a physical model selection,
            not a numerical tolerance.

        Returns
        -------
        numpy.ndarray
            Expected accretion rate per d ln(ma) d ln(1+zacc), after host-scatter averaging;
            first axis is redshift.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------
        - Yang et al. (2011), https://arxiv.org/abs/1104.1757

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        zacc_2d = zacc.reshape(-1, 1)
        M200_0 = self.Mzzi(Mhost, zacc_2d, z0)
        logM200_0 = np.log10(M200_0)

        xxi, wwi = hermgauss(N_herm)
        xxi = xxi.reshape(-1, 1, 1)
        wwi = wwi.reshape(-1, 1, 1)
        # Eq. (21) in Yang et al. (2011)
        sigmalogM200 = 0.12 - 0.15 * np.log10(M200_0 / Mhost)
        logM200 = np.sqrt(2.0) * sigmalogM200 * xxi + logM200_0
        M200 = 10.0**logM200

        mmax = np.minimum(M200, Mhost / 2.0)
        Mmax = np.minimum(M200_0 + mmax, Mhost)

        if Na_model == 3:
            zlist = zacc_2d * np.linspace(1, 0, Nrand)
            iMmax = np.argmin(np.abs(self.Mzzi(Mhost, zlist, z0) - Mmax), axis=-1)
            z_Max = zlist[np.arange(len(zlist)), iMmax]
            z_Max_3d = z_Max.reshape(N_herm, len(zlist), 1)
            delcM = self.deltac_func(z_Max_3d)
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(Mmax)
            sa = self.s_func(ma)
            # those reside in the exponential part of Eq. (14)
            d1, d2, s1, s2, smin, allowed = np.broadcast_arrays(
                delcM, delca, sM, sa, self.s_func(mmax), mmax > ma
            )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = self._normalized_yang_kernel(
                d1[allowed], d2[allowed], s1[allowed], s2[allowed], smin[allowed]
            )
        elif Na_model == 1:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            d_norm, s_norm, x_norm = np.broadcast_arrays(delca, sM, xmin)
            if np.any(x_norm < 0):
                raise ValueError("EPS normalization requires a nonnegative variance support gap.")
            positive = x_norm > 0
            normB = np.full(x_norm.shape, np.inf)
            d_pos, s_pos, x_pos = d_norm[positive], s_norm[positive], x_norm[positive]
            normB[positive] = (
                1.0
                / np.sqrt(2 * np.pi)
                * d_pos
                * 2.0
                / x_pos**0.5
                * special.hyp2f1(0.5, 0.0, 1.5, -s_pos / x_pos)
            )
            d, s1, s2, norm, allowed = np.broadcast_arrays(delca, sM, sa, normB, mmax > ma)
            if np.any((s2 - s1)[allowed] <= 0):
                raise ValueError(
                    "EPS accretion requires positive variance gaps in its active mass domain."
                )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = self.Ffunc(d[allowed], s1[allowed], s2[allowed]) / norm[allowed]
        elif Na_model == 2:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            d_norm, s_norm, x_norm = np.broadcast_arrays(delca, sM, xmin)
            if np.any(x_norm < 0):
                raise ValueError("EPS normalization requires a nonnegative variance support gap.")
            positive = x_norm > 0
            normB = np.full(x_norm.shape, np.inf)
            d_pos, s_pos, x_pos = d_norm[positive], s_norm[positive], x_norm[positive]
            normB[positive] = (
                1.0
                / np.sqrt(2.0 * np.pi)
                * d_pos
                * 0.57
                * (d_pos / np.sqrt(s_pos)) ** -0.01
                * (2.0 / (1.0 - 0.38))
                * s_pos ** (-0.38 / 2.0)
                * x_pos ** (0.5 * (0.38 - 1.0))
                * special.hyp2f1(0.5 * (1 - 0.38), -0.38 / 2.0, 0.5 * (3.0 - 0.38), -s_pos / x_pos)
            )
            d, s1, s2, norm, allowed = np.broadcast_arrays(delca, sM, sa, normB, mmax > ma)
            if np.any((s2 - s1)[allowed] <= 0):
                raise ValueError(
                    "EPS accretion requires positive variance gaps in its active mass domain."
                )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = (
                self.Ffunc(d[allowed], s1[allowed], s2[allowed])
                * self.Gfunc(d[allowed], s1[allowed], s2[allowed])
                / norm[allowed]
            )
        else:
            raise ValueError("Na_model must be 1, 2, or 3.")
        if not np.all(np.isfinite(Phi)):
            raise ValueError("Non-finite EPS accretion kernel inside its active mass domain.")
        # calculate Na
        if N_herm == 1:
            F2t = Phi
            F2 = F2t[0]
        else:
            F2 = np.sum(Phi * wwi / np.sqrt(np.pi), axis=0)
        Na = F2 * self.dsdm(ma, 0.0) * self.dMdz(Mhost, zacc_2d, z0) * (1.0 + zacc_2d)
        return Na

    def subhalo_catalogs_calc(
        self,
        M0,
        redshift=0.0,
        dz=0.01,
        zmax=5.0,
        N_ma=500,
        sigmalogc=0.128,
        N_herm=20,
        logmamin=6,
        logmamax=None,
        N_hermNa=200,
        Na_model=3,
        ct_th=0.0,
        M0_at_redshift=False,
        method="picard",
        **kwargs: Any,
    ):
        r"""Run SIDM physical stages through the shared population executor.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        redshift : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        dz : float
            Positive nominal spacing of the accretion-redshift quadrature grid.
        zmax : float
            Largest requested accretion redshift, above the target epoch; the actual grid
            convention belongs to this variant.
        N_ma : int
            Number of logarithmically spaced accretion-mass nodes.
        sigmalogc : float
            Nonnegative standard deviation of log10 concentration, in dex.
        N_herm : int
            Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
            concentration in catalogue stages or host history in Na_calc.
        logmamin : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        logmamax : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        N_hermNa : int
            Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
        Na_model : {1, 2, 3}
            Yang EPS accretion/normalization prescription. This is a physical model selection,
            not a numerical tolerance.
        ct_th : float
            Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
            is applied once through its mask.
        M0_at_redshift : bool
            Interpret the input host mass at the target redshift and invert the prescribed
            history to obtain its z=0 reference value.
        method : str
            picard (default), dop853, odeint, or an explicitly selected retained perturbative
            approximation.
        **kwargs : dict
            Additional keyword options forwarded to the called routine; reserved scientific
            controls remain owned by that routine.

        Returns
        -------
        dict[str, WeightedSubhaloCatalog]
            cdm_reference and sidm catalogues sharing aligned physical columns and independent
            generation factors, with distinct survival weights.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        Masses are Msun, radii Mpc, densities Msun/Mpc^3 and named velocities km/s.
        Formation-invalid SIDM entries remain explicitly zero and have zero survival weight. The
        original arange(target+dz,zmax+dz,dz) grid convention is retained; non-aligned zmax may
        be exceeded by less than one nominal step.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004
        """
        if method not in {
            "picard",
            "dop853",
            "odeint",
            "pert0",
            "pert1",
            "pert2",
            "pert2_shanks",
            "pert3",
        }:
            raise ValueError(f"Invalid stripping method: {method}")
        self.calculation_parameters = {
            "M0": M0,
            "redshift": redshift,
            "dz": dz,
            "zmax": zmax,
            "N_ma": N_ma,
            "sigmalogc": sigmalogc,
            "N_herm": N_herm,
            "logmamin": logmamin,
            "logmamax": logmamax,
            "N_hermNa": N_hermNa,
            "Na_model": Na_model,
            "ct_th": ct_th,
            "M0_at_redshift": M0_at_redshift,
            "method": method,
            "solver_options": kwargs,
        }
        if M0_at_redshift:
            Mz = M0
            M0_list = np.logspace(0.0, 3.0, 1000) * Mz
            fint = interp1d(self.Mzi(M0_list, redshift), M0_list)
            M0 = fint(Mz)
        self.M0 = M0
        self.redshift = redshift

        _zmax = np.linspace(redshift, 10.0, 1001)
        z_dummy = np.linspace(redshift, _zmax, 1000)
        t_L = integrate.simpson(
            1.0 / (self.Hubble(z_dummy) * (1.0 + z_dummy)),
            x=z_dummy,
            axis=0,
        )
        self.lookback_time = interp1d(_zmax, t_L)
        z_dummy = np.linspace(redshift, 1000.0, 10000)
        self.t_U = integrate.simpson(
            1.0 / (self.Hubble(z_dummy) * (1.0 + z_dummy)),
            x=z_dummy,
        )

        zdist = np.arange(redshift + dz, zmax + dz, dz)
        if logmamax is None:
            logmamax = np.log10(0.1 * M0 / self.Msun)
        ma200_z0 = np.logspace(logmamin, logmamax, N_ma) * self.Msun
        ma_z0 = self.Mvir_from_M200_fit(ma200_z0, redshift)

        ma200_0 = np.empty_like(ma200_z0)
        ma_0 = np.empty_like(ma_z0)
        if redshift == 0.0:
            ma200_0 = ma200_z0
            ma_0 = ma_z0
        else:
            ma200_0_list = np.logspace(0.0, 2.0, 200) * ma200_z0.reshape(-1, 1)
            ma_0_list = np.logspace(0.0, 2.0, 200) * ma_z0.reshape(-1, 1)
            for index in np.arange(len(ma200_z0)):
                fint_ma200 = interp1d(self.Mzi(ma200_0_list[index], redshift), ma200_0_list[index])
                fint_ma = interp1d(self.Mzi(ma_0_list[index], redshift), ma_0_list[index])
                ma200_0[index] = fint_ma200(ma200_z0[index])
                ma_0[index] = fint_ma(ma_z0[index])

        z_f = (
            -0.0064 * (np.log10(ma_0 / self.Msun)) ** 2
            + 0.0237 * np.log10(ma_0 / self.Msun)
            + 1.8837
        )
        t_f = self.t_U - self.lookback_time(z_f)

        ma200_matrix = self.Mzi(ma200_0, zdist[:, np.newaxis])
        ma_matrix = self.Mvir_from_M200_fit(ma200_matrix, zdist[:, np.newaxis])
        accretion = z_f > zdist[:, np.newaxis]
        active = accretion.any(axis=1)
        zdist_accreted = zdist[active]
        ma200_matrix_accreted = ma200_matrix[active]
        ma_matrix_accreted = ma_matrix[active]

        solver = TidalStrippingSolver(
            backend_config=self.itamae_backend,
            M0=M0,
            z_min=redshift,
            z_max=zmax,
            n_z_interp=64,
        )

        Na = self.Na_calc(
            ma_matrix,
            zdist,
            M0,
            z0=0.0,
            N_herm=N_hermNa,
            Nrand=1000,
            Na_model=Na_model,
        )
        Na_total = integrate.simpson(
            integrate.simpson(Na, x=np.log(ma_matrix)),
            x=np.log(1.0 + zdist),
        )
        weight_base = Na / (1.0 + zdist.reshape(-1, 1))
        weight_base = weight_base / np.sum(weight_base) * Na_total
        weight_base_active = weight_base[active]

        slice_builder = SIDMAccretionSlices(
            model=self,
            zdist_accreted=zdist_accreted,
            ma_matrix_accreted=ma_matrix_accreted,
            z_f=z_f,
            ma200_0=ma200_0,
            sigmalogc=sigmalogc,
            N_herm=N_herm,
            ma200_matrix_accreted=ma200_matrix_accreted,
            weight_base_active=weight_base_active,
            t_f=t_f,
        )
        slices = [slice_builder.build(index) for index in range(len(zdist_accreted))]
        execution = PopulationComponents(
            initializer=SIDMInitialStructure(model=self),
            evolver=SIDMProfileEvolution(
                model=self,
                solver=solver,
                redshift=redshift,
                method=method,
                kwargs=kwargs,
                N_herm=N_herm,
                ma200_z0=ma200_z0,
            ),
            survival=SIDMSurvival(ct_threshold=ct_th),
            columns=SIDMCatalogColumns(),
        ).execute(
            [batch for batch, _ in slices],
            contexts=[context for _, context in slices],
        )
        columns = {
            name.replace("rmax_", "r_max_"): np.asarray(value)
            for name, value in execution.columns.items()
        }
        return {
            state: WeightedSubhaloCatalog(
                columns=columns,
                weights={
                    **dict(execution.weight_factors),
                    "weight_survival": execution.survival[state].astype(float),
                },
                metadata=build_calculation_metadata(
                    variant="sashimi-si",
                    distribution_name="sashimi-si",
                    module_file=__file__,
                    model_identifier=f"sashimi-si:{state}:v2",
                    backend_identifier=self.itamae_backend.identifier,
                    source_identifier="sashimi-si:upstream-physics:e17d3664dac677b604fd4ff02fb2af105a6937fa",
                    calculation_specification=CALCULATION_SPECIFICATION,
                    variance_identifier="sashimi-si:analytic-cdm-fit:v1",
                    power_identifier="sashimi-si:cdm-linear-power:v1",
                    solver_identifier=f"sashimi-si:gravothermal-tidal-stripping:{method}:itamae:v3",
                    extra={
                        "state": state,
                        "default_stripping_method": "picard",
                        "picard_diagnostics": solver._picard_backend.diagnostics()
                        if method == "picard" and hasattr(solver, "_picard_backend")
                        else None,
                        "physical_parameters": self.physical_parameters,
                        "calculation_parameters": self.calculation_parameters,
                        "cosmology_parameters": {
                            "omega_m0": self.OmegaM,
                            "h": self.h,
                            "omega_lambda0": self.OmegaL,
                        },
                        "density_convention": "SI calibrated rounded G",
                        "effective_cross_section_evaluation": "positive-laguerre64-20to703:polynomial6-above703:v1",
                        "weight_factorization": "generation-stage",
                        "validity_policy": "formed-before-accretion; shared-CDM-truncation; SIDM-profile",
                        "uncomputed_sidm_value": "zero where valid_accretion is false",
                        "stage_unit_contract": "Msun-Mpc-km/s:v1",
                        "column_units": {
                            "mass": "Msun",
                            "length": "Mpc",
                            "velocity": "km / s",
                            "density": "Msun / Mpc3",
                        },
                    },
                ),
            )
            for state in ("cdm_reference", "sidm")
        }

    def subhalo_catalog_calc(self, *args: Any, state: str = "sidm", **kwargs: Any):
        r"""Calculate one named state view as a weighted ITAMAE catalog.

        Parameters
        ----------
        state : {"sidm", "cdm_reference", "cdm"}
            Select the SIDM view or the paired CDM reference; cdm is an alias for cdm_reference.
        *args : tuple
            Positional population inputs accepted by subhalo_catalogs_calc, beginning with host
            M0.
        **kwargs : dict
            Named population and numerical settings accepted by subhalo_catalogs_calc.

        Returns
        -------
        WeightedSubhaloCatalog
            Requested named survival view of the paired calculation.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004
        """
        if state == "cdm":
            state = "cdm_reference"
        if state not in {"cdm_reference", "sidm"}:
            raise ValueError("state must be 'cdm_reference' (or alias 'cdm') or 'sidm'.")
        catalogs = self.subhalo_catalogs_calc(*args, **kwargs)
        return catalogs[state]

    def subhalo_properties_calc(self, *args, return_weight_factors=False, **kwargs):
        r"""Format the paired catalogs as the historical 27-field tuple.

        Velocities in this format are Mpc/s; named catalogs use km/s.

        Parameters
        ----------
        return_weight_factors : bool
            Also return independent generation/survival factors alongside the historical
            paired-state tuple.
        *args : tuple
            Positional population inputs forwarded to subhalo_catalogs_calc.
        **kwargs : dict
            Population and numerical settings forwarded to subhalo_catalogs_calc.

        Returns
        -------
        tuple
            Historical 27-field paired-state result; if return_weight_factors is true, return
            (result, named factor dictionaries). Velocities are converted back to Mpc/s in this
            format.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        CDM-reference and SIDM views share accretion nodes and generation weights. Formation
        before accretion is a domain requirement; invalid histories are not integrated backward
        and carry zero survival weight. SIDM profile validity adds a separate mask to the shared
        CDM tidal-truncation selection.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004
        """
        catalogs = self.subhalo_catalogs_calc(*args, **kwargs)
        cdm, sidm = catalogs["cdm_reference"], catalogs["sidm"]
        names = (
            "m200_acc",
            "z_acc",
            "r_s_cdm_acc",
            "rho_s_cdm_acc",
            "r_max_cdm_acc",
            "v_max_cdm_acc",
            "r_s_sidm_acc",
            "rho_s_sidm_acc",
            "r_c_sidm_acc",
            "r_max_sidm_acc",
            "v_max_sidm_acc",
            "m_bound",
            "r_s_cdm",
            "rho_s_cdm",
            "r_max_cdm",
            "v_max_cdm",
            "r_s_sidm",
            "rho_s_sidm",
            "r_c_sidm",
            "r_max_sidm",
            "v_max_sidm",
            "c_t_cdm",
            "collapse_time_ratio",
        )
        values = tuple(
            cdm.columns[name] * (self.km / self.s)
            if name.startswith("v_max_")
            else cdm.columns[name]
            for name in names
        )
        result = (
            *values,
            cdm.weight_final,
            sidm.weight_final,
            cdm.columns["survive_cdm"],
            cdm.columns["survive_sidm"],
        )
        if return_weight_factors:
            return result, {state: dict(catalog.weights) for state, catalog in catalogs.items()}
        return result


units_and_constants = SIUnits
cosmology = HaloModel
halo_model = HaloModel
subhalo_properties = SubhaloProperties
create_itamae_model = SubhaloProperties
__all__ = [
    "units_and_constants",
    "cosmology",
    "HaloModel",
    "SubhaloProperties",
    "TidalStrippingSolver",
    "SIDM_cross_section",
    "SIDM_parametric_model",
    "halo_model",
    "subhalo_properties",
    "create_itamae_model",
    "CALCULATION_SPECIFICATION",
]
