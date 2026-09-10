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
from sashimi_si_physics import (
    SIHaloKernels,
    SIUnits,
    SITidalKernels,
    SIDM_cross_section,
    SIDM_parametric_model,
)
from sashimi_si_itamae_components import (
    SIDMAccretionSlices,
    SIDMInitialStructure,
    SIDMProfileEvolution,
    SIDMSurvival,
    SIDMCatalogColumns,
)

CALCULATION_SPECIFICATION = "sashimi-si:paired-catalog:2026-09-10:v1"


class HaloModel(SIHaloKernels):
    """Planck2018 SI calibration with an explicit common background backend."""

    calculation_specification = CALCULATION_SPECIFICATION

    def __init__(self, *, backend_config=None, cosmology_backend=None):
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
        """Return the Hubble rate in the inverse-second unit."""
        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the published SI rounded-G convention."""
        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        """Return the normalized linear growth factor."""
        return np.asarray(self.itamae_cosmology.growth_factor(z))


class TidalStrippingSolver(HaloModel, SITidalKernels):
    """SI tidal evolution retaining the selected immutable backend."""

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
        HaloModel.__init__(self, backend_config=backend_config, cosmology_backend=cosmology_backend)
        SITidalKernels.__init__(self, M0, z_min=z_min, z_max=z_max, n_z_interp=n_z_interp)


class SubhaloProperties(HaloModel, SIDM_parametric_model):
    """Calculate aligned CDM-reference/SIDM states with independent weights."""

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
        """Returns Eq. (12) of Yang et al. (2011)

        Refereces
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        return 1 / np.sqrt(2.0 * np.pi) * dela / (s2 - s1) ** 1.5

    def Gfunc(self, dela, s1, s2):
        """Returns the G function used in Eq. (13) of Yang et al. (2011)

        References
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        G0 = 0.57
        gamma1 = 0.38
        gamma2 = -0.01
        sig1 = np.sqrt(s1)
        sig2 = np.sqrt(s2)
        return G0 * pow(sig2 / sig1, gamma1) * pow(dela / sig1, gamma2)

    def Ffunc_Yang(self, delc1, delc2, s1, s2):
        """Returns Eq. (14) of Yang et al. (2011)

        References
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        return (
            1.0
            / np.sqrt(2.0 * np.pi)
            * (delc2 - delc1)
            / (s2 - s1) ** 1.5
            * np.exp(-((delc2 - delc1) ** 2) / (2.0 * (s2 - s1)))
        )

    def Na_calc(self, ma, zacc, Mhost, z0=0.0, N_herm=200, Nrand=1000, Na_model=3):
        """Returns Na, Eq. (3) of Yang et al. (2011)

        Parameters
        ---
        ma : float
            The mass of the subhalo.
        zacc : float
            The accretion redshift of the subhalo.
        Mhost : float
            The mass of the host halo.
        z0 : float
            The redshift.
        N_herm : int
            The number of Hermite-Gauss quadrature points.
        Nrand : int
            The number of random numbers.
        Na_model : int
            The model to calculate Na. Default is 3.

        Returns
        ---
        Na : float
            The value of Na.

        References
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
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
            xmax = (delca - delcM) ** 2 / (2.0 * (self.s_func(mmax) - sM))
            normB = special.gamma(0.5) * special.gammainc(0.5, xmax) / np.sqrt(np.pi)
            # those reside in the exponential part of Eq. (14)
            d1, d2, s1, s2, norm, allowed = np.broadcast_arrays(
                delcM, delca, sM, sa, normB, mmax > ma
            )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = (
                self.Ffunc_Yang(d1[allowed], d2[allowed], s1[allowed], s2[allowed]) / norm[allowed]
            )
        elif Na_model == 1:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            normB = (
                1.0
                / np.sqrt(2 * np.pi)
                * delca
                * 2.0
                / xmin**0.5
                * special.hyp2f1(0.5, 0.0, 1.5, -sM / xmin)
            )
            d, s1, s2, norm, allowed = np.broadcast_arrays(delca, sM, sa, normB, mmax > ma)
            Phi = np.zeros(s2.shape)
            Phi[allowed] = self.Ffunc(d[allowed], s1[allowed], s2[allowed]) / norm[allowed]
        elif Na_model == 2:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            normB = (
                1.0
                / np.sqrt(2.0 * np.pi)
                * delca
                * 0.57
                * (delca / np.sqrt(sM)) ** -0.01
                * (2.0 / (1.0 - 0.38))
                * sM ** (-0.38 / 2.0)
                * xmin ** (0.5 * (0.38 - 1.0))
                * special.hyp2f1(0.5 * (1 - 0.38), -0.38 / 2.0, 0.5 * (3.0 - 0.38), -sM / xmin)
            )
            d, s1, s2, norm, allowed = np.broadcast_arrays(delca, sM, sa, normB, mmax > ma)
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
        method="pert2_shanks",
        **kwargs: Any,
    ):
        """Run SIDM physical stages through the shared population executor."""
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
        for name in tuple(columns):
            if name.startswith("v_max_"):
                columns[name] = columns[name] / (self.km / self.s)
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
                    solver_identifier=f"sashimi-si:gravothermal-tidal-stripping:{method}:v2",
                    extra={
                        "state": state,
                        "physical_parameters": self.physical_parameters,
                        "calculation_parameters": self.calculation_parameters,
                        "cosmology_parameters": {
                            "omega_m0": self.OmegaM,
                            "h": self.h,
                            "omega_lambda0": self.OmegaL,
                        },
                        "density_convention": "SI calibrated rounded G",
                        "weight_factorization": "generation-stage",
                        "validity_policy": "formed-before-accretion; shared-CDM-truncation; SIDM-profile",
                        "uncomputed_sidm_value": "zero where valid_accretion is false",
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
        """Calculate one named state view as a weighted ITAMAE catalog."""
        if state == "cdm":
            state = "cdm_reference"
        if state not in {"cdm_reference", "sidm"}:
            raise ValueError("state must be 'cdm_reference' (or alias 'cdm') or 'sidm'.")
        catalogs = self.subhalo_catalogs_calc(*args, **kwargs)
        return catalogs[state]

    def subhalo_properties_calc(self, *args, return_weight_factors=False, **kwargs):
        """Format the paired catalogs as the historical 27-field tuple.

        Velocities in this format are Mpc/s; named catalogs use km/s.
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
