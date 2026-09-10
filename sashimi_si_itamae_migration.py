"""Incremental ITAMAE migration layer for SASHIMI-SI.

Shared LCDM background calculations, Shanks acceleration, NFW mass inversion,
and structured weighted-catalog output are supplied by ITAMAE. SIDM cross
sections, gravothermal evolution, calibrated profile maps, formation-time
choices, and disruption prescriptions remain in ``sashimi_si.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.evolution import shanks_transform
from itamae.execution import PopulationComponents
from itamae.halo import invert_nfw_mass_function
from itamae.provenance import build_migration_metadata
from itamae.types import WeightedSubhaloCatalog
from itamae.units import NativeUnits
from sashimi_si import TidalStrippingSolver, halo_model, subhalo_properties

from sashimi_si_itamae_components import (
    SIDMAccretionSlices,
    SIDMInitialStructure,
    SIDMProfileEvolution,
    SIDMSurvival,
    SIDMCatalogColumns,
)

_Base = TypeVar("_Base", bound=type)


class ItamaeMigrationMixin:
    """Supply common ITAMAE mechanisms to legacy SASHIMI-SI classes.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the legacy class.
    backend_config : itamae.backends.BackendConfig, optional
        Immutable ITAMAE backend selection. The native NumPy, unit, and
        flat-LCDM backends are used by default.
    physics_mode : {"consistent", "legacy"}, optional
        Compatibility label shared by SASHIMI migration façades. SASHIMI-SI
        has no known mode-dependent physical correction, so both values select
        the same canonical SIDM equations and defaults.
    cosmology_backend : object, optional
        Deprecated convenience route for selecting only an ITAMAE-compatible
        cosmology backend. It cannot be combined with ``backend_config``.
    **kwargs
        Keyword arguments forwarded to the legacy class.

    Notes
    -----
    A provisional backend is attached before ``super().__init__`` because the
    tidal solver constructs interpolation tables during initialization and calls
    overridable cosmology methods. The exact legacy density normalization is
    finalized after the base class has initialized its unit constants.
    """

    def __init__(
        self,
        *args: Any,
        backend_config: BackendConfig | None = None,
        cosmology_backend: Any | None = None,
        physics_mode: str = "consistent",
        **kwargs: Any,
    ) -> None:
        if physics_mode not in {"consistent", "legacy"}:
            raise ValueError("physics_mode must be 'consistent' or 'legacy'.")
        if backend_config is not None and cosmology_backend is not None:
            raise ValueError("Pass either backend_config or cosmology_backend, not both.")
        using_default_backend = backend_config is None and cosmology_backend is None
        if backend_config is None:
            backend = cosmology_backend or NativeFlatLCDM()
            backend_config = BackendConfig(cosmology=backend, units=NativeUnits())
        else:
            backend = backend_config.cosmology
        if not isinstance(backend_config.units, NativeUnits):
            raise ValueError(
                "The SASHIMI-SI migration currently emits canonical floating "
                "catalog arrays and therefore requires itamae.units.NativeUnits."
            )
        omega_m0 = float(np.asarray(backend.omega_m(0.0)))
        h = float(np.asarray(backend.H(0.0))) / 100.0
        if not np.isclose(omega_m0, 0.315, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "The result-preserving SASHIMI-SI migration requires "
                f"OmegaM=0.315; received {omega_m0}."
            )
        if not np.isclose(h, 0.674, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"The result-preserving SASHIMI-SI migration requires h=0.674; received {h}."
            )
        self.itamae_cosmology = backend
        self.itamae_backend = backend_config
        self.physics_mode = physics_mode
        self._rho_crit_scale = 1.0
        super().__init__(*args, **kwargs)

        if using_default_backend:
            backend = NativeFlatLCDM(omega_m0=self.OmegaM, h=self.h)
            self.itamae_cosmology = backend
            self.itamae_backend = BackendConfig(
                cosmology=backend,
                units=backend_config.units,
                array=backend_config.array,
            )
        if hasattr(backend, "omega_m0"):
            self.OmegaM = float(backend.omega_m0)
            self.OmegaL = 1.0 - self.OmegaM
        if hasattr(backend, "h"):
            self.h = float(backend.h)

        self.H0 = float(np.asarray(backend.H(0.0))) * self.km / self.s / self.Mpc
        backend_rho0 = float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        legacy_rho0 = 3.0 * self.H0**2 / (8.0 * np.pi * self.G)
        self._rho_crit_scale = legacy_rho0 / backend_rho0
        self.rhocrit0 = legacy_rho0
        if isinstance(self, TidalStrippingSolver):
            self.reset_interpolation(
                z_max=self.z_max,
                z_min=self.z_min,
                n_z=self.n_z_interp,
            )
        if isinstance(self, subhalo_properties):
            self.tidal_solver_factory = ItamaeTidalStrippingSolver
            self.ct_func = invert_nfw_mass_function

    def Hubble(self, z: Any) -> np.ndarray:
        """Return the Hubble rate in the legacy inverse-second unit."""
        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the legacy SASHIMI-SI convention."""
        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        """Return the normalized linear growth factor."""
        return np.asarray(self.itamae_cosmology.growth_factor(z))

    def subhalo_mass_stripped_pert2_shanks(self, ma, za, z):
        """Evaluate second-order stripping with ITAMAE Shanks acceleration.

        The SASHIMI-specific rule that disables acceleration for corrections
        smaller than two percent is preserved.
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

    def _calculate_population(
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

        solver = self.tidal_solver_factory(
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
        columns = execution.columns
        result = (
            columns["m200_acc"],
            columns["z_acc"],
            columns["r_s_cdm_acc"],
            columns["rho_s_cdm_acc"],
            columns["rmax_cdm_acc"],
            columns["v_max_cdm_acc"],
            columns["r_s_sidm_acc"],
            columns["rho_s_sidm_acc"],
            columns["r_c_sidm_acc"],
            columns["rmax_sidm_acc"],
            columns["v_max_sidm_acc"],
            columns["m_bound"],
            columns["r_s_cdm"],
            columns["rho_s_cdm"],
            columns["rmax_cdm"],
            columns["v_max_cdm"],
            columns["r_s_sidm"],
            columns["rho_s_sidm"],
            columns["r_c_sidm"],
            columns["rmax_sidm"],
            columns["v_max_sidm"],
            columns["c_t_cdm"],
            columns["collapse_time_ratio"],
            execution.weight_factors["weight_base"]
            * execution.weight_factors["weight_concentration"]
            * execution.survival["cdm_reference"].astype(float),
            execution.weight_factors["weight_base"]
            * execution.weight_factors["weight_concentration"]
            * execution.survival["sidm"].astype(float),
            columns["survive_cdm"],
            columns["survive_sidm"],
        )
        factors = {
            "cdm_reference": {
                **dict(execution.weight_factors),
                "weight_survival": execution.survival["cdm_reference"].astype(float),
            },
            "sidm": {
                **dict(execution.weight_factors),
                "weight_survival": execution.survival["sidm"].astype(float),
            },
        }
        return result, factors, columns

    def subhalo_properties_calc(self, *args, return_weight_factors=False, **kwargs):
        """Return the historical tuple format of the generated population."""
        result, factors, _ = self._calculate_population(*args, **kwargs)
        return (result, factors) if return_weight_factors else result

    def catalogs_from_legacy(
        self,
        result,
        *,
        weight_factors: Mapping[str, Mapping[str, Any]],
        validity: Mapping[str, Any] | None = None,
    ) -> Mapping[str, WeightedSubhaloCatalog]:
        """Convert legacy arrays and generation-stage factors into catalogs.

        Parameters
        ----------
        result : tuple
            Exact tuple returned by ``sashimi_si.subhalo_properties_calc``.
        weight_factors : mapping
            Independent base, concentration, and survival factors captured
            before multiplication in the legacy calculation.

        Returns
        -------
        mapping
            Mapping with ``"cdm_reference"`` and ``"sidm"`` weighted catalogs.

        Notes
        -----
        The legacy tuple retains velocities in Mpc/s. Catalog velocities are
        converted to ITAMAE's canonical km/s unit.
        """

        if len(result) != 27:
            raise ValueError(f"Expected 27 legacy outputs, received {len(result)}.")

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
        velocity_names = {
            "v_max_cdm_acc",
            "v_max_sidm_acc",
            "v_max_cdm",
            "v_max_sidm",
        }
        columns = {}
        for index, name in enumerate(names):
            values = np.asarray(result[index])
            if name in velocity_names:
                values = values / (self.km / self.s)
            columns[name] = values
        columns["survive_cdm"] = np.asarray(result[25], dtype=bool)
        columns["survive_sidm"] = np.asarray(result[26], dtype=bool)
        if validity is not None:
            columns.update(validity)
        expected_states = {"cdm_reference", "sidm"}
        if set(weight_factors) != expected_states:
            raise ValueError(f"weight_factors must contain {sorted(expected_states)}.")

        common_extra = {
            "column_units": {
                "mass": "Msun",
                "length": "Mpc",
                "velocity": "km / s",
                "density": "Msun / Mpc3",
            },
            "weight_factorization": "generation-stage",
            "validity_policy": "formed-before-accretion; shared-CDM-truncation; SIDM-profile",
            "uncomputed_sidm_value": "zero where valid_accretion is false",
            "physics_mode_equivalence": "legacy=consistent",
            "cosmology_parameters": {
                "omega_m0": float(self.OmegaM),
                "h": float(self.h),
                "omega_lambda0": float(self.OmegaL),
            },
        }
        return {
            "cdm_reference": WeightedSubhaloCatalog(
                columns=columns,
                weights=weight_factors["cdm_reference"],
                metadata=build_migration_metadata(
                    variant="sashimi-si",
                    distribution_name="sashimi-si",
                    module_file=__file__,
                    model_identifier="sashimi-si:cdm-reference:v1",
                    backend_identifier=self.itamae_backend.identifier,
                    source_identifier=(
                        "sashimi-si:upstream-physics:e17d3664dac677b604fd4ff02fb2af105a6937fa"
                    ),
                    physics_mode=self.physics_mode,
                    variance_identifier="sashimi-si:analytic-cdm-fit:v1",
                    power_identifier="sashimi-si:cdm-linear-power:v1",
                    solver_identifier="sashimi-si:gravothermal-tidal-stripping:v1",
                    extra={**common_extra, "state": "cdm_reference"},
                ),
            ),
            "sidm": WeightedSubhaloCatalog(
                columns=columns,
                weights=weight_factors["sidm"],
                metadata=build_migration_metadata(
                    variant="sashimi-si",
                    distribution_name="sashimi-si",
                    module_file=__file__,
                    model_identifier="sashimi-si:sidm-parametric:v1",
                    backend_identifier=self.itamae_backend.identifier,
                    source_identifier=(
                        "sashimi-si:upstream-physics:e17d3664dac677b604fd4ff02fb2af105a6937fa"
                    ),
                    physics_mode=self.physics_mode,
                    variance_identifier="sashimi-si:analytic-cdm-fit:v1",
                    power_identifier="sashimi-si:cdm-linear-power:v1",
                    solver_identifier="sashimi-si:gravothermal-tidal-stripping:v1",
                    extra={**common_extra, "state": "sidm"},
                ),
            ),
        }

    def subhalo_catalogs_calc(self, *args: Any, **kwargs: Any):
        """Calculate and return both CDM-reference and SIDM catalog views."""
        if "return_weight_factors" in kwargs:
            raise ValueError("subhalo_catalogs_calc manages return_weight_factors internally.")
        result, weight_factors, columns = self._calculate_population(*args, **kwargs)
        return self.catalogs_from_legacy(
            result,
            weight_factors=weight_factors,
            validity={"valid_accretion": columns["valid_accretion"]},
        )

    def subhalo_catalog_calc(self, *args: Any, state: str = "sidm", **kwargs: Any):
        """Calculate one named state view as a weighted ITAMAE catalog."""
        if state == "cdm":
            state = "cdm_reference"
        if state not in {"cdm_reference", "sidm"}:
            raise ValueError("state must be 'cdm_reference' (or alias 'cdm') or 'sidm'.")
        catalogs = self.subhalo_catalogs_calc(*args, **kwargs)
        return catalogs[state]


def migrate_class(base_class: _Base) -> _Base:
    """Create an ITAMAE-backed subclass without modifying the legacy class."""
    return type(
        f"Itamae{base_class.__name__[0].upper()}{base_class.__name__[1:]}",
        (ItamaeMigrationMixin, base_class),
        {"__module__": __name__},
    )


ItamaeHaloModel = migrate_class(halo_model)
ItamaeTidalStrippingSolver = migrate_class(TidalStrippingSolver)
ItamaeSubhaloProperties = migrate_class(subhalo_properties)


def create_itamae_model(
    *,
    backend_config: BackendConfig | None = None,
    physics_mode: str = "consistent",
    **model_parameters: Any,
) -> ItamaeSubhaloProperties:
    """Construct the explicit opt-in ITAMAE-backed SASHIMI-SI model.

    Parameters
    ----------
    backend_config
        Immutable ITAMAE backend configuration.
    physics_mode
        ``"consistent"`` by default. ``"legacy"`` is accepted for a common
        migration façade; both modes intentionally use the same SASHIMI-SI
        physical prescription.
    **model_parameters
        SASHIMI-SI physical parameters such as ``sigma0_m``, ``w``, ``beta``,
        and ``tt_th``.

    Returns
    -------
    ItamaeSubhaloProperties
        Model preserving SASHIMI-SI physics with shared ITAMAE mechanisms.
    """
    return ItamaeSubhaloProperties(
        backend_config=backend_config,
        physics_mode=physics_mode,
        **model_parameters,
    )


__all__ = [
    "create_itamae_model",
    "ItamaeHaloModel",
    "ItamaeMigrationMixin",
    "ItamaeSubhaloProperties",
    "ItamaeTidalStrippingSolver",
    "migrate_class",
]
