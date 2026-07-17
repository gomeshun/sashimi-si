"""Incremental ITAMAE migration layer for SASHIMI-SI.

Shared LCDM background calculations, Shanks acceleration, NFW mass inversion,
and structured weighted-catalog output are supplied by ITAMAE. SIDM cross
sections, gravothermal evolution, calibrated profile maps, formation-time
choices, and disruption prescriptions remain in ``sashimi_si.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

import numpy as np

from itamae.backends import BackendConfig
from itamae.cosmology import NativeFlatLCDM
from itamae.evolution import shanks_transform
from itamae.halo import invert_nfw_mass_function
from itamae.types import CatalogMetadata, WeightedSubhaloCatalog
from itamae.units import NativeUnits
from sashimi_si import TidalStrippingSolver, halo_model, subhalo_properties

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
            raise ValueError(
                "Pass either backend_config or cosmology_backend, not both."
            )
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
                "The result-preserving SASHIMI-SI migration requires "
                f"h=0.674; received {h}."
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
        backend_rho0 = (
            float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        )
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
        eps_2 = (
            self.eps_20(za, z)
            + ln_ma * self.eps_21(za, z)
            + ln_ma**2 * self.eps_22(za, z)
        )
        partial_0 = eps_0
        partial_1 = eps_0 + eps_1
        partial_2 = partial_1 + eps_2
        accelerated = shanks_transform(partial_0, partial_1, partial_2)
        with np.errstate(divide="ignore", invalid="ignore"):
            small_correction = np.abs((eps_1 + eps_2) / eps_0) < 0.02
        eps = np.where(small_correction, partial_2, accelerated)
        return ma * np.exp(eps)

    def subhalo_properties_calc(self, *args: Any, **kwargs: Any):
        """Run the legacy SIDM equations with migrated numerical components.

        Solver and NFW-inversion dependencies are attached to this instance,
        leaving the legacy module and other model instances unchanged.
        """
        return super().subhalo_properties_calc(*args, **kwargs)

    def catalogs_from_legacy(
        self,
        result,
        *,
        weight_factors: Mapping[str, Mapping[str, Any]],
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
        expected_states = {"cdm_reference", "sidm"}
        if set(weight_factors) != expected_states:
            raise ValueError(
                f"weight_factors must contain {sorted(expected_states)}."
            )

        common_extra = {
            "column_units": {
                "mass": "Msun",
                "length": "Mpc",
                "velocity": "km / s",
                "density": "Msun / Mpc3",
            },
            "weight_factorization": "generation-stage",
            "physics_mode": self.physics_mode,
            "physics_mode_equivalence": "legacy=consistent",
        }
        return {
            "cdm_reference": WeightedSubhaloCatalog(
                columns=columns,
                weights=weight_factors["cdm_reference"],
                metadata=CatalogMetadata(
                    model_identifier="sashimi-si:cdm-reference:v1",
                    backend_identifier=self.itamae_backend.identifier,
                    source_identifier="sashimi-si:upstream-physics:e17d366",
                    extra={**common_extra, "state": "cdm_reference"},
                ),
            ),
            "sidm": WeightedSubhaloCatalog(
                columns=columns,
                weights=weight_factors["sidm"],
                metadata=CatalogMetadata(
                    model_identifier="sashimi-si:sidm-parametric:v1",
                    backend_identifier=self.itamae_backend.identifier,
                    source_identifier="sashimi-si:upstream-physics:e17d366",
                    extra={**common_extra, "state": "sidm"},
                ),
            ),
        }

    def subhalo_catalogs_calc(self, *args: Any, **kwargs: Any):
        """Calculate and return both CDM-reference and SIDM catalog views."""
        if "return_weight_factors" in kwargs:
            raise ValueError(
                "subhalo_catalogs_calc manages return_weight_factors internally."
            )
        result, weight_factors = super().subhalo_properties_calc(
            *args,
            return_weight_factors=True,
            **kwargs,
        )
        return self.catalogs_from_legacy(
            result,
            weight_factors=weight_factors,
        )

    def subhalo_catalog_calc(self, *args: Any, state: str = "sidm", **kwargs: Any):
        """Calculate one named state view as a weighted ITAMAE catalog."""
        if state == "cdm":
            state = "cdm_reference"
        if state not in {"cdm_reference", "sidm"}:
            raise ValueError(
                "state must be 'cdm_reference' (or alias 'cdm') or 'sidm'."
            )
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
