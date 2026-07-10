"""Incremental ITAMAE migration layer for SASHIMI-SI.

Shared LCDM background calculations, Shanks acceleration, NFW mass inversion,
and structured weighted-catalog output are supplied by ITAMAE. SIDM cross
sections, gravothermal evolution, calibrated profile maps, formation-time
choices, and disruption prescriptions remain in ``sashimi_si.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

import numpy as np

from itamae.cosmology import NativeFlatLCDM
from itamae.evolution import shanks_transform
from itamae.halo import invert_nfw_mass_function
from itamae.types import WeightedSubhaloCatalog
import sashimi_si as _legacy
from sashimi_si import TidalStrippingSolver, halo_model, subhalo_properties

_Base = TypeVar("_Base", bound=type)


class ItamaeMigrationMixin:
    """Supply common ITAMAE mechanisms to legacy SASHIMI-SI classes.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the legacy class.
    cosmology_backend : object, optional
        ITAMAE-compatible cosmology backend. The native flat-LCDM backend is
        used by default.
    **kwargs
        Keyword arguments forwarded to the legacy class.

    Notes
    -----
    A provisional backend is attached before ``super().__init__`` because the
    tidal solver constructs interpolation tables during initialization and calls
    overridable cosmology methods. The exact legacy density normalization is
    finalized after the base class has initialized its unit constants.
    """

    def __init__(self, *args: Any, cosmology_backend: Any | None = None, **kwargs: Any) -> None:
        backend = cosmology_backend or NativeFlatLCDM()
        self.itamae_cosmology = backend
        self._rho_crit_scale = 1.0
        super().__init__(*args, **kwargs)

        if cosmology_backend is None:
            backend = NativeFlatLCDM(omega_m0=self.OmegaM, h=self.h)
            self.itamae_cosmology = backend
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

        The method temporarily injects the ITAMAE-backed tidal solver into the
        legacy module and replaces the interpolated NFW inverse with ITAMAE's
        root solve. Both changes are restored in ``finally`` blocks so importing
        this adapter does not alter other legacy instances.
        """

        old_solver = _legacy.TidalStrippingSolver
        old_ct_func = getattr(self, "ct_func", None)
        _legacy.TidalStrippingSolver = ItamaeTidalStrippingSolver
        self.ct_func = invert_nfw_mass_function
        try:
            return super().subhalo_properties_calc(*args, **kwargs)
        finally:
            _legacy.TidalStrippingSolver = old_solver
            if old_ct_func is not None:
                self.ct_func = old_ct_func

    @staticmethod
    def catalogs_from_legacy(result) -> Mapping[str, WeightedSubhaloCatalog]:
        """Convert the 27-element legacy result into named CDM and SIDM catalogs.

        Parameters
        ----------
        result : tuple
            Exact tuple returned by ``sashimi_si.subhalo_properties_calc``.

        Returns
        -------
        mapping
            Mapping with ``"cdm"`` and ``"sidm"`` weighted catalogs. The arrays
            are shared between the two views where possible.

        Notes
        -----
        Legacy ``weightCDM`` already contains the CDM survival and accretion
        masks; ``weightSIDM`` additionally contains the SIDM validity mask.
        These factors cannot be disentangled after tuple construction, so each
        catalog records ``legacy_survival_folded=True`` in its metadata.
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
        columns = {name: np.asarray(result[index]) for index, name in enumerate(names)}
        columns["survive_cdm"] = np.asarray(result[25], dtype=bool)
        columns["survive_sidm"] = np.asarray(result[26], dtype=bool)
        weight_cdm = np.asarray(result[23], dtype=float)
        weight_sidm = np.asarray(result[24], dtype=float)

        common_metadata = {
            "model": "SASHIMI-SI",
            "legacy_survival_folded": True,
            "catalog_schema": "itamae-migration-v1",
        }
        return {
            "cdm": WeightedSubhaloCatalog(
                columns=columns,
                weights={"legacy_cdm": weight_cdm},
                metadata={**common_metadata, "state": "cdm"},
            ),
            "sidm": WeightedSubhaloCatalog(
                columns=columns,
                weights={"legacy_sidm": weight_sidm},
                metadata={**common_metadata, "state": "sidm"},
            ),
        }

    def subhalo_catalogs_calc(self, *args: Any, **kwargs: Any):
        """Calculate and return both CDM-reference and SIDM catalog views."""
        return self.catalogs_from_legacy(self.subhalo_properties_calc(*args, **kwargs))

    def subhalo_catalog_calc(self, *args: Any, state: str = "sidm", **kwargs: Any):
        """Calculate one named state view as a weighted ITAMAE catalog."""
        catalogs = self.subhalo_catalogs_calc(*args, **kwargs)
        try:
            return catalogs[state]
        except KeyError as error:
            raise ValueError("state must be either 'cdm' or 'sidm'.") from error


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

__all__ = [
    "ItamaeHaloModel",
    "ItamaeMigrationMixin",
    "ItamaeSubhaloProperties",
    "ItamaeTidalStrippingSolver",
    "migrate_class",
]
