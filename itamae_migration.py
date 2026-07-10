"""ITAMAE compatibility layer for incremental SASHIMI-SI migration.

The adapter delegates shared background-cosmology calculations to ITAMAE while
leaving the SIDM cross-section, gravothermal evolution, disruption criteria, and
public legacy classes unchanged. This separation allows direct regression tests
before duplicated cosmology code is removed from ``sashimi_si.py``.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from itamae.cosmology import NativeFlatLCDM
from sashimi_si import halo_model

_Base = TypeVar("_Base", bound=type)


class ItamaeCosmologyMixin:
    """Delegate shared cosmology calculations to an ITAMAE backend.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the legacy SASHIMI-SI class.
    cosmology_backend : object, optional
        Object implementing the ITAMAE cosmology interface. When omitted, the
        native flat-LCDM backend is initialized from the legacy ``OmegaM`` and
        ``h`` values.
    **kwargs
        Keyword arguments forwarded to the legacy SASHIMI-SI class.

    Notes
    -----
    SASHIMI-SI uses Mpc, solar mass, and seconds as dimensionless floating-point
    base units. ITAMAE returns Hubble rates in km s^-1 Mpc^-1 and densities in
    Msun Mpc^-3. This mixin converts those quantities explicitly at the boundary.

    The historical SASHIMI constants are rounded differently from the constants
    used by ITAMAE. The adapter therefore preserves the legacy critical-density
    normalization at redshift zero and delegates only its redshift dependence.
    This prevents a convention-only change from being mistaken for SIDM physics.
    """

    def __init__(self, *args: Any, cosmology_backend: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        backend = cosmology_backend or NativeFlatLCDM(omega_m0=self.OmegaM, h=self.h)
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
        """Return the Hubble rate in the legacy inverse-second unit.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift.

        Returns
        -------
        numpy.ndarray
            Hubble rate with the broadcast shape of ``z``.
        """

        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the legacy SASHIMI-SI unit convention.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift.

        Returns
        -------
        numpy.ndarray
            Critical density in solar masses per cubic Mpc.
        """

        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        """Return the normalized linear growth factor.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift.

        Returns
        -------
        numpy.ndarray
            Dimensionless growth factor normalized to unity at redshift zero.
        """

        return np.asarray(self.itamae_cosmology.growth_factor(z))


def migrate_class(base_class: _Base) -> _Base:
    """Create an ITAMAE-backed subclass without modifying the legacy class.

    Parameters
    ----------
    base_class : type
        Legacy SASHIMI-SI class to adapt.

    Returns
    -------
    type
        Dynamically constructed subclass using :class:`ItamaeCosmologyMixin`.
    """

    return type(
        f"Itamae{base_class.__name__[0].upper()}{base_class.__name__[1:]}",
        (ItamaeCosmologyMixin, base_class),
        {"__module__": __name__},
    )


ItamaeHaloModel = migrate_class(halo_model)

__all__ = ["ItamaeCosmologyMixin", "ItamaeHaloModel", "migrate_class"]
