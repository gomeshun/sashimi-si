"""Compatibility import aliases for the standard SIDM API."""

from sashimi_si import (
    HaloModel,
    SubhaloProperties,
    TidalStrippingSolver,
    halo_model,
    subhalo_properties,
    create_itamae_model,
    SIDM_cross_section,
    SIDM_parametric_model,
)

__all__ = [
    "HaloModel",
    "SubhaloProperties",
    "TidalStrippingSolver",
    "halo_model",
    "subhalo_properties",
    "create_itamae_model",
    "SIDM_cross_section",
    "SIDM_parametric_model",
]
