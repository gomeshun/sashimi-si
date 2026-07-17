"""Public opt-in façade for the ITAMAE-backed SASHIMI-SI implementation.

Importing this module does not replace or mutate the established
``sashimi_si`` classes. The aliases below expose parallel classes that retain
the SASHIMI-SI cross sections, gravothermal evolution, formation-time choice,
and disruption prescription while using ITAMAE shared mechanisms and catalog
types.
"""

from sashimi_si_itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
    create_itamae_model,
)

halo_model = ItamaeHaloModel
TidalStrippingSolver = ItamaeTidalStrippingSolver
subhalo_properties = ItamaeSubhaloProperties

__all__ = [
    "TidalStrippingSolver",
    "create_itamae_model",
    "halo_model",
    "subhalo_properties",
]
