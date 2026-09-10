"""Import aliases to the standard API; no alternative calculation path."""

from sashimi_si import (
    HaloModel as ItamaeHaloModel,
    SubhaloProperties as ItamaeSubhaloProperties,
    TidalStrippingSolver as ItamaeTidalStrippingSolver,
    create_itamae_model,
)

__all__ = [
    "ItamaeHaloModel",
    "ItamaeSubhaloProperties",
    "ItamaeTidalStrippingSolver",
    "create_itamae_model",
]
