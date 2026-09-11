"""Variance adapter for the SASHIMI-SI ITAMAE migration.

The shared CDM variance fit remains owned by ``sashimi_si.py``. SIDM profile
evolution does not modify the initial linear-density variance, so the adapter
exposes that CDM population component independently of SIDM structural physics.
"""

from __future__ import annotations

from typing import Any

from itamae.variance import CallableVarianceModel
from ._itamae_migration import ItamaeHaloModel


def make_variance_model(model: Any | None = None) -> CallableVarianceModel:
    """Wrap the SASHIMI-SI initial-population variance implementation.

    Parameters
    ----------
    model : object, optional
        SASHIMI-SI halo model. A migrated model is constructed when omitted.

    Returns
    -------
    itamae.variance.CallableVarianceModel
        Common variance interface backed by ``sigmaMz`` and ``dsdm``.
    """

    model = model or ItamaeHaloModel()
    return CallableVarianceModel(
        identifier="sashimi-si:analytic-cdm-fit:v1",
        sigma_function=lambda mass, z: model.sigmaMz(mass, z),
        derivative_function=lambda mass, z: model.dsdm(mass, z),
    )


__all__ = ["make_variance_model"]
