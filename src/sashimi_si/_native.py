"""Immutable SIDM specifications with explicit target-epoch reference grids."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ._native_support import (
    bounds,
    choice,
    freeze,
    host_at_zero,
    integer,
    merge,
    ode_options,
    plain,
    redshift_grid,
    scalar,
    settings_identifier,
)

_DEFAULTS: dict[str, dict[str, Any]] = {
    "interaction": {
        "prescription": "yang2023",
        "sigma0_cm2_g": 147.1,
        "velocity_scale_km_s": 24.33,
        "collapse_time_ratio_cap": 1.1,
    },
    "accretion": {
        "prescription": "yang2011",
        "model": 3,
        "mass_nodes": 500,
        "redshift_step": 0.01,
        "host_history_nodes": 200,
    },
    "concentration": {"prescription": "correa2015", "scatter_dex": 0.128, "quadrature_nodes": 20},
    "stripping": {
        "prescription": "cdm-tidal",
        "solver": "pert2_shanks",
        "interpolation_nodes": 64,
        "solver_options": {},
    },
    "disruption": {"prescription": "paired-formation-truncation-profile", "ct_threshold": 0.0},
}
_SOLVERS = ("pert0", "pert1", "pert2", "pert2_shanks", "pert3", "odeint")


def _resolve(previous=None, **overrides):
    values = merge(_DEFAULTS, previous, overrides)
    for group in values:
        choice(
            values[group]["prescription"],
            group + ".prescription",
            (_DEFAULTS[group]["prescription"],),
        )
    i, a, c, s, d = (values[key] for key in _DEFAULTS)
    i["sigma0_cm2_g"] = scalar(i["sigma0_cm2_g"], "sigma0_cm2_g", positive=True)
    i["velocity_scale_km_s"] = scalar(
        i["velocity_scale_km_s"], "velocity_scale_km_s", positive=True
    )
    i["collapse_time_ratio_cap"] = scalar(i["collapse_time_ratio_cap"], "collapse_time_ratio_cap")
    if i["collapse_time_ratio_cap"] > 1.1:
        raise ValueError(
            "collapse_time_ratio_cap must lie in the native supported numerical range [0, 1.1]."
        )
    a["model"] = integer(a["model"], "accretion.model")
    if a["model"] not in (1, 2, 3):
        raise ValueError("accretion.model must be 1, 2, or 3.")
    for group, key, minimum in (
        (a, "mass_nodes", 2),
        (a, "host_history_nodes", 1),
        (c, "quadrature_nodes", 1),
        (s, "interpolation_nodes", 2),
    ):
        group[key] = integer(group[key], key, minimum)
    a["redshift_step"] = scalar(a["redshift_step"], "redshift_step", positive=True)
    c["scatter_dex"] = scalar(c["scatter_dex"], "scatter_dex")
    d["ct_threshold"] = scalar(d["ct_threshold"], "ct_threshold")
    s["solver"] = choice(s["solver"], "solver", _SOLVERS)
    s["solver_options"] = ode_options(s["solver_options"], s["solver"])
    return freeze(values)


@dataclass(frozen=True, slots=True)
class _Preparation:
    redshift_nodes: np.ndarray
    interpolation_nodes: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SIDM:
    """Reusable SI-owned model settings; every calculation has fresh state."""

    _settings: Mapping[str, Any] = field(default_factory=_resolve, init=False, repr=False)

    @property
    def resolved_settings(self):
        return self._settings

    def configure(
        self,
        *,
        interaction=None,
        accretion=None,
        concentration=None,
        stripping=None,
        disruption=None,
    ):
        """Return a validated new specification with partial process overrides."""
        result = SIDM()
        object.__setattr__(
            result,
            "_settings",
            _resolve(
                self._settings,
                interaction=interaction,
                accretion=accretion,
                concentration=concentration,
                stripping=stripping,
                disruption=disruption,
            ),
        )
        return result

    def population(
        self,
        *,
        host_mass_msun,
        host_mass_definition="200c",
        host_mass_redshift=0.0,
        redshift=0.0,
        reference_mass_range_msun=(1e6, None),
        reference_mass_definition="200c",
        reference_mass_redshift=None,
        accretion_redshift_range=None,
        state="paired",
    ):
        """Return paired CDM-reference/SIDM catalogs, or one explicit state.

        SI's mass coordinate is a reference M200c grid at the output epoch,
        not a fixed accretion-mass grid. Actual accretion masses are evolved
        backward through host-history relations and formation gates. The
        optional upper reference bound remains 0.1*host_mass_at_z0, preserving
        the legacy convention despite the reference grid's nonzero epoch.
        If formation selection removes every candidate redshift node, raise
        ValueError before solver setup rather than return empty catalogs.
        """
        choice(state, "state", ("paired", "cdm_reference", "sidm"))
        choice(host_mass_definition, "host_mass_definition", ("200c",))
        choice(reference_mass_definition, "reference_mass_definition", ("200c",))
        mass = scalar(host_mass_msun, "host_mass_msun", positive=True)
        epoch = scalar(host_mass_redshift, "host_mass_redshift")
        target = scalar(redshift, "redshift")
        reference_epoch = (
            target
            if reference_mass_redshift is None
            else scalar(reference_mass_redshift, "reference_mass_redshift")
        )
        if reference_epoch != target:
            raise ValueError(
                "SI reference_mass_redshift must equal output redshift; an accretion-mass grid is not equivalent."
            )
        lo, hi = bounds(reference_mass_range_msun, "reference_mass_range_msun", mass=True)
        i, a, c, s, d = (self._settings[key] for key in _DEFAULTS)
        nodes, zlo, zhi, policy = redshift_grid(
            target, accretion_redshift_range, a["redshift_step"], 5.0
        )
        from . import SubhaloProperties

        model = SubhaloProperties(
            sigma0_m=i["sigma0_cm2_g"],
            w=i["velocity_scale_km_s"],
            tt_th=i["collapse_time_ratio_cap"],
        )
        mass0 = host_at_zero(model, mass, epoch)
        upper_policy = "0.1*host-M200c-at-z0" if hi is None else "explicit"
        hi = 0.1 * mass0 if hi is None else hi
        if lo >= hi:
            raise ValueError("Resolved reference mass upper bound must exceed lower bound.")
        metadata = {
            "native_api": "sashimi-si:immutable-specification:v1",
            "resolved_settings": plain(self._settings),
            "native_configuration_identifier": settings_identifier("sashimi-si", self._settings),
            "host_mass_msun": mass,
            "host_mass_definition": "200c",
            "host_mass_redshift": epoch,
            "host_mass_z0": mass0,
            "target_redshift": target,
            "reference_mass_range_msun": [lo, hi],
            "reference_mass_definition": "200c",
            "reference_mass_redshift": reference_epoch,
            "reference_mass_upper_policy": upper_policy,
            "mass_coordinate": "target-epoch-reference-M200c; backward-evolved accretion masses",
            "accretion_redshift_range": [zlo, zhi],
            "accretion_candidate_redshift_nodes": nodes.tolist(),
            "accretion_redshift_range_policy": policy,
            "paired_states": ["cdm_reference", "sidm"],
            "independent_state_survival": True,
            "collapse_time_ratio_cap_units": "dimensionless (t-t_formation)/t_collapse",
            "collapse_ratio_cap_policy": "native numerical support [0,1.1]; not an independent calibration claim",
            "inactive_legacy_parameters": {"beta": 4},
            "fixed_numerics": {
                "pre_accretion_history_nodes": 100,
                "evolution_history_nodes": 100,
                "accretion_auxiliary_redshift_nodes": 1000,
                "host_inversion_nodes": 1000,
                "host_inversion_log10_bracket": [0.0, 3.0],
            },
            "stripping_interpolation_nodes": s["interpolation_nodes"],
        }
        catalogs = model._calculate_population(
            mass0,
            redshift=target,
            dz=a["redshift_step"],
            zmax=zhi,
            N_ma=a["mass_nodes"],
            sigmalogc=c["scatter_dex"],
            N_herm=c["quadrature_nodes"],
            logmamin=np.log10(lo),
            logmamax=np.log10(hi),
            N_hermNa=a["host_history_nodes"],
            Na_model=a["model"],
            ct_th=d["ct_threshold"],
            M0_at_redshift=False,
            method=s["solver"],
            preparation=_Preparation(nodes, s["interpolation_nodes"], metadata),
            **dict(s["solver_options"]),
        )
        return MappingProxyType(catalogs) if state == "paired" else catalogs[state]
