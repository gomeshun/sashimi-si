"""SIDM prescriptions and paired states composed through ITAMAE.

The pre-accretion histories, cross section, collapse/profile maps and survival
rules belong to SASHIMI-SI. Components share node identity and base weights;
CDM reference and SIDM survival remain separate. Internal historical units are
converted at the adapter's explicit catalog boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from itamae.measure import build_accretion_batch
from itamae.numerics import gauss_hermite_lognormal


@dataclass(frozen=True, slots=True)
class SIDMAccretionSlices:
    """Build concentration nodes and pre-accretion CDM histories."""

    model: Any
    zdist_accreted: np.ndarray
    ma_matrix_accreted: np.ndarray
    z_f: np.ndarray
    ma200_0: np.ndarray
    sigmalogc: float
    N_herm: int
    ma200_matrix_accreted: np.ndarray
    weight_base_active: np.ndarray
    t_f: np.ndarray

    def build(self, index: int):
        za = self.zdist_accreted[index]
        ma = self.ma_matrix_accreted[index]
        z_ba = np.linspace(self.z_f, za, 100)
        m200_ba = self.model.Mzi(self.ma200_0, z_ba)
        c200_med_ba = self.model.conc200(m200_ba, z_ba)
        r200_ba = (
            3.0 * m200_ba / (4.0 * np.pi * self.model.rhocrit0 * self.model.g(z_ba) * 200.0)
        ) ** (1.0 / 3.0)
        c200_ba, concentration_weights_ba = gauss_hermite_lognormal(
            c200_med_ba,
            self.sigmalogc,
            order=self.N_herm,
        )
        rs_ba = r200_ba / c200_ba
        rhos_ba = m200_ba / (4.0 * np.pi * rs_ba**3 * self.model.fc(c200_ba))
        rmax_ba = 2.1626 * rs_ba
        Vmax_ba = np.sqrt(rhos_ba * 4.0 * np.pi * self.model.G / 4.625) * rs_ba
        batch = build_accretion_batch(
            self.ma200_matrix_accreted[index],
            za,
            c200_ba[:, -1, :],
            self.weight_base_active[index],
            concentration_weights_ba[:, -1, :],
            mvir_acc=ma,
            metadata={
                "model": "sashimi-si",
                "calculation_specification": self.model.calculation_specification,
            },
        )
        context = {
            "redshift_index": index,
            "ma": ma,
            "za": za,
            "z_ba": z_ba,
            "rmax_ba": rmax_ba,
            "Vmax_ba": Vmax_ba,
            "t_f": self.t_f,
            "valid_accretion": self.z_f > za,
        }
        return batch, context


@dataclass(frozen=True, slots=True)
class SIDMInitialStructure:
    """Extract the accreted CDM structure at each shared node."""

    model: Any

    def initialize(self, batch, context):
        return {
            "r_s_cdm_acc": (context["rmax_ba"][:, -1, :] / 2.1626).reshape(-1),
            "rho_s_cdm_acc": (
                4.625
                / (4.0 * np.pi * self.model.G)
                * (context["Vmax_ba"][:, -1, :] / (context["rmax_ba"][:, -1, :] / 2.1626)) ** 2
            ).reshape(-1),
            "rmax_cdm_acc": context["rmax_ba"][:, -1, :].reshape(-1),
            "v_max_cdm_acc": context["Vmax_ba"][:, -1, :].reshape(-1),
        }


@dataclass(frozen=True, slots=True)
class SIDMProfileEvolution:
    """Evolve tidal mass and the paired CDM/SIDM profile histories."""

    model: Any
    solver: Any
    redshift: float
    method: str
    kwargs: dict[str, Any]
    N_herm: int
    ma200_z0: np.ndarray

    def evolve(self, batch, initial, context):
        n_mass = context["ma"].size
        ma = context["ma"]
        za = context["za"]
        zcalc = np.linspace(za, self.redshift, 100)
        m_aa = self.solver.subhalo_mass_stripped(
            ma,
            za,
            zcalc,
            method=self.method,
            **self.kwargs,
        )
        rmax_acc = np.expand_dims(
            initial["rmax_cdm_acc"].reshape(self.N_herm, n_mass),
            axis=1,
        )
        Vmax_acc = np.expand_dims(
            initial["v_max_cdm_acc"].reshape(self.N_herm, n_mass),
            axis=1,
        )
        Vmax_aa = Vmax_acc * (2.0**0.4 * (m_aa / ma) ** 0.3 * (1.0 + m_aa / ma) ** -0.4)
        rmax_aa = rmax_acc * (2.0**-0.3 * (m_aa / ma) ** 0.4 * (1.0 + m_aa / ma) ** 0.3)
        rmax_cdm = rmax_aa[:, -1, :]
        v_max_cdm = Vmax_aa[:, -1, :]
        r_s_cdm = rmax_cdm / 2.1626
        rho_s_cdm = (4.625 / (4.0 * np.pi * self.model.G)) * (v_max_cdm / r_s_cdm) ** 2
        c_t_cdm = self.model.ct_func(m_aa[-1] / (4.0 * np.pi * rho_s_cdm * r_s_cdm**3))

        z = np.concatenate(
            (
                context["z_ba"],
                zcalc[1:].reshape(-1, 1) * np.ones_like(self.ma200_z0),
            ),
            axis=0,
        )
        t = self.model.t_U - self.model.lookback_time(z)
        Vmax_CDM = np.concatenate(
            (context["Vmax_ba"], Vmax_aa[:, 1:]),
            axis=1,
        )
        rmax_CDM = np.concatenate(
            (context["rmax_ba"], rmax_aa[:, 1:]),
            axis=1,
        )
        # Formation is a domain boundary, already a zero-weight gate in the
        # archived SI model. Never integrate a SIDM history backwards in time.
        valid = context["valid_accretion"]
        shape = (self.N_herm, n_mass)
        outputs = [np.zeros(shape) for _ in range(5)]
        acc_outputs = [np.zeros(shape) for _ in range(5)]
        tt_ratio = np.zeros(shape)
        if np.any(valid):
            t_c = self.model.t_collapse(
                self.model.sigma_eff_m(Vmax_CDM[..., valid]),
                rmax_CDM[..., valid],
                Vmax_CDM[..., valid],
            )
            tt_ratio[:, valid] = ((self.model.t_U - context["t_f"][valid]) / t_c)[:, -1, :]
            evolved_valid = self.model.param_model.master_function(
                Vmax_CDM[..., valid],
                rmax_CDM[..., valid],
                t[..., valid],
                context["t_f"][valid],
            )
            t2 = self.model.t_U - self.model.lookback_time(context["z_ba"][..., valid])
            acc_valid = self.model.param_model.master_function(
                context["Vmax_ba"][..., valid],
                context["rmax_ba"][..., valid],
                t2,
                context["t_f"][valid],
            )
            for destination, value in zip(outputs, evolved_valid, strict=True):
                destination[:, valid] = value
            for destination, value in zip(acc_outputs, acc_valid, strict=True):
                destination[:, valid] = value
        Vmax_sidm, rmax_sidm, rho_s_sidm, r_s_sidm, r_c_sidm = outputs
        v_max_sidm_acc, rmax_sidm_acc, rho_s_sidm_acc, r_s_sidm_acc, r_c_sidm_acc = acc_outputs
        return {
            "valid_accretion": np.broadcast_to(valid, shape).reshape(-1),
            "r_s_sidm_acc": r_s_sidm_acc.reshape(-1),
            "rho_s_sidm_acc": rho_s_sidm_acc.reshape(-1),
            "r_c_sidm_acc": r_c_sidm_acc.reshape(-1),
            "rmax_sidm_acc": rmax_sidm_acc.reshape(-1),
            "v_max_sidm_acc": v_max_sidm_acc.reshape(-1),
            "m_bound": np.broadcast_to(m_aa[-1], (self.N_herm, n_mass)).reshape(-1),
            "r_s_cdm": r_s_cdm.reshape(-1),
            "rho_s_cdm": rho_s_cdm.reshape(-1),
            "rmax_cdm": rmax_cdm.reshape(-1),
            "v_max_cdm": v_max_cdm.reshape(-1),
            "r_s_sidm": r_s_sidm.reshape(-1),
            "rho_s_sidm": rho_s_sidm.reshape(-1),
            "r_c_sidm": r_c_sidm.reshape(-1),
            "rmax_sidm": rmax_sidm.reshape(-1),
            "v_max_sidm": Vmax_sidm.reshape(-1),
            "c_t_cdm": c_t_cdm.reshape(-1),
            "collapse_time_ratio": tt_ratio.reshape(-1),
        }


@dataclass(frozen=True, slots=True)
class SIDMSurvival:
    """Apply C-truncation and SIDM-profile validity rules to paired states."""

    ct_threshold: float

    def select(self, batch, initial, evolved, context):
        formed = evolved["valid_accretion"]
        cdm = formed & (evolved["c_t_cdm"] > self.ct_threshold)
        profile_valid = np.ones_like(formed, dtype=bool)
        for name in (
            "v_max_sidm",
            "rmax_sidm",
            "v_max_sidm_acc",
            "rmax_sidm_acc",
            "r_c_sidm",
            "r_c_sidm_acc",
        ):
            profile_valid &= evolved[name] >= 0.0
        return {"cdm_reference": cdm, "sidm": cdm & profile_valid}


@dataclass(frozen=True, slots=True)
class SIDMCatalogColumns:
    """Name both states without changing units or row order."""

    def build(self, batch, initial, evolved, survival_masks, context):
        return {
            "valid_accretion": evolved["valid_accretion"],
            "m200_acc": batch.m200_acc,
            "z_acc": batch.z_acc,
            "r_s_cdm_acc": initial["r_s_cdm_acc"],
            "rho_s_cdm_acc": initial["rho_s_cdm_acc"],
            "rmax_cdm_acc": initial["rmax_cdm_acc"],
            "v_max_cdm_acc": initial["v_max_cdm_acc"],
            "r_s_sidm_acc": evolved["r_s_sidm_acc"],
            "rho_s_sidm_acc": evolved["rho_s_sidm_acc"],
            "r_c_sidm_acc": evolved["r_c_sidm_acc"],
            "rmax_sidm_acc": evolved["rmax_sidm_acc"],
            "v_max_sidm_acc": evolved["v_max_sidm_acc"],
            "m_bound": evolved["m_bound"],
            "r_s_cdm": evolved["r_s_cdm"],
            "rho_s_cdm": evolved["rho_s_cdm"],
            "rmax_cdm": evolved["rmax_cdm"],
            "v_max_cdm": evolved["v_max_cdm"],
            "r_s_sidm": evolved["r_s_sidm"],
            "rho_s_sidm": evolved["rho_s_sidm"],
            "r_c_sidm": evolved["r_c_sidm"],
            "rmax_sidm": evolved["rmax_sidm"],
            "v_max_sidm": evolved["v_max_sidm"],
            "c_t_cdm": evolved["c_t_cdm"],
            "collapse_time_ratio": evolved["collapse_time_ratio"],
            "survive_cdm": survival_masks["cdm_reference"],
            "survive_sidm": survival_masks["sidm"],
        }


__all__ = [
    "SIDMAccretionSlices",
    "SIDMInitialStructure",
    "SIDMProfileEvolution",
    "SIDMSurvival",
    "SIDMCatalogColumns",
]
