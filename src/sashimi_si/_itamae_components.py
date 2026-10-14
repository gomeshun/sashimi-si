"""SIDM prescriptions and paired states composed through ITAMAE.

The pre-accretion histories, cross section, collapse/profile maps and survival
rules belong to SASHIMI-SI. Components share node identity and base weights;
CDM reference and SIDM survival remain separate. Stage states use canonical
Msun, Mpc, km/s and Msun/Mpc^3. The SI-owned history context and physical kernels
retain their historical Mpc/s velocity unit; conversions are explicit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from itamae.measure import build_accretion_batch
from itamae.numerics import gauss_hermite_lognormal


@dataclass(frozen=True, slots=True)
class SIDMAccretionSlices:
    """Build concentration nodes and pre-accretion CDM histories.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    zdist_accreted : numpy.ndarray
        Accretion-redshift slices having at least one node that formed before accretion.
    ma_matrix_accreted : numpy.ndarray
        Virial accretion masses in Msun, shape (n_active_redshift,n_mass).
    z_f : numpy.ndarray
        Dimensionless formation redshift for each halo-mass track; valid accretion requires
        z_f > z_acc.
    ma200_0 : numpy.ndarray
        M200 reference masses at z=0 used to construct pre-accretion halo histories.
    sigmalogc : float
        Nonnegative standard deviation of log10 concentration, in dex.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.
    ma200_matrix_accreted : numpy.ndarray
        M200 accretion masses in Msun on the same active redshift/mass grid.
    weight_base_active : numpy.ndarray
        EPS expected-count factors on the active redshift/mass slices, before concentration
        and survival factors.
    t_f : numpy.ndarray
        Formation cosmic time in SI internal seconds, aligned with each halo-mass track.

    Notes
    -----
    Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
    Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
    in concentration-major then mass order, and the same generation weights support both
    survival views.
    """

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
        r"""Build concentration quadrature and pre-accretion CDM histories for one active epoch.

        Parameters
        ----------
        index : int
            Index of the selected accretion-redshift slice in zdist and population_2d.

        Returns
        -------
        tuple
            Aligned AccretionBatch plus context containing 100-node formation-to-accretion
            tracks in internal velocity units.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
        Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
        in concentration-major then mass order, and the same generation weights support both
        survival views.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691
        """
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
    """Extract the accreted CDM structure at each shared node.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.

    Notes
    -----
    Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
    Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
    in concentration-major then mass order, and the same generation weights support both
    survival views.
    """

    model: Any

    def initialize(self, batch, context):
        r"""Extract the accreted NFW reference state and convert its circular speed to km/s.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Flattened CDM scale radius/density, rmax and Vmax at accretion in canonical stage
            units.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
        Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
        in concentration-major then mass order, and the same generation weights support both
        survival views.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691
        """
        return {
            "r_s_cdm_acc": (context["rmax_ba"][:, -1, :] / 2.1626).reshape(-1),
            "rho_s_cdm_acc": (
                4.625
                / (4.0 * np.pi * self.model.G)
                * (context["Vmax_ba"][:, -1, :] / (context["rmax_ba"][:, -1, :] / 2.1626)) ** 2
            ).reshape(-1),
            "rmax_cdm_acc": context["rmax_ba"][:, -1, :].reshape(-1),
            "v_max_cdm_acc": context["Vmax_ba"][:, -1, :].reshape(-1)
            / (self.model.km / self.model.s),
        }


@dataclass(frozen=True, slots=True)
class SIDMProfileEvolution:
    """Evolve tidal mass and the paired CDM/SIDM profile histories.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    solver : TidalStrippingSolver
        SI-owned tidal provider preserving its background, host history and SI internal mass
        units.
    redshift : float or array_like
        Dimensionless redshift. The cosmological background requires z > -1; population
        nodes use nonnegative redshifts.
    method : str
        picard (default), dop853, odeint, or an explicitly selected retained perturbative
        approximation.
    kwargs : dict
        Additional keyword options forwarded to the called routine; reserved scientific
        controls remain owned by that routine.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.
    ma200_z0 : numpy.ndarray
        Reference M200 grid at the requested final epoch, used to align post-accretion
        histories.

    Notes
    -----
    Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
    Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
    in concentration-major then mass order, and the same generation weights support both
    survival views.
    """

    model: Any
    solver: Any
    redshift: float
    method: str
    kwargs: dict[str, Any]
    N_herm: int
    ma200_z0: np.ndarray

    def evolve(self, batch, initial, context):
        r"""Evolve the tidal mass history and gravothermal SIDM response on the same CDM reference
        tracks.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Aligned CDM/SIDM profiles, formation-validity mask and collapse-time ratios;
            unformed SIDM entries are explicit zeros.

        Notes
        -----
        The factors 2**0.4*mu**0.3/(1+mu)**0.4 and 2**(-0.3)*mu**0.4*(1+mu)**0.3 are the cuspy
        tidal tracks of Penarrubia et al., reproduced in Hiroshima et al., Eqs. (8)-(9), with mu
        the retained bound-mass fraction. The NFW profile is from Navarro et al. The rounded
        factors rmax/rs = 2.163 (2.1626 where explicitly used) and 4.625 are NFW maximum-
        circular-speed identities, as written in Hiroshima et al., Eqs. (5)-(7), rather than
        independent empirical fits.

        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
        Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
        in concentration-major then mass order, and the same generation weights support both
        survival views.

        References
        ----------

        .. [cite-tracks] Penarrubia et al. (2010), The impact of dark matter cusps and cores on the satellite galaxy population around spiral galaxies, Eq. (8), Fig. 6.
            https://arxiv.org/abs/1002.3376

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107
        """
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
            initial["v_max_cdm_acc"].reshape(self.N_herm, n_mass) * (self.model.km / self.model.s),
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
            "v_max_sidm_acc": v_max_sidm_acc.reshape(-1) / (self.model.km / self.model.s),
            "m_bound": np.broadcast_to(m_aa[-1], (self.N_herm, n_mass)).reshape(-1),
            "r_s_cdm": r_s_cdm.reshape(-1),
            "rho_s_cdm": rho_s_cdm.reshape(-1),
            "rmax_cdm": rmax_cdm.reshape(-1),
            "v_max_cdm": v_max_cdm.reshape(-1) / (self.model.km / self.model.s),
            "r_s_sidm": r_s_sidm.reshape(-1),
            "rho_s_sidm": rho_s_sidm.reshape(-1),
            "r_c_sidm": r_c_sidm.reshape(-1),
            "rmax_sidm": rmax_sidm.reshape(-1),
            "v_max_sidm": Vmax_sidm.reshape(-1) / (self.model.km / self.model.s),
            "c_t_cdm": c_t_cdm.reshape(-1),
            "collapse_time_ratio": tt_ratio.reshape(-1),
        }


@dataclass(frozen=True, slots=True)
class SIDMSurvival:
    r"""Apply C-truncation and SIDM-profile validity rules to paired states.

    Parameters
    ----------
    ct_threshold : float
        Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
        is applied once through its mask.

    Notes
    -----
    The literature threshold rt/rs > 0.77 follows the Hayashi et al. disruption prescription
    adopted by Hiroshima et al. A caller-supplied threshold takes precedence in variants
    that expose it (C and SI currently default to 0). This is a model selection rule; an
    exact-boundary test does not calibrate disruption physics.

    Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
    Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
    in concentration-major then mass order, and the same generation weights support both
    survival views.

    References
    ----------

    .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
        https://arxiv.org/abs/astro-ph/0203004

    .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
        https://arxiv.org/abs/1803.07691
    """

    ct_threshold: float

    def select(self, batch, initial, evolved, context):
        r"""Apply formation, CDM truncation and SIDM profile-domain conditions to paired states.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict[str, numpy.ndarray]
            cdm_reference mask requires formation and c_t>threshold; sidm also requires the
            listed SIDM speeds/radii/core radii to be nonnegative.

        Notes
        -----
        The literature threshold rt/rs > 0.77 follows the Hayashi et al. disruption prescription
        adopted by Hiroshima et al. A caller-supplied threshold takes precedence in variants
        that expose it (C and SI currently default to 0). This is a model selection rule; an
        exact-boundary test does not calibrate disruption physics.

        Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
        Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
        in concentration-major then mass order, and the same generation weights support both
        survival views.

        References
        ----------

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691
        """
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
    """Name both states without changing units or row order.

    Notes
    -----
    Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
    Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
    in concentration-major then mass order, and the same generation weights support both
    survival views.
    """

    def build(self, batch, initial, evolved, survival_masks, context):
        """Expose both physical state views on the same node axis without combining their weights.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        survival_masks : Mapping[str, numpy.ndarray]
            Named boolean survival decisions aligned with the physical state; kept separate from
            population and concentration weights.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Canonical-unit CDM/SIDM columns and separate survival masks.

        Notes
        -----
        Stage arrays use Msun, Mpc, km/s and Msun/Mpc^3, but the SI history/context kernels use
        Mpc/s. Every velocity conversion occurs explicitly at that boundary. Nodes are flattened
        in concentration-major then mass order, and the same generation weights support both
        survival views.
        """
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
