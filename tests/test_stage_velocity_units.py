"""The shared executor receives km/s at every SI state boundary."""

import json
from pathlib import Path

import numpy as np

from sashimi_si import SubhaloProperties
from sashimi_si_itamae_components import SIDMInitialStructure, SIDMProfileEvolution


def test_real_stage_velocities_follow_canonical_nfw_identity(monkeypatch):
    initial_states, evolved_states = [], []
    initialize = SIDMInitialStructure.initialize
    evolve = SIDMProfileEvolution.evolve

    def record_initial(self, *args):
        state = initialize(self, *args)
        initial_states.append(state)
        return state

    def record_evolved(self, *args):
        state = evolve(self, *args)
        evolved_states.append(state)
        return state

    monkeypatch.setattr(SIDMInitialStructure, "initialize", record_initial)
    monkeypatch.setattr(SIDMProfileEvolution, "evolve", record_evolved)
    parameters = json.loads((Path(__file__).parent / "references/B-all.json").read_text())[
        "calculation"
    ]["parameters"]
    model = SubhaloProperties()
    catalogs = model.subhalo_catalogs_calc(**parameters)
    g_canonical = model.G / (model.km / model.s) ** 2
    for states, suffix in ((initial_states, "cdm_acc"), (evolved_states, "cdm")):
        assert states
        for state in states:
            expected = np.sqrt(
                4 * np.pi * g_canonical * state[f"rho_s_{suffix}"] / 4.625
            ) * state[f"r_s_{suffix}"]
            np.testing.assert_allclose(state[f"v_max_{suffix}"], expected, rtol=5e-15)
    for name in ("v_max_cdm", "v_max_sidm", "v_max_cdm_acc", "v_max_sidm_acc"):
        states = initial_states if name == "v_max_cdm_acc" else evolved_states
        np.testing.assert_array_equal(
            catalogs["sidm"].columns[name], np.concatenate([state[name] for state in states])
        )
