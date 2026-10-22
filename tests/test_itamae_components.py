import numpy as np

from sashimi_si_itamae_components import SIDMSurvival


def test_survival_combines_shared_cdm_gate_with_sidm_profile_validity():
    """Combine tidal disruption with physically admissible SIDM structure.

    Notes
    -----
    Pass criteria: The toy c_t array gives CDM mask [False,True,True,True]; negative SIDM
    core, radius or velocity reduces it to [False,True,False,False].

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    evolved = {
        name: np.ones(4)
        for name in (
            "v_max_sidm",
            "rmax_sidm",
            "v_max_sidm_acc",
            "rmax_sidm_acc",
            "r_c_sidm",
            "r_c_sidm_acc",
        )
    }
    evolved["valid_accretion"] = np.ones(4, dtype=bool)
    evolved["c_t_cdm"] = np.array([0.1, 0.2, 0.3, 0.4])
    evolved["r_c_sidm"][2] = -1.0
    evolved["rmax_sidm"][3] = -1.0
    evolved["v_max_sidm"][3] = -1.0
    masks = SIDMSurvival(ct_threshold=0.1).select(None, {}, evolved, None)
    np.testing.assert_array_equal(masks["cdm_reference"], [False, True, True, True])
    np.testing.assert_array_equal(masks["sidm"], [False, True, False, False])
