import numpy as np

from sashimi_si_itamae_components import SIDMSurvival


def test_survival_keeps_cdm_and_sidm_validity_independent():
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
    evolved["c_t_cdm"] = np.array([0.1, 0.2, 0.3, 0.4])
    evolved["r_c_sidm"][2] = -1.0
    evolved["rmax_sidm"][3] = -1.0
    evolved["v_max_sidm"][3] = -1.0
    masks = SIDMSurvival(ct_threshold=0.1).select(None, {}, evolved, None)
    np.testing.assert_array_equal(masks["cdm_reference"], [False, True, True, True])
    np.testing.assert_array_equal(masks["sidm"], [True, True, False, False])
