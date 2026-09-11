"""Independent accuracy check where exp(a)*Ei(-a) nearly cancels unity."""

import json
from pathlib import Path
import numpy as np
from sashimi_si import SIDM_cross_section


def test_effective_cross_section_nodes_match_high_precision_definition():
    reference = json.loads(
        (
            Path(__file__).parent / "references/effective-cross-section-high-precision.json"
        ).read_text()
    )
    model = SIDM_cross_section()
    f = model.sigma_eff_m_interpolate_analytical(1.0, reference["w_km_s"] * model.km / model.s)
    actual = f(np.asarray(reference["velocity_km_s"]) * model.km / model.s)
    np.testing.assert_allclose(actual, reference["ratio"], rtol=5e-13, atol=0)
