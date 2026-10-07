"""Native SIDM contracts against independent pre-edit installed results."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from itamae.types import WeightedSubhaloCatalog

from sashimi_si import SIDM, SubhaloProperties

ROOT = Path(__file__).parent / "references/native-api-baseline"


def configured(name="default"):
    record = json.loads((ROOT / (name + ".json")).read_text())
    p, constructor = record["parameters"], record["constructor"]
    groups = {
        "accretion": {
            "mass_nodes": p["N_ma"],
            "redshift_step": p["dz"],
            "host_history_nodes": p["N_hermNa"],
        },
        "concentration": {"scatter_dex": p["sigmalogc"], "quadrature_nodes": p["N_herm"]},
    }
    groups["interaction"] = {
        "sigma0_cm2_g": constructor.get("sigma0_m", 147.1),
        "velocity_scale_km_s": constructor.get("w", 24.33),
        "collapse_time_ratio_cap": constructor.get("tt_th", 1.1),
    }
    groups["accretion"]["model"] = p["Na_model"]
    groups["stripping"] = {
        "solver": p["method"],
        "solver_options": {k: p[k] for k in ("rtol", "atol") if k in p},
    }
    groups["disruption"] = {"ct_threshold": p["ct_th"]}
    request = {
        "host_mass_msun": p["M0"],
        "redshift": p["redshift"],
        "host_mass_redshift": p["redshift"] if p.get("M0_at_redshift") else 0.0,
        "reference_mass_range_msun": (10 ** p["logmamin"], 10 ** p["logmamax"]),
        "reference_mass_redshift": p["redshift"],
        "accretion_redshift_range": (p["redshift"], p["zmax"]),
    }
    return SIDM().configure(**groups), request, record


def as_pair(result):
    return (
        result
        if isinstance(result, dict) or not hasattr(result, "columns")
        else {"default": result}
    )


@pytest.mark.parametrize("name", [p.stem for p in ROOT.glob("*.json")])
def test_native_and_legacy_match_independent_baseline(name):
    model, request, record = configured(name)
    assert record["source"] == "ce5a3c11518a609ac056bb136653268b05b1ecd7"
    path = ROOT / (name + ".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    native = as_pair(model.population(**request))
    legacy = SubhaloProperties(**record["constructor"]).subhalo_catalogs_calc(
        **record["parameters"]
    )
    with np.load(path) as saved:
        for state, catalog in native.items():
            for key, array in {**catalog.columns, **catalog.weights}.items():
                expected = saved[state + "__" + key]
                if array.dtype.kind == "b":
                    np.testing.assert_array_equal(array, expected)
                else:
                    np.testing.assert_allclose(array, expected, rtol=5e-12, atol=1e-300)
            for key, array in catalog.columns.items():
                np.testing.assert_array_equal(array, legacy[state].columns[key])
            for key, array in catalog.weights.items():
                np.testing.assert_array_equal(array, legacy[state].weights[key])


def test_configuration_is_detached_immutable_and_partial():
    model, _, _ = configured()
    values = {"scatter_dex": 0.23}
    next_model = model.configure(concentration=values)
    values["scatter_dex"] = 999
    assert next_model.resolved_settings["concentration"]["scatter_dex"] == 0.23
    assert model.resolved_settings["concentration"]["scatter_dex"] == 0.128
    assert next_model.resolved_settings["accretion"] == model.resolved_settings["accretion"]
    with pytest.raises(TypeError):
        next_model.resolved_settings["concentration"]["scatter_dex"] = 0
    with pytest.raises(FrozenInstanceError):
        next_model._settings = {}


def test_repeated_runs_do_not_reuse_mutable_population_state():
    model, request, _ = configured()
    first = as_pair(model.population(**request))
    saved = {state: {k: v.copy() for k, v in cat.columns.items()} for state, cat in first.items()}
    model.configure(concentration={"scatter_dex": 0.2}).population(
        **{**request, "host_mass_msun": 2e10}
    )
    again = as_pair(model.population(**request))
    for state in saved:
        for key, value in saved[state].items():
            np.testing.assert_array_equal(value, first[state].columns[key])
            np.testing.assert_array_equal(value, again[state].columns[key])
    assert not hasattr(model, "catalog") and not hasattr(model, "M0")


def test_executed_provenance_and_weights_roundtrip(tmp_path):
    model, request, _ = configured()
    products = as_pair(model.population(**request))
    for state, catalog in products.items():
        assert (
            catalog.metadata["resolved_settings"]["accretion"]["mass_nodes"]
            == model.resolved_settings["accretion"]["mass_nodes"]
        )
        assert catalog.metadata["host_mass_redshift"] == 0.0
        path = tmp_path / (state + ".npz")
        catalog.to_npz(path)
        restored = WeightedSubhaloCatalog.from_npz(path)
        assert restored.metadata == catalog.metadata
        for key in catalog.weights:
            np.testing.assert_array_equal(restored.weights[key], catalog.weights[key])
        np.testing.assert_array_equal(
            catalog.weight_final, np.prod(list(catalog.weights.values()), axis=0)
        )


@pytest.mark.parametrize(
    "options",
    [
        {"rtol": 0.0},
        {"h0": 0.1},
        {"mxordn": 0},
        {"mxords": 6},
        {"hmin": 0.2, "hmax": 0.1},
        {"Dfun": lambda x: x},
        {"tfirst": True},
    ],
)
def test_bad_ode_controls_rejected_before_execution(options):
    model, _, _ = configured()
    with pytest.raises((ValueError, TypeError)):
        model.configure(stripping={"solver": "odeint", "solver_options": options})


def test_unknown_and_unsupported_component_settings_fail_early():
    model, _, _ = configured()
    for group in (
        {"concentration": {"relation": lambda x: x}},
        {"accretion": {"mass_nodes": True}},
        {"stripping": {"solver": "unknown"}},
        {"disruption": {"ct_threshold": -1}},
        {"concentration": {"prescription": "unknown"}},
        {"stripping": {"ignored": 1}},
    ):
        with pytest.raises((ValueError, TypeError)):
            model.configure(**group)


def test_explicit_off_grid_redshift_support_is_bounded():
    model, request, _ = configured()
    catalog = next(
        iter(
            as_pair(
                model.population(
                    **{
                        **request,
                        "redshift": 0.1,
                        "reference_mass_redshift": 0.1,
                        "accretion_redshift_range": (0.2, 1.3),
                    }
                )
            ).values()
        )
    )
    assert np.max(catalog.columns["z_acc"]) <= 1.3
    np.testing.assert_allclose(np.unique(catalog.columns["z_acc"]), [0.7, 1.2], rtol=0, atol=2e-16)
    assert catalog.metadata["accretion_redshift_range_policy"] == "explicit-bounded"


def test_reference_epoch_is_not_an_accretion_mass_alias():
    model, request, _ = configured("nonzero")
    with pytest.raises(ValueError, match="must equal output redshift"):
        model.population(**{**request, "reference_mass_redshift": 0.0})
    with pytest.raises(TypeError):
        model.population(**request, accretion_mass_range_msun=(1e5, 1e7))
    result = model.population(**request)
    assert set(result) == {"cdm_reference", "sidm"}
    for state, catalog in result.items():
        assert catalog.metadata["state"] == state
        assert catalog.metadata["reference_mass_redshift"] == 0.5
        assert catalog.metadata["stage_unit_contract"] == "Msun-Mpc-km/s:v1"
        assert catalog.metadata["column_units"]["velocity"] == "km / s"
    assert not np.array_equal(result["sidm"].columns["m200_acc"][:4], np.geomspace(1e5, 1e7, 4))
    with pytest.raises(TypeError):
        result["sidm"] = None
    with pytest.raises(ValueError, match="state"):
        model.population(**request, state="unknown")
    selected = model.population(**request, state="sidm")
    for key in selected.weights:
        np.testing.assert_array_equal(selected.weights[key], result["sidm"].weights[key])


def test_interaction_changes_sidm_but_not_cdm_reference_structure():
    model, request, _ = configured()
    original = model.population(**request)
    changed = model.configure(
        interaction={"sigma0_cm2_g": 100.0, "velocity_scale_km_s": 30.0}
    ).population(**request)
    np.testing.assert_array_equal(
        original["cdm_reference"].columns["r_s_cdm"], changed["cdm_reference"].columns["r_s_cdm"]
    )
    np.testing.assert_array_equal(
        original["cdm_reference"].columns["m_bound"], changed["cdm_reference"].columns["m_bound"]
    )
    assert not np.array_equal(
        original["sidm"].columns["r_s_sidm"], changed["sidm"].columns["r_s_sidm"]
    )
    assert (
        original["sidm"].metadata["model_identifier"]
        != changed["sidm"].metadata["model_identifier"]
    )
    assert (
        original["cdm_reference"].metadata["model_identifier"]
        != original["sidm"].metadata["model_identifier"]
    )


def test_ratio_cap_supported_endpoints_and_legacy_finite_outputs():
    model, request, record = configured()
    for cap in (0.0, 0.9, 1.1):
        native = model.configure(interaction={"collapse_time_ratio_cap": cap}).population(**request)
        old = SubhaloProperties(tt_th=cap).subhalo_catalogs_calc(**record["parameters"])
        for state in native:
            for key, array in native[state].columns.items():
                assert np.all(np.isfinite(array))
                np.testing.assert_array_equal(array, old[state].columns[key])
            np.testing.assert_array_equal(
                native[state].weights["weight_survival"], old[state].weights["weight_survival"]
            )
        assert native["sidm"].metadata["collapse_time_ratio_cap_units"].startswith("dimensionless")
    for cap in (-0.1, 1.10001):
        with pytest.raises(ValueError):
            model.configure(interaction={"collapse_time_ratio_cap": cap})


def test_upper_reference_bound_still_uses_host_zero_mass():
    model, request, _ = configured("host_epoch")
    native = model.population(**{**request, "reference_mass_range_msun": (1e5, None)})
    meta = native["sidm"].metadata
    assert meta["reference_mass_range_msun"][1] == 0.1 * meta["host_mass_z0"]
    assert meta["reference_mass_upper_policy"] == "0.1*host-M200c-at-z0"
    assert meta["reference_mass_redshift"] == request["redshift"]


@pytest.mark.parametrize("state", ["paired", "cdm_reference", "sidm"])
def test_empty_formation_support_rejected_before_solver_setup(monkeypatch, state):
    model, request, _ = configured()

    def cannot_prepare_solver(*args, **kwargs):
        raise AssertionError("Empty formation support must fail before solver setup.")

    monkeypatch.setattr("sashimi_si.TidalStrippingSolver", cannot_prepare_solver)
    with pytest.raises(ValueError, match="No accretion redshift nodes.*formation"):
        model.population(
            **{**request, "accretion_redshift_range": (2.0, 3.0), "state": state}
        )


def test_formation_filter_distinguishes_candidate_and_executed_rows():
    model, request, _ = configured()
    result = model.population(**{**request, "accretion_redshift_range": (0.0, 3.0)})["sidm"]
    candidate = result.metadata["accretion_candidate_redshift_nodes"]
    executed = result.metadata["accretion_executed_redshift_nodes"]
    assert candidate == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert len(executed) < len(candidate)
    np.testing.assert_array_equal(executed, np.unique(result.columns["z_acc"]))
    assert result.metadata["accretion_executed_redshift_support"] == [executed[0], executed[-1]]


@pytest.mark.parametrize(
    "override",
    [
        {"host_mass_msun": 0.0},
        {"host_mass_msun": True},
        {"host_mass_definition": "vir"},
        {"redshift": float("nan")},
        {"reference_mass_range_msun": (10.0, 1.0)},
        {"accretion_redshift_range": (0.0, 0.1)},
    ],
)
def test_invalid_problem_inputs_are_rejected(override):
    model, request, _ = configured()
    with pytest.raises((ValueError, TypeError)):
        model.population(**{**request, **override})


def test_omitted_redshift_domain_preserves_legacy_support():
    model, request, record = configured()
    request.pop("accretion_redshift_range")
    native = as_pair(model.population(**request))
    parameters = {**record["parameters"], "zmax": 5.0}
    legacy = SubhaloProperties().subhalo_catalogs_calc(**parameters)
    for state, catalog in native.items():
        assert catalog.metadata["accretion_redshift_range_policy"] == "legacy-default-grid"
        for key, array in catalog.columns.items():
            np.testing.assert_array_equal(array, legacy[state].columns[key])
        for key, array in catalog.weights.items():
            np.testing.assert_array_equal(array, legacy[state].weights[key])
