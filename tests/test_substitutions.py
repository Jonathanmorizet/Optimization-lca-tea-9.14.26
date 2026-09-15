"""Tests for discrete management-practice substitutions."""

import numpy as np
import pytest

from lib.substitutions import (
    PracticeOption, Substitution, apply_substitutions, decode_choices,
    default_substitutions, unsourced_options,
)


@pytest.fixture
def inventory():
    return (
        np.array([100.0, 20.0, 10.0, 4.0]),
        np.array([1.75, 2.0, 3.0, 4.0]),
        ["Transplants (initial)", "Roundup", "Diesel tractor", "Other"],
        np.array([False, True, True, False]),
    )


def test_default_substitutions_are_nonempty():
    """One live decision: weed control. Rootstock and genetics were removed --
    see test_no_hypothetical_practices_offered for why."""
    assert len(default_substitutions()) >= 1


def test_decode_first_bucket_selects_baseline():
    subs = default_substitutions()
    assert all(o is s.options[0] for o, s in zip(decode_choices([0, 0, 0], subs), subs))


def test_decode_last_bucket_selects_last_option():
    subs = default_substitutions()
    assert all(o is s.options[-1] for o, s in zip(decode_choices([0.999] * len(subs), subs), subs))


def test_decode_clamps_out_of_range_genes():
    sub = Substitution("x", "X", "", (PracticeOption("a"), PracticeOption("b")))
    assert decode_choices([-10], [sub])[0].name == "a"
    assert decode_choices([10], [sub])[0].name == "b"


def test_baseline_options_are_identity(inventory):
    amounts, costs, materials, mask = inventory
    subs = default_substitutions()
    choices = [s.options[0] for s in subs]
    new_a, new_c, trees, _ = apply_substitutions(amounts, costs, materials, 1900, mask, choices, subs)
    assert new_a == pytest.approx(amounts)
    assert new_c == pytest.approx(costs)
    assert trees == pytest.approx(1900)


def test_mow_program_changes_herbicide_and_diesel(inventory):
    amounts, costs, materials, mask = inventory
    sub = default_substitutions()[0]
    new_a, _, _, names = apply_substitutions(
        amounts, costs, materials, 1900, mask, [sub.options[1]], [sub]
    )
    assert new_a[1] == pytest.approx(7.0)
    assert new_a[2] == pytest.approx(18.5)
    assert names == ["Mow-dominant program"]


def test_scenario_applies_before_choices(inventory):
    amounts, costs, materials, mask = inventory
    sub = Substitution("x", "X", "", (PracticeOption("base"),))
    scenario = PracticeOption("scenario", amount_multipliers=(("other", 0.5),))
    new_a, _, _, _ = apply_substitutions(
        amounts, costs, materials, 1900, mask, [sub.options[0]], [sub], scenario=scenario
    )
    assert new_a[3] == pytest.approx(2.0)


def test_inputs_are_not_mutated(inventory):
    amounts, costs, materials, mask = inventory
    before_a, before_c = amounts.copy(), costs.copy()
    sub = default_substitutions()[0]
    apply_substitutions(amounts, costs, materials, 1900, mask, [sub.options[1]], [sub])
    assert np.array_equal(amounts, before_a)
    assert np.array_equal(costs, before_c)


def test_unsourced_options_are_visible():
    flagged = unsourced_options(default_substitutions())
    assert flagged
    assert any("Mow" in item for item in flagged), (
        "the mow-dominant ratios are still placeholders and must stay visible"
    )


def test_choice_count_must_match_substitutions(inventory):
    amounts, costs, materials, mask = inventory
    sub = default_substitutions()[0]
    with pytest.raises(ValueError):
        apply_substitutions(amounts, costs, materials, 1900, mask, [], [sub])


def test_no_hypothetical_practices_offered():
    """Grafted rootstock and improved genetics must not be selectable.

    Genetics is hypothetical -- no commercial seed, so no price and no action a
    grower can take. Grafted rootstock is real but site-conditional: the
    survival gain only exists where Phytophthora pressure exists, and the model
    cannot observe that, so offering it means recommending grafting on clean
    ground. Both belong in the manuscript's sensitivity analysis instead.
    """
    from lib.substitutions import default_substitutions

    text = " ".join(
        s.key + s.label + s.help + " " + " ".join(o.name for o in s.options)
        for s in default_substitutions()
    ).lower()
    for banned in ("graft", "rootstock", "genetic", "improved genetics"):
        assert banned not in text, f"{banned!r} is back in the choice set"


def test_direct_matrix_is_tracked_and_never_hits_combustion_rows():
    """Results must be able to split upstream from direct, and diesel must
    carry no direct layer -- its activity already includes combustion."""
    import numpy as np

    from lib.data_model import (
        build_optimizer_arrays,
        default_farm_config,
        inventory_table_for_editor,
    )

    arrays = build_optimizer_arrays(default_farm_config(), inventory_table_for_editor())
    direct = np.asarray(arrays["direct_matrix"], dtype=float)
    assert direct.shape == np.asarray(arrays["impact_matrix"]).shape
    assert direct.any(), "expected some direct field emissions"
    for i, (name, stage) in enumerate(zip(arrays["materials"], arrays["stages"])):
        if stage != "production":
            assert not direct[i].any(), f"{name} ({stage}) must get no direct layer"


def test_carbon_params_are_not_user_editable_in_the_ui():
    """Sequestration is measured for the functional unit, not a preference."""
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "pages" / "2_Optimize.py").read_text()
    assert "CarbonParams()" in page
    for widget in ("Above-ground CO2 per tree", "Below-ground CO2 per tree"):
        assert f'number_input("{widget}' not in page
