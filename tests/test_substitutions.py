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
    assert len(default_substitutions()) >= 3


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


def test_grafted_rootstock_adds_cost_and_survival(inventory):
    amounts, costs, materials, mask = inventory
    sub = default_substitutions()[1]
    _, new_c, trees, _ = apply_substitutions(
        amounts, costs, materials, 1900, mask, [sub.options[1]], [sub]
    )
    assert new_c[0] == pytest.approx(4.25)
    assert trees > 1900


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
    assert any("Grafted" in item for item in flagged)


def test_choice_count_must_match_substitutions(inventory):
    amounts, costs, materials, mask = inventory
    sub = default_substitutions()[0]
    with pytest.raises(ValueError):
        apply_substitutions(amounts, costs, materials, 1900, mask, [], [sub])
