"""Unit tests for continuous optimizer mechanics."""

import numpy as np
import pytest

from lib.defaults import GWP_COL
from lib.optimize import (
    _make_individual_factory, _repair_factory, apply_individual,
    evaluate_budget_constrained, evaluate_compliance_constrained,
    evaluate_cost_gwp_constrained, evaluate_cost_gwp_impact_constrained,
    evaluate_cost_only_constrained, evaluate_single_impact_constrained,
    run_nsga2_constrained, run_single_constrained, split_genome,
)


@pytest.fixture
def simple():
    return dict(
        base=np.array([10.0, 20.0, 30.0]),
        costs=np.array([2.0, 3.0, 4.0]),
        matrix=np.array([[1.0, 3.0], [2.0, 2.0], [4.0, 1.0]]),
        cols=[GWP_COL, "other"],
        scale=np.array([True, False, False]),
        eff=np.array([False, True, False]),
    )


def test_apply_individual_scale_and_efficiency(simple):
    out, scale = apply_individual([1.1, 0.8], simple["base"], simple["scale"], simple["eff"])
    assert scale == pytest.approx(1.1)
    assert out == pytest.approx([11, 17.6, 30])


def test_apply_individual_missing_efficiency_defaults_to_one(simple):
    out, _ = apply_individual([0.9], simple["base"], simple["scale"], simple["eff"])
    assert out == pytest.approx([9, 18, 30])


def test_split_genome():
    assert split_genome([1.1, 0.9, 1.0, 0.75], 2, 1) == (1.1, [0.9, 1.0], [0.75])


def test_repair_is_hard():
    ind = [9.0, -4.0, 2.0]
    _repair_factory(0.2, 0.1, 1, 1)(ind)
    assert ind == pytest.approx([1.2, 0.9, 0.999999])


def test_individual_factory_respects_gene_classes():
    make = _make_individual_factory(2, 0.2, 0.1, 1)
    for _ in range(20):
        ind = make()
        assert 0.8 <= ind[0] <= 1.2
        assert all(0.9 <= x <= 1.1 for x in ind[1:3])
        assert 0 <= ind[3] < 1


def _args(s):
    return (
        s["costs"], s["matrix"], s["cols"], s["base"], 100,
        s["scale"], s["eff"], 0.2, 0.2,
    )


def test_cost_gwp_returns_two_objectives(simple):
    assert evaluate_cost_gwp_constrained([1.0, 1.0], *_args(simple)) == pytest.approx((200, 170))


def test_output_floor_adds_penalty(simple):
    a = evaluate_cost_gwp_constrained([0.8, 1.0], *_args(simple))
    b = evaluate_cost_gwp_constrained([0.8, 1.0], *_args(simple), min_trees=100)
    assert b[0] > a[0]


def test_three_objective_evaluator(simple):
    result = evaluate_cost_gwp_impact_constrained(
        [1.0, 1.0], *_args(simple), third_impact_col="other"
    )
    assert result == pytest.approx((200, 170, 100))


def test_budget_penalizes_overspend(simple):
    ok = evaluate_budget_constrained([1.0, 1.0], *_args(simple), budget_limit=250)
    bad = evaluate_budget_constrained([1.0, 1.0], *_args(simple), budget_limit=100)
    assert bad[0] > ok[0]


def test_compliance_penalizes_missed_target(simple):
    ok = evaluate_compliance_constrained([1.0, 1.0], *_args(simple), gwp_target=200)
    bad = evaluate_compliance_constrained([1.0, 1.0], *_args(simple), gwp_target=100)
    assert bad[0] > ok[0]


def test_cost_only(simple):
    result = evaluate_cost_only_constrained(
        [1.0, 1.0], simple["costs"], simple["base"], 100,
        simple["scale"], simple["eff"], 0.2, 0.2
    )
    assert result == pytest.approx((200,))


def test_single_impact_present(simple):
    result = evaluate_single_impact_constrained(
        [1.0, 1.0], simple["matrix"], "other", simple["cols"], simple["base"], 100,
        simple["scale"], simple["eff"], 0.2, 0.2
    )
    assert result == pytest.approx((100,))


def test_single_impact_missing_is_zero(simple):
    result = evaluate_single_impact_constrained(
        [1.0, 1.0], simple["matrix"], "missing", simple["cols"], simple["base"], 100,
        simple["scale"], simple["eff"], 0.2, 0.2
    )
    assert result == pytest.approx((0,))


def test_nsga_front_stays_in_bounds(simple):
    front = run_nsga2_constrained(
        30, 5, 0.7, 0.3, simple["costs"], simple["matrix"], simple["cols"],
        simple["base"], 100, simple["scale"], simple["eff"], 0.2, 0.2, seed=4
    )
    assert front
    assert all(0.8 <= ind[0] <= 1.2 and 0.8 <= ind[1] <= 1.2 for ind in front)


def test_single_runner_stays_in_bounds(simple):
    best = run_single_constrained(
        evaluate_cost_only_constrained, 24, 4, 0.7, 0.3,
        simple["base"], 100, simple["scale"], simple["eff"], 0.2, 0.2,
        simple["costs"], seed=3,
    )
    assert 0.8 <= best[0] <= 1.2
