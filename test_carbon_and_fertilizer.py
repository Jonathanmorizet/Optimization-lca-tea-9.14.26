"""Tests for biogenic carbon and direct field emissions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from lib.carbon import (
    CarbonParams,
    allocated_cultivation_gwp,
    net_gwp,
    retained_at_100_years,
    sequestration,
)
from lib.data_model import (
    build_optimizer_arrays,
    default_farm_config,
    inventory_table_for_editor,
)
from lib.defaults import GWP_COL
from lib.fertilizer import (
    FRAC_GASF_BY_FORM,
    FertiliserProduct,
    characterised_direct,
    default_products,
    direct_emissions,
    emissions_for_plan,
)
from lib.optimize import evaluate_with_substitutions, run_nsga2_constrained

ALLOC = 0.893
TREES = 1900
N_BASE = 362.88


# --- carbon --------------------------------------------------------------


def test_manuscript_carbon_stock():
    p = CarbonParams()
    assert p.co2_per_tree == pytest.approx(18.3)
    assert p.above_ground_co2 == pytest.approx(14.1)
    assert p.below_ground_co2 == pytest.approx(4.2)
    assert sequestration(1900, p) == pytest.approx(34770.0)


def test_allocation_splits_burden_not_uptake():
    """Cultivation burden is shared with the wreath; biogenic uptake is not."""
    p = CarbonParams()
    assert allocated_cultivation_gwp(1000.0, p) == pytest.approx(893.0)
    assert p.wreath_allocation == pytest.approx(0.107)
    # uptake is unallocated -- scaling trees scales it linearly
    assert sequestration(2 * 1900, p) == pytest.approx(2 * sequestration(1900, p))


def test_net_gwp_is_negative_at_farm_gate():
    """The stand is a large net sink before downstream stages."""
    p = CarbonParams()
    assert net_gwp(5765.0, 1900, "acre", p) < -20000


def test_per_tree_basis_makes_sequestration_a_constant():
    """The key subtlety: on a per-tree basis the credit cannot rank plans.

    Two plans with different tree counts but the same cost per tree must get
    the same net GWP per tree. That is a property of the functional unit, and
    it is why the optimizer works per acre.
    """
    p = CarbonParams()
    a = net_gwp(5765.0, 1900, "tree", p)
    b = net_gwp(5765.0 * 2, 3800, "tree", p)
    assert a == pytest.approx(b)


def test_acre_basis_does_rank_plans():
    p = CarbonParams()
    small = net_gwp(4623.0, 1520, "acre", p)
    large = net_gwp(6907.0, 2280, "acre", p)
    assert large < small, "more trees must mean lower net GWP per acre"


def test_retention_is_far_below_the_harvest_stock():
    """Guards against presenting a stock as a durable removal."""
    p = CarbonParams()
    assert retained_at_100_years(1900, p) < 0.2 * sequestration(1900, p)


def test_bad_basis_and_allocation_rejected():
    with pytest.raises(ValueError):
        net_gwp(1.0, 1.0, basis="hectare")
    with pytest.raises(ValueError):
        CarbonParams(tree_allocation=1.5)


# --- the front -----------------------------------------------------------


@pytest.fixture(scope="module")
def arrays():
    return build_optimizer_arrays(default_farm_config(), inventory_table_for_editor())


def _front(arrays, carbon):
    return run_nsga2_constrained(
        popsize=120, ngen=50, cxpb=0.7, mutpb=0.3,
        costs=arrays["costs"], matrix=arrays["impact_matrix"],
        impact_cols=arrays["traci_impact_cols"],
        base_amounts=arrays["base_amounts"], baseline_trees=arrays["baseline_trees"],
        scale_mask=arrays["scale_mask"], efficiency_mask=arrays["efficiency_mask"],
        max_scale_dev=0.20, max_eff_dev=0.20, seed=5, n_obj=2, n_choice=0,
        eval_func=evaluate_with_substitutions, materials=arrays["materials"],
        substitutions=[], objective_cols=[GWP_COL], per_tree=False,
        carbon_params=carbon,
    )


def test_sequestration_alone_creates_a_front(arrays):
    """The core fix: no substitutions needed once uptake is in the objective."""
    front = _front(arrays, CarbonParams())
    pts = np.array([ind.fitness.values for ind in front])
    assert len(front) > 20
    spread = (pts[:, 0].max() - pts[:, 0].min()) / pts[:, 0].min()
    assert spread > 0.10, "expected a wide cost spread across the front"
    assert int(pts[:, 0].argmin()) != int(pts[:, 1].argmin())


def test_cheapest_and_greenest_sit_at_opposite_scale_bounds(arrays):
    front = _front(arrays, CarbonParams())
    pts = np.array([ind.fitness.values for ind in front])
    assert front[int(pts[:, 0].argmin())][0] < front[int(pts[:, 1].argmin())][0]


def test_bounds_are_hard_not_penalised(arrays):
    """Regression: cxBlend extrapolates, and a penalty cannot hold a bound
    against a large negative term. Scale once reached 63x baseline."""
    front = _front(arrays, CarbonParams())
    for ind in front:
        assert 0.8 - 1e-9 <= ind[0] <= 1.2 + 1e-9


# --- direct field emissions ----------------------------------------------


def test_reproduces_baseline_spec_values():
    e = direct_emissions(synthetic_n=N_BASE * ALLOC, frac_gasf=0.11).scaled(1 / TREES)
    assert e.f_sn == pytest.approx(0.1705536, abs=1e-7)
    assert e.nh3 == pytest.approx(0.0227811, abs=1e-7)
    assert e.nitrate == pytest.approx(0.1812741, abs=1e-7)
    assert e.n2o_direct == pytest.approx(0.0042882, abs=1e-7)
    assert e.n2o_volatilisation == pytest.approx(0.0004127, abs=1e-7)
    assert e.n2o_leaching == pytest.approx(0.0007076, abs=1e-7)
    assert e.gwp == pytest.approx(1.6117325, abs=1e-6)


def test_mixed_application_uses_the_synthetic_ef1():
    """IPCC Table 11.1 footnote 5. The 0.006 organic factor comes from trials
    applying organic amendments ALONE and must not be split out of a blend."""
    mixed = direct_emissions(synthetic_n=0.5 * N_BASE * ALLOC, organic_n=N_BASE * ALLOC)
    assert mixed.ef1_used == pytest.approx(0.016)
    assert mixed.scaled(1 / TREES).gwp == pytest.approx(2.5294137, abs=1e-5)


def test_exclusively_organic_uses_the_organic_ef1():
    only_org = direct_emissions(synthetic_n=0.0, organic_n=100.0)
    assert only_org.ef1_used == pytest.approx(0.006)


def test_dry_climate_uses_a_single_ef1():
    dry = direct_emissions(synthetic_n=100.0, organic_n=50.0, climate="dry")
    assert dry.ef1_used == pytest.approx(0.005)
    assert dry.n2o_leaching == pytest.approx(0.0), "no leaching default in dry climates"


def test_frac_gasf_spans_fifteenfold_across_products():
    """Product choice is a real substitution built from published defaults."""
    assert FRAC_GASF_BY_FORM["urea"] / FRAC_GASF_BY_FORM["nitrate"] == pytest.approx(15.0)
    acid = {}
    for p in default_products():
        if p.is_organic:
            continue
        kg = p.product_for_available_n(N_BASE * ALLOC)
        em = emissions_for_plan([(p, kg)]).scaled(1 / TREES)
        acid[p.name] = characterised_direct(em)["kg SO2-Eq/Unit"]
    assert max(acid.values()) / min(acid.values()) > 10


def test_npk_compound_defaults_to_an_based_not_aggregate():
    """Table 7A.2 treats NPK compounds as a 50/50 AN/CAN mix."""
    npk = next(p for p in default_products() if p.name.startswith("NPK"))
    assert npk.frac_gasf == pytest.approx(0.05)


def test_litter_pan_doubling():
    """50% availability means twice the total N for the same available N."""
    litter = next(p for p in default_products() if p.is_organic)
    kg = litter.product_for_available_n(90.72)
    assert litter.total_n(kg) == pytest.approx(2 * 90.72)
    assert kg == pytest.approx(6278.0, rel=1e-3)


def test_liming_co2_is_direct_and_additive():
    """IPCC Ch.11 s.11.3. Absent from the current inventory entirely."""
    without = direct_emissions(synthetic_n=100.0)
    with_lime = direct_emissions(synthetic_n=100.0, lime_kg=1000.0)
    assert with_lime.lime_co2 == pytest.approx(1000.0 * 0.12 * 44 / 12)
    assert with_lime.gwp > without.gwp


def test_no_direct_contribution_to_untouched_categories():
    ch = characterised_direct(direct_emissions(synthetic_n=100.0))
    for absent in ("CTUh/Unit", "kg CFC-11-Eq/Unit", "kg O3-Eq/Unit"):
        assert absent not in ch


def test_invalid_products_rejected():
    with pytest.raises(ValueError):
        FertiliserProduct(name="bad", n_fraction=1.4)
    with pytest.raises(ValueError):
        FertiliserProduct(name="bad", n_fraction=0.2, form="not a form")


# --- stage and basis flags ----------------------------------------------


def test_stage_flags_separate_production_from_combustion(arrays):
    """The upstream/direct boundary must be explicit, because the failure mode
    is silent: nobody notices a double count, the number is just too high."""
    by_name = dict(zip(arrays["materials"], arrays["stages"]))
    fert = [v for k, v in by_name.items() if "fertilizer" in k.lower()]
    diesel = [v for k, v in by_name.items() if "diesel" in k.lower()]
    assert fert and all(v == "production" for v in fert)
    assert diesel and all(v == "combustion" for v in diesel)


def test_no_material_is_both_stages(arrays):
    assert set(arrays["stages"]) <= {"production", "combustion", "none"}


def test_pesticides_declare_an_active_ingredient_basis(arrays):
    by_name = dict(zip(arrays["materials"], arrays["bases"]))
    for name, basis in by_name.items():
        if any(k in name.lower() for k in ("roundup", "crossbow", "tristar")):
            assert basis == "active_ingredient", f"{name} needs an explicit basis"


def test_product_to_active_ingredient_reproduces_the_lci_figure():
    """1,386 fl oz/acre/rotation at ~47.4% gives the LCI's 22.725 kg."""
    from lib.fertilizer import product_to_active_ingredient

    assert product_to_active_ingredient(1386, 0.474) == pytest.approx(22.7, abs=0.1)


def test_salt_to_acid_conversion_reduces_mass():
    """Labels state the salt; ecoinvent pesticide datasets are per kg acid."""
    from lib.fertilizer import to_acid_equivalent

    assert to_acid_equivalent(22.725, "glyphosate_potassium") == pytest.approx(18.63, abs=0.01)
    assert to_acid_equivalent(22.725, "glyphosate_isopropylamine") < to_acid_equivalent(
        22.725, "glyphosate_potassium"
    )
    with pytest.raises(ValueError):
        to_acid_equivalent(1.0, "not a salt")
