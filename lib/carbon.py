"""Biogenic carbon for the Fraser fir system.

Why this module matters more than its size suggests
----------------------------------------------------
Every other quantity in the model scales with *inputs*: fertilizer, diesel,
sprays and cost all rise when you do more. Sequestration is the only term that
scales with *output* -- it rises when you harvest more trees. That asymmetry is
what makes cost and carbon genuinely conflict:

    cost/acre      is minimised by shrinking the block
    net GWP/acre   is minimised by growing it

Without this term, every gene improves both objectives at once and the Pareto
front collapses to a single corner (see
tests/test_optimize.py::test_cost_and_gwp_share_a_minimizing_corner).

Basis matters, and it is easy to get wrong
-------------------------------------------
Sequestration is a fixed 18.3 kg CO2 per 7-ft tree *by definition of the
functional unit* -- the manuscript holds biomass constant across scenarios and
varies only time and survival. So:

  * per TREE  : every plan carries the same -18.3 constant. It cancels out of
                the ranking entirely and changes nothing.
  * per ACRE  : it multiplies tree count and dominates the objective.

Optimise per acre. Report per tree. ``net_gwp`` enforces this with an explicit
basis argument rather than leaving it to a caller's assumption.

System boundary
---------------
These figures are FARM GATE. Cradle-to-gate net GWP is -15.31 kg CO2-eq/tree;
cradle-to-grave is +6.18 once consumer pickup and end-of-life are included. The
sign flips. A grower controls cultivation and not consumer travel, so the
optimizer works at the farm gate and the cradle-to-grave figure is shown
alongside as context, never optimised against.

Stock, not removal
------------------
18.3 kg CO2/tree is a carbon STOCK at harvest. Under the baseline mulching
scenario only ~14% of above-ground and ~3.2% of root carbon remains at 100
years. Never present the farm-gate number as permanent removal.

Source: Life Cycle Assessment of North Carolina Fraser Fir Christmas Trees
(Morizet-Davis et al.), Sections 2.2.1, 3.2 and Tables S11-S16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Basis = Literal["acre", "tree"]


@dataclass(frozen=True)
class CarbonParams:
    """Biogenic carbon parameters for one 7-ft Fraser fir.

    All values are editable in the UI; defaults are the manuscript baseline.
    """

    # Carbon stock at harvest, kg CO2 per marketable 7-ft tree
    above_ground_co2: float = 14.1
    below_ground_co2: float = 4.2

    # Share of shared cultivation burden assigned to the tree; the remainder
    # goes to the recovered-greenery wreath by-product on a wet-mass basis.
    tree_allocation: float = 0.893

    # 100-year retention fractions under the baseline mulching scenario
    mulch_retention_above_ground: float = 0.14
    root_retention_below_ground: float = 0.032

    def __post_init__(self) -> None:
        if not 0.0 < self.tree_allocation <= 1.0:
            raise ValueError("tree_allocation must be in (0, 1]")
        for name in (
            "above_ground_co2",
            "below_ground_co2",
            "mulch_retention_above_ground",
            "root_retention_below_ground",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def co2_per_tree(self) -> float:
        """Total biogenic CO2 stock at harvest, kg per tree."""
        return self.above_ground_co2 + self.below_ground_co2

    @property
    def wreath_allocation(self) -> float:
        return 1.0 - self.tree_allocation


def sequestration(trees: float, params: CarbonParams | None = None) -> float:
    """Total biogenic CO2 stock at harvest for a stand, kg CO2.

    Positive number. Subtract it from cultivation emissions to get net GWP.
    """
    p = params or CarbonParams()
    return float(trees) * p.co2_per_tree


def allocated_cultivation_gwp(
    cultivation_gwp: float, params: CarbonParams | None = None
) -> float:
    """Share of cultivation GWP borne by the trees rather than the wreath.

    The app's inventory covers the whole plantation, so cultivation burdens must
    be allocated before they are compared against tree-only carbon uptake. No
    biogenic CO2 is allocated to the wreath.
    """
    p = params or CarbonParams()
    return float(cultivation_gwp) * p.tree_allocation


def net_gwp(
    cultivation_gwp: float,
    trees: float,
    basis: Basis = "acre",
    params: CarbonParams | None = None,
    allocate: bool = True,
) -> float:
    """Farm-gate net GWP: allocated cultivation emissions minus carbon uptake.

    ``basis="acre"`` returns kg CO2-eq for the whole stand -- use this as the
    optimizer objective, because sequestration only varies with tree count.

    ``basis="tree"`` divides by tree count for reporting. Note that on this
    basis the sequestration term is a constant -p.co2_per_tree for every plan,
    so it cannot influence a ranking. That is a property of the functional
    unit, not a bug.
    """
    if basis not in ("acre", "tree"):
        raise ValueError(f"basis must be 'acre' or 'tree', got {basis!r}")
    p = params or CarbonParams()
    burden = allocated_cultivation_gwp(cultivation_gwp, p) if allocate else float(cultivation_gwp)
    total = burden - sequestration(trees, p)
    if basis == "tree":
        return total / max(float(trees), 1e-9)
    return total


def retained_at_100_years(trees: float, params: CarbonParams | None = None) -> float:
    """Biogenic CO2 still stored after 100 years under baseline mulching, kg.

    Use this whenever the farm-gate stock is presented, so a carbon stock is
    never mistaken for a durable removal.
    """
    p = params or CarbonParams()
    per_tree = (
        p.above_ground_co2 * p.mulch_retention_above_ground
        + p.below_ground_co2 * p.root_retention_below_ground
    )
    return float(trees) * per_tree
