"""Discrete grower-practice substitutions used by the optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PracticeOption:
    name: str
    amount_multipliers: tuple[tuple[str, float], ...] = ()
    cost_multipliers: tuple[tuple[str, float], ...] = ()
    cost_additions: tuple[tuple[str, float], ...] = ()
    tree_multiplier: float = 1.0
    placeholder: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.tree_multiplier <= 0:
            raise ValueError("tree_multiplier must be positive")
        for _, value in self.amount_multipliers + self.cost_multipliers:
            if value < 0:
                raise ValueError("multipliers must be non-negative")


@dataclass(frozen=True)
class Substitution:
    key: str
    label: str
    help: str
    options: tuple[PracticeOption, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("a substitution needs at least one option")


def default_substitutions() -> list[Substitution]:
    """Management choices a grower can actually make this season.

    Deliberately excluded
    ---------------------
    Grafted PRR-resistant rootstock and genetically improved planting stock.
    Both were removed rather than demoted, for different reasons:

      * Improved genetics is hypothetical. Commercial-scale seed is only
        "anticipated in the coming years", so no market price exists and a
        grower cannot act on the recommendation.

      * Grafted rootstock is real but site-conditional. The 88.5% -> 97.5%
        survival gain represents elimination of Phytophthora-attributable
        mortality, so on clean ground it costs money and buys nothing. The
        model cannot observe site pressure, so offering it as an option means
        recommending grafting where it does nothing.

    Both belong in the manuscript's sensitivity analysis, not in a tool that
    tells someone what to do this season. test_no_hypothetical_practices below
    fails if either is re-added.
    """
    return [
        Substitution(
            key="weed_control",
            label="Weed-control strategy",
            help="Compare the recorded spray program with a mow-dominant program.",
            options=(
                PracticeOption("Recorded herbicide program"),
                PracticeOption(
                    "Mow-dominant program",
                    amount_multipliers=(
                        ("roundup", 0.35),
                        ("crossbow", 0.35),
                        ("diesel", 1.85),
                        ("tractor use", 1.30),
                    ),
                    placeholder=True,
                    note="Replace ratios with field trial or grower records.",
                ),
            ),
        ),
    ]


def decode_choices(
    genes: Sequence[float], substitutions: Sequence[Substitution]
) -> list[PracticeOption]:
    """Decode one gene per substitution into an option."""
    decoded: list[PracticeOption] = []
    for i, sub in enumerate(substitutions):
        gene = float(genes[i]) if i < len(genes) else 0.0
        gene = min(max(gene, 0.0), 0.999999)
        index = min(int(gene * len(sub.options)), len(sub.options) - 1)
        decoded.append(sub.options[index])
    return decoded


def _apply_option(
    amounts: np.ndarray,
    costs: np.ndarray,
    materials: Sequence[str],
    trees: float,
    option: PracticeOption,
) -> tuple[np.ndarray, np.ndarray, float]:
    names = [str(x).lower() for x in materials]
    for keyword, multiplier in option.amount_multipliers:
        for i, name in enumerate(names):
            if keyword.lower() in name:
                amounts[i] *= float(multiplier)
    for keyword, multiplier in option.cost_multipliers:
        for i, name in enumerate(names):
            if keyword.lower() in name:
                costs[i] *= float(multiplier)
    for keyword, addition in option.cost_additions:
        for i, name in enumerate(names):
            if keyword.lower() in name:
                costs[i] += float(addition)
    return amounts, costs, float(trees) * option.tree_multiplier


def apply_substitutions(
    amounts,
    costs,
    materials: Sequence[str],
    trees: float,
    efficiency_mask,
    choices: Sequence[PracticeOption],
    substitutions: Sequence[Substitution],
    scenario: PracticeOption | Iterable[PracticeOption] | None = None,
):
    """Apply selected practices without mutating caller-owned arrays."""
    new_amounts = np.asarray(amounts, dtype=float).copy()
    new_costs = np.asarray(costs, dtype=float).copy()
    if len(new_amounts) != len(materials) or len(new_costs) != len(materials):
        raise ValueError("amounts, costs, and materials must have equal lengths")
    if len(choices) != len(substitutions):
        raise ValueError("one decoded choice is required for each substitution")

    selected: list[PracticeOption] = []
    if scenario is not None:
        if isinstance(scenario, PracticeOption):
            selected.append(scenario)
        else:
            selected.extend(list(scenario))
    selected.extend(choices)

    names: list[str] = []
    new_trees = float(trees)
    for option in selected:
        new_amounts, new_costs, new_trees = _apply_option(
            new_amounts, new_costs, materials, new_trees, option
        )
        if option in choices:
            names.append(option.name)
    return new_amounts, new_costs, new_trees, names


def unsourced_options(substitutions: Sequence[Substitution]) -> list[str]:
    """List enabled options that still use placeholder economics."""
    return [
        f"{sub.label}: {option.name}"
        for sub in substitutions
        for option in sub.options
        if option.placeholder
    ]
