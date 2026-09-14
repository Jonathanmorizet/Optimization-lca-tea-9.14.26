"""DEAP / NSGA-II evaluation and runners (no Streamlit, no runtime pip).

Individual encoding: [production_scale, efficiency_1, efficiency_2, ...]
Scale materials vary with production_scale only.
Efficiency materials vary with production_scale * efficiency_i (running index).
Fixed materials stay at the baseline amount.

The original cost-vs-GWP / cost-only / single-impact evaluators used
``np.where(efficiency_mask[:i])[0].size`` which is correct, but budget and
compliance already used a running counter. All paths now share apply_individual.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import numpy as np
from deap import base, creator, tools

from .defaults import GWP_COL

_CREATOR_READY = {"nsga": False, "single": False}


def set_seed(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def apply_individual(
    ind: Sequence[float],
    base_amounts: np.ndarray,
    scale_mask: np.ndarray,
    efficiency_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Decode an individual into material amounts and production_scale."""
    production_scale = float(ind[0])
    efficiency_factors = np.asarray(ind[1:], dtype=float)
    base = np.asarray(base_amounts, dtype=float).flatten()
    scale_mask = np.asarray(scale_mask, dtype=bool).flatten()
    efficiency_mask = np.asarray(efficiency_mask, dtype=bool).flatten()
    final = np.copy(base)
    eff_i = 0
    n_eff = len(efficiency_factors)
    for i in range(len(final)):
        if scale_mask[i]:
            final[i] = base[i] * production_scale
        elif efficiency_mask[i]:
            factor = float(efficiency_factors[eff_i]) if eff_i < n_eff else 1.0
            final[i] = base[i] * production_scale * factor
            eff_i += 1
    return final, production_scale


def _bound_penalty(value: float, lo: float, hi: float, weight: float) -> float:
    if value < lo:
        return weight * abs(value - lo)
    if value > hi:
        return weight * abs(value - hi)
    return 0.0


def _scale_eff_penalties(
    production_scale: float,
    efficiency_factors: np.ndarray,
    max_scale_dev: float,
    max_eff_dev: float,
    weight: float = 10_000.0,
) -> float:
    penalty = _bound_penalty(
        production_scale, 1 - max_scale_dev, 1 + max_scale_dev, weight
    )
    lo, hi = 1 - max_eff_dev, 1 + max_eff_dev
    for ef in np.asarray(efficiency_factors, dtype=float).flatten():
        penalty += _bound_penalty(float(ef), lo, hi, weight)
    return float(penalty)


def _impact(
    amounts: np.ndarray,
    impact_matrix: np.ndarray,
    impact_cols: Sequence[str],
    colname: str,
) -> float:
    """Total burden for one TRACI column. Returns 0.0 if the column is absent."""
    try:
        idx = list(impact_cols).index(colname)
    except ValueError:
        return 0.0
    col = np.asarray(impact_matrix, dtype=float)[:, idx].flatten()
    a = np.asarray(amounts, dtype=float).flatten()
    n = min(len(a), len(col))
    return float(np.dot(a[:n], col[:n]))


def _gwp(amounts: np.ndarray, impact_matrix: np.ndarray, impact_cols: Sequence[str]) -> float:
    return _impact(amounts, impact_matrix, impact_cols, GWP_COL)


def _output_floor_penalty(
    production_scale: float,
    baseline_trees: float,
    min_trees: Optional[float],
    weight: float = 1_000_000.0,
) -> float:
    """Penalize plans that meet the objectives by simply growing fewer trees.

    Without this, cost and every TRACI column are monotonically increasing in
    every gene, so the cheapest and cleanest plan is always the smallest one and
    the Pareto front collapses to a single corner. The floor is what forces the
    search to choose *which* inputs to cut rather than how much farm to abandon.
    """
    if min_trees is None:
        return 0.0
    actual = float(baseline_trees) * float(production_scale)
    if actual < float(min_trees):
        return weight * float(min_trees - actual)
    return 0.0


def _cost(amounts: np.ndarray, costs: np.ndarray) -> float:
    a = np.asarray(amounts, dtype=float).flatten()
    c = np.asarray(costs, dtype=float).flatten()
    n = min(len(a), len(c))
    return float(np.dot(a[:n], c[:n]))


def evaluate_cost_gwp_constrained(
    ind,
    costs,
    impact_matrix,
    impact_cols,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
    min_trees=None,
):
    """Minimize (cost, GWP) subject to an output floor.

    ``min_trees`` is the number of harvested trees the plan may not fall below.
    Pass None only for tests that deliberately exercise the unconstrained case.
    """
    final, scale = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    penalty = _scale_eff_penalties(
        scale, ind[1:], max_scale_deviation, max_efficiency_deviation
    )
    penalty += _output_floor_penalty(scale, baseline_trees, min_trees)
    return _cost(final, costs) + penalty, _gwp(final, impact_matrix, impact_cols) + penalty


def evaluate_cost_gwp_impact_constrained(
    ind,
    costs,
    impact_matrix,
    impact_cols,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
    third_impact_col,
    min_trees=None,
):
    """Minimize (cost, GWP, one further TRACI category) subject to an output floor.

    The third axis exposes burden shifting: materials differ by orders of
    magnitude in how much carbon versus nutrient or toxicity burden a dollar
    buys, so a plan that is good on GWP alone can be worse on eutrophication or
    ozone depletion. Two objectives cannot show that; three can.
    """
    final, scale = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    penalty = _scale_eff_penalties(
        scale, ind[1:], max_scale_deviation, max_efficiency_deviation
    )
    penalty += _output_floor_penalty(scale, baseline_trees, min_trees)
    return (
        _cost(final, costs) + penalty,
        _gwp(final, impact_matrix, impact_cols) + penalty,
        _impact(final, impact_matrix, impact_cols, third_impact_col) + penalty,
    )


def evaluate_budget_constrained(
    ind,
    costs,
    impact_matrix,
    impact_cols,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
    budget_limit,
):
    """Maximize trees, minimize GWP, subject to budget. Returns (-trees, gwp)."""
    final, scale = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    total_cost = _cost(final, costs)
    penalty = _scale_eff_penalties(
        scale, ind[1:], max_scale_deviation, max_efficiency_deviation
    )
    if total_cost > budget_limit:
        penalty += 100_000.0 * float(total_cost - budget_limit)
    actual_trees = float(baseline_trees) * scale
    gwp = _gwp(final, impact_matrix, impact_cols)
    return float(-actual_trees + penalty), float(gwp + penalty)


def evaluate_compliance_constrained(
    ind,
    costs,
    impact_matrix,
    impact_cols,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
    gwp_target,
):
    """Minimize cost while meeting a GWP cap. Returns (cost,)."""
    final, scale = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    total_cost = _cost(final, costs)
    gwp = _gwp(final, impact_matrix, impact_cols)
    penalty = _scale_eff_penalties(
        scale, ind[1:], max_scale_deviation, max_efficiency_deviation, weight=100_000.0
    )
    if gwp > gwp_target:
        penalty += 1_000_000.0 * float(gwp - gwp_target)
    return (float(total_cost + penalty),)


def evaluate_cost_only_constrained(
    ind,
    costs,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
):
    final, scale = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    penalty = _scale_eff_penalties(
        scale, ind[1:], max_scale_deviation, max_efficiency_deviation
    )
    return (_cost(final, costs) + penalty,)


def evaluate_single_impact_constrained(
    ind,
    matrix,
    colname,
    cols,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
):
    final, scale = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    penalty = _scale_eff_penalties(
        scale, ind[1:], max_scale_deviation, max_efficiency_deviation
    )
    try:
        idx = list(cols).index(colname)
        impact = float(np.dot(final, np.asarray(matrix, dtype=float)[:, idx].flatten()[: len(final)]))
    except (ValueError, IndexError):
        impact = 0.0
    return (impact + penalty,)


def split_genome(ind: Sequence[float], n_eff: int, n_choice: int):
    """Return (scale, efficiency_genes, choice_genes).

    Needed because the bound penalties must not be applied to choice genes,
    which legitimately live in [0, 1) rather than around 1.0.
    """
    scale = float(ind[0])
    eff = list(ind[1 : 1 + int(n_eff)])
    choice = list(ind[1 + int(n_eff) : 1 + int(n_eff) + int(n_choice)])
    return scale, eff, choice


def evaluate_with_substitutions(
    ind,
    costs,
    impact_matrix,
    impact_cols,
    base_amounts,
    baseline_trees,
    scale_materials_mask,
    efficiency_materials_mask,
    max_scale_deviation,
    max_efficiency_deviation,
    materials,
    substitutions,
    objective_cols,
    min_trees=None,
    per_tree=True,
    scenario=None,
    carbon_params=None,
):
    """Minimize a tuple of burdens over continuous inputs AND discrete practices.

    ``objective_cols`` names the impact columns after cost, e.g.
    ``[GWP_COL, "kg N-Eq/Unit"]`` gives a 3-tuple of (cost, GWP, eutrophication).

    ``scenario`` is a fixed management context (e.g. PRR-resistant rootstock)
    applied to every individual before the optimizer's own choices. It is not
    searched over: the user pins it and fronts are compared across scenarios.

    ``per_tree=True`` divides by harvested trees. This matters: substitutions
    that change survival or rotation change the denominator, and a grower cares
    about cost and carbon per tree sold, not per acre.
    """
    from .substitutions import apply_substitutions, decode_choices

    n_eff = int(np.sum(np.asarray(efficiency_materials_mask, dtype=bool)))
    n_choice = len(substitutions)
    scale, eff_genes, choice_genes = split_genome(ind, n_eff, n_choice)

    amounts, _ = apply_individual(
        ind, base_amounts, scale_materials_mask, efficiency_materials_mask
    )
    choices = decode_choices(choice_genes, substitutions)
    amounts, unit_costs, trees, _ = apply_substitutions(
        amounts,
        costs,
        materials,
        float(baseline_trees) * scale,
        efficiency_materials_mask,
        choices,
        substitutions,
        scenario=scenario,
    )

    penalty = _scale_eff_penalties(
        scale, eff_genes, max_scale_deviation, max_efficiency_deviation
    )
    if min_trees is not None and trees < float(min_trees):
        penalty += 1_000_000.0 * float(min_trees - trees)

    denom = max(float(trees), 1e-9) if per_tree else 1.0
    out = [_cost(amounts, unit_costs) / denom + penalty]
    for col in objective_cols:
        burden = _impact(amounts, impact_matrix, impact_cols, col)
        if carbon_params is not None and col == GWP_COL:
            # Biogenic uptake offsets cultivation emissions. This is the only
            # term that scales with output rather than input, and it is what
            # makes cost and carbon genuinely conflict. On a per-tree basis it
            # reduces to an additive constant and cannot affect the ranking --
            # see lib/carbon.py -- so callers wanting it to bite must pass
            # per_tree=False.
            from .carbon import net_gwp

            burden = net_gwp(burden, trees, basis="acre", params=carbon_params)
        out.append(burden / denom + penalty)
    return tuple(out)


def _repair_factory(max_scale_dev: float, max_eff_dev: float, n_eff: int, n_choice: int):
    """Clip every gene back into its box after crossover or mutation.

    Bounds used to be enforced only by a fixed penalty inside the evaluators,
    plus clipping inside the mutation operator. cxBlend extrapolates beyond its
    parents, so crossover children could leave the box and were held in only by
    that penalty. Once biogenic sequestration entered the objective the penalty
    became cheap relative to the carbon "gained" by inflating production scale,
    and the search escaped the box entirely (scale reached 63x baseline).

    A penalty cannot reliably bound a variable that appears in a large negative
    term. Repair can, so bounds are now hard.
    """
    first_choice = 1 + int(n_eff)

    def repair(ind):
        ind[0] = float(np.clip(ind[0], 1 - max_scale_dev, 1 + max_scale_dev))
        for i in range(1, min(first_choice, len(ind))):
            ind[i] = float(np.clip(ind[i], 1 - max_eff_dev, 1 + max_eff_dev))
        for i in range(first_choice, len(ind)):
            ind[i] = float(np.clip(ind[i], 0.0, 0.999999))
        return ind

    return repair


def _reset_creator(kind) -> None:
    """Rebuild the DEAP creator for n minimization objectives.

    Accepts an int (number of objectives) or the legacy "nsga" / "single" strings.
    """
    if kind == "nsga":
        n_obj = 2
    elif kind == "single":
        n_obj = 1
    else:
        n_obj = int(kind)
    if n_obj < 1:
        raise ValueError(f"n_obj must be >= 1, got {n_obj}")
    for name in ("FitnessMin", "Individual"):
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMin", base.Fitness, weights=tuple([-1.0] * n_obj))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def _make_individual_factory(
    n_eff: int, max_scale_dev: float, max_eff_dev: float, n_choice: int = 0
):
    """Genome layout: [scale] + [efficiency × n_eff] + [choice × n_choice].

    Choice genes live in [0, 1) and are floored to an option index at decode
    time, so cxBlend and the gaussian mutation below need no special casing.
    """

    def create_individual():
        ind = [random.uniform(1 - max_scale_dev, 1 + max_scale_dev)]
        for _ in range(int(n_eff)):
            ind.append(random.uniform(1 - max_eff_dev, 1 + max_eff_dev))
        for _ in range(int(n_choice)):
            ind.append(random.random())
        return ind

    return create_individual


def _bounded_mutate_factory(
    max_scale_dev: float, max_eff_dev: float, n_eff: int = 0, n_choice: int = 0
):
    """Gaussian mutation clipped per gene class.

    Choice genes get a wider sigma because a small nudge inside one option's
    bucket is a no-op — they need to be able to cross a bucket boundary.
    """
    first_choice = 1 + int(n_eff) if n_choice else None

    def bounded_mutate(ind, mu=0.0, sigma=0.05, indpb=0.2):
        mutated = list(ind)
        for i in range(len(mutated)):
            if random.random() >= indpb:
                continue
            is_choice = first_choice is not None and i >= first_choice
            step = random.gauss(mu, 0.25 if is_choice else sigma)
            mutated[i] += step
            if is_choice:
                mutated[i] = float(np.clip(mutated[i], 0.0, 0.999999))
            elif i == 0:
                mutated[i] = float(np.clip(mutated[i], 1 - max_scale_dev, 1 + max_scale_dev))
            else:
                mutated[i] = float(np.clip(mutated[i], 1 - max_eff_dev, 1 + max_eff_dev))
        return (mutated,)

    return bounded_mutate


def run_nsga2_constrained(
    popsize,
    ngen,
    cxpb,
    mutpb,
    costs,
    matrix,
    impact_cols,
    base_amounts,
    baseline_trees,
    scale_mask,
    efficiency_mask,
    max_scale_dev,
    max_eff_dev,
    eval_func: Optional[Callable] = None,
    seed: Optional[int] = None,
    n_obj: int = 2,
    n_choice: int = 0,
    **eval_kwargs,
):
    """NSGA-II. Returns the first non-dominated front (list of Individuals).

    ``n_obj`` must match the tuple length returned by ``eval_func``. selNSGA2's
    crowding distance generalizes to any number of objectives, so 3 works
    without changing the selection operator.
    """
    set_seed(seed)
    _reset_creator(n_obj)
    n_eff = int(np.sum(np.asarray(efficiency_mask, dtype=bool)))
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        _make_individual_factory(n_eff, max_scale_dev, max_eff_dev, n_choice),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    if eval_func is None:
        eval_func = evaluate_cost_gwp_constrained
    toolbox.register(
        "evaluate",
        eval_func,
        costs=costs,
        impact_matrix=matrix,
        impact_cols=impact_cols,
        base_amounts=base_amounts,
        baseline_trees=baseline_trees,
        scale_materials_mask=scale_mask,
        efficiency_materials_mask=efficiency_mask,
        max_scale_deviation=max_scale_dev,
        max_efficiency_deviation=max_eff_dev,
        **eval_kwargs,
    )
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register(
        "mutate", _bounded_mutate_factory(max_scale_dev, max_eff_dev, n_eff, n_choice)
    )
    toolbox.register("select", tools.selNSGA2)

    repair = _repair_factory(max_scale_dev, max_eff_dev, n_eff, n_choice)
    pop = toolbox.population(n=popsize)
    for ind in pop:
        repair(ind)
        fit = toolbox.evaluate(ind)
        if not isinstance(fit, tuple) or len(fit) != n_obj:
            raise RuntimeError(
                f"NSGA evaluate must return a {n_obj}-tuple, got {fit!r}"
            )
        ind.fitness.values = fit

    for _gen in range(ngen):
        offspring = toolbox.select(pop, popsize)
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                child1, child2 = toolbox.mate(offspring[i - 1], offspring[i])
                offspring[i - 1] = creator.Individual(child1)
                offspring[i] = creator.Individual(child2)
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated, = toolbox.mutate(offspring[i])
                offspring[i] = creator.Individual(mutated)
        for ind in offspring:
            repair(ind)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, popsize)

    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]


def run_single_constrained(
    obj_func,
    popsize,
    ngen,
    cxpb,
    mutpb,
    base_amounts,
    baseline_trees,
    scale_mask,
    efficiency_mask,
    max_scale_dev,
    max_eff_dev,
    *args,
    seed: Optional[int] = None,
    **kwargs,
):
    """Single-objective GA. Returns the Hall-of-Fame individual."""
    set_seed(seed)
    _reset_creator("single")
    n_eff = int(np.sum(np.asarray(efficiency_mask, dtype=bool)))
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        _make_individual_factory(n_eff, max_scale_dev, max_eff_dev),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        obj_func,
        *args,
        base_amounts=base_amounts,
        baseline_trees=baseline_trees,
        scale_materials_mask=scale_mask,
        efficiency_materials_mask=efficiency_mask,
        max_scale_deviation=max_scale_dev,
        max_efficiency_deviation=max_eff_dev,
        **kwargs,
    )
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", _bounded_mutate_factory(max_scale_dev, max_eff_dev))
    toolbox.register("select", tools.selTournament, tournsize=3)

    def _safe_eval(ind):
        try:
            fit = toolbox.evaluate(ind)
            if not isinstance(fit, tuple):
                fit = (fit,)
            if any(not np.isfinite(v) for v in fit):
                return (1e10,)
            return fit
        except Exception:
            return (1e10,)

    repair = _repair_factory(max_scale_dev, max_eff_dev, n_eff, 0)
    pop = toolbox.population(n=popsize)
    for ind in pop:
        repair(ind)
        ind.fitness.values = _safe_eval(ind)
    hof = tools.HallOfFame(1)

    for _gen in range(ngen):
        offspring = toolbox.select(pop, popsize)
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                child1, child2 = toolbox.mate(offspring[i - 1], offspring[i])
                offspring[i - 1] = creator.Individual(child1)
                offspring[i] = creator.Individual(child2)
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated, = toolbox.mutate(offspring[i])
                offspring[i] = creator.Individual(mutated)
        for ind in offspring:
            repair(ind)
            ind.fitness.values = _safe_eval(ind)
        pop[:] = offspring
        hof.update(pop)

    return hof[0]
