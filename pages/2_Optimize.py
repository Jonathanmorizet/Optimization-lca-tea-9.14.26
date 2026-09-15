"""Part 2: grower-language optimization UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from lib.data_model import baseline_totals, build_optimizer_arrays
from lib.carbon import CarbonParams, net_gwp, retained_at_100_years, sequestration
from lib.defaults import GWP_COL
from lib.substitutions import (
    apply_substitutions,
    decode_choices,
    default_substitutions,
    unsourced_options,
)
from lib.optimize import (
    apply_individual,
    evaluate_cost_gwp_impact_constrained,
    evaluate_with_substitutions,
    split_genome,
    evaluate_budget_constrained,
    evaluate_compliance_constrained,
    evaluate_cost_only_constrained,
    evaluate_single_impact_constrained,
    run_nsga2_constrained,
    run_single_constrained,
)
from lib.results import (
    decision_card,
    excel_bytes,
    gwp_vector,
    hotspot_tables,
    knee_index,
    material_change_table,
    metrics_for_ind,
    pareto_dataframe,
    plot_budget,
    plot_cost_gwp,
)

st.set_page_config(page_title="Optimize", page_icon="📈", layout="wide")
st.title("Optimize the farm plan")
st.sidebar.caption("Model revision: weed-control-only · 2026-09-15")

# Keep the grower-facing optimizer limited to decisions supported by the
# deployed model.  This page-level allowlist is intentional defense in depth:
# even if an unsupported scenario is accidentally restored to
# default_substitutions(), it must not reappear as a selectable decision.
ALLOWED_PRACTICE_KEYS = frozenset({"weed_control"})

if not st.session_state.get("opt_ready"):
    st.warning("Save a farm inventory first.")
    st.page_link("pages/1_Farm_Setup.py", label="Go to Farm Setup", icon="🚜")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

try:
    arrays = build_optimizer_arrays(st.session_state.farm_config, st.session_state.inventory_df)
except Exception as exc:
    st.error(f"Could not build optimizer arrays: {exc}")
    st.stop()

base = baseline_totals(arrays)
gwp_u = gwp_vector(arrays)
materials = arrays["materials"]
traci_cols = arrays["traci_impact_cols"]

st.info(
    f"Baseline **{base['trees']:,.0f} trees**: ${base['total_cost']:,.0f} total "
    f"(${base['cost_per_tree']:.2f}/tree) · {base['total_gwp']:,.0f} kg CO2-eq "
    f"({base['gwp_per_tree']:.2f} kg/tree). Source: {st.session_state.get('source', 'workbook')}."
)

GOALS = {
    "Compare practice options (recommended)": "practices",
    "Show cost vs carbon tradeoffs": "tradeoff",
    "Max trees under my budget": "budget",
    "Cut carbon by X% at lowest cost": "compliance",
    "Lowest cost only": "cost",
    "Minimize one environmental impact": "impact",
}
goal_label = st.selectbox("What do you want the tool to do?", list(GOALS.keys()))
goal = GOALS[goal_label]

budget_limit = None
gwp_reduction_pct = 15
selected_impact = GWP_COL if GWP_COL in traci_cols else (traci_cols[0] if traci_cols else GWP_COL)

subs = []
third_axis = None
keep_pct = 100
carbon = None

if goal in ("practices", "tradeoff"):
    st.markdown("#### Biogenic carbon")
    use_carbon = st.checkbox(
        "Credit carbon stored in the harvested trees",
        value=True,
        help=(
            "Sequestration is the only term that scales with output rather "
            "than input, so it is what makes cost and carbon genuinely "
            "conflict. Without it the tradeoff curve collapses to a point."
        ),
    )
    if use_carbon:
        # Fixed from the manuscript, not a user input. 14.1 kg above-ground and
        # 4.2 kg below-ground per 7-ft tree are measured values for the
        # functional unit, and 0.893 is the wet-mass allocation between tree and
        # wreath. Letting a grower type over these would let them manufacture
        # any net-GWP result they liked.
        carbon = CarbonParams()
        m1, m2, m3 = st.columns(3)
        m1.metric("Above-ground CO2", f"{carbon.above_ground_co2:.1f} kg/tree")
        m2.metric("Below-ground CO2", f"{carbon.below_ground_co2:.1f} kg/tree")
        m3.metric("Burden share to trees", f"{carbon.tree_allocation:.3f}")
        st.caption(
            "Fixed from the published LCA for the 7-ft functional unit "
            "(Sections 2.2.1 and 3.2). Not editable: these are measured values, "
            "not preferences."
        )
        stock = sequestration(base["trees"], carbon)
        kept = retained_at_100_years(base["trees"], carbon)
        st.caption(
            f"Baseline stand stores **{stock:,.0f} kg CO2** at harvest "
            f"({carbon.co2_per_tree:.1f} kg/tree). Farm-gate net GWP "
            f"**{net_gwp(base['total_gwp'], base['trees'], 'acre', carbon):,.0f} "
            f"kg CO2-eq/acre**."
        )
        st.warning(
            f"This is a carbon **stock at harvest**, not a durable removal. "
            f"Under baseline mulching only about {kept:,.0f} kg "
            f"({100 * kept / max(stock, 1e-9):.0f}%) remains at 100 years. "
            "Results are farm gate: cradle-to-grave GWP is positive once "
            "consumer pickup and end-of-life are included, neither of which a "
            "grower controls."
        )

if goal in ("practices", "tradeoff"):
    st.markdown("#### Keep the block productive")
    keep_pct = st.slider(
        "Don't drop below this share of baseline trees (%)", 80, 110, 100, 5,
        help=(
            "Without a floor the cheapest and lowest-carbon plan is simply a "
            "smaller farm, and the tradeoff curve collapses to a single point."
        ),
    )
    st.caption(
        f"Plans must harvest at least **{base['trees'] * keep_pct / 100:,.0f} trees**."
    )

if goal == "practices":
    st.markdown("#### Practices the optimizer may choose between")
    all_subs = [
        sub for sub in default_substitutions()
        if sub.key in ALLOWED_PRACTICE_KEYS
    ]
    picked = st.multiselect(
        "Include these decisions",
        options=[sub.key for sub in all_subs],
        default=[sub.key for sub in all_subs],
        format_func=lambda k: next(x.label for x in all_subs if x.key == k),
    )
    subs = [sub for sub in all_subs if sub.key in picked]
    for sub in subs:
        st.caption(f"**{sub.label}** — {sub.help}")
    flagged = unsourced_options(subs)
    if flagged:
        st.warning(
            "These options carry placeholder cost premiums that are not in "
            "either workbook. Replace them with real quotes before reporting, "
            "or sweep them and report the breakeven premium instead:\n\n"
            + "\n".join(f"- {f}" for f in flagged)
        )

if goal in ("practices", "tradeoff"):
    others = [c for c in traci_cols if c != GWP_COL]
    if others:
        third_axis = st.selectbox(
            "Third objective (optional) — exposes burden shifting",
            ["(none — cost and carbon only)"] + others,
            index=1 if others else 0,
            help=(
                "Carbon bought per dollar varies by orders of magnitude across "
                "materials, and it does not track the other TRACI categories. "
                "A plan optimized on carbon alone can worsen nutrient or "
                "toxicity burdens."
            ),
        )
        if third_axis.startswith("("):
            third_axis = None

if goal == "budget":
    budget_limit = st.number_input(
        "Season / rotation budget ($)",
        min_value=float(base["total_cost"] * 0.5),
        max_value=float(base["total_cost"] * 2.0),
        value=float(base["total_cost"]),
        step=100.0,
    )
elif goal == "compliance":
    gwp_reduction_pct = st.slider("Cut carbon by (%)", 0, 50, 15, 5)
    st.caption(f"Target GWP ≤ {base['total_gwp'] * (1 - gwp_reduction_pct / 100):,.1f} kg CO2-eq")
elif goal == "impact":
    selected_impact = st.selectbox("Which impact to minimize?", traci_cols)

with st.expander("Advanced — genetic algorithm and material types", expanded=False):
    a1, a2, a3, a4 = st.columns(4)
    popsize = a1.slider("Population size", 20, 200, 80)
    ngen = a2.slider("Generations", 10, 200, 40)
    cxpb = a3.slider("Crossover probability", 0.0, 1.0, 0.7)
    mutpb = a4.slider("Mutation probability", 0.0, 1.0, 0.3)
    b1, b2 = st.columns(2)
    max_scale_pct = b1.slider("Production (tree count) ±%", 0, 30, 10)
    max_eff_pct = b2.slider("Input efficiency ±%", 5, 30, 20)
    st.markdown("Override which inputs scale with trees vs. can be used more/less efficiently.")
    scale_idx = st.multiselect(
        "SCALE (moves with tree count)",
        options=list(range(len(materials))),
        default=list(np.where(arrays["scale_mask"])[0]),
        format_func=lambda i: materials[i],
    )
    remaining = [i for i in range(len(materials)) if i not in scale_idx]
    default_eff = [i for i in remaining if arrays["efficiency_mask"][i]]
    eff_idx = st.multiselect(
        "EFFICIENCY (fertilizer, diesel, sprays)",
        options=remaining,
        default=default_eff,
        format_func=lambda i: materials[i],
    )
    seed = st.number_input("Random seed (optional, 0 = unset)", min_value=0, value=42, step=1)

scale_mask = np.zeros(len(materials), dtype=bool)
scale_mask[scale_idx] = True
efficiency_mask = np.zeros(len(materials), dtype=bool)
efficiency_mask[eff_idx] = True
arrays = dict(arrays)
arrays["scale_mask"] = scale_mask
arrays["efficiency_mask"] = efficiency_mask
max_scale_dev = max_scale_pct / 100.0
max_eff_dev = max_eff_pct / 100.0
seed_arg = int(seed) if seed else None

ga = dict(popsize=popsize, ngen=ngen, cxpb=cxpb, mutpb=mutpb)


def practice_dataframe(front, arrays, subs, obj_cols):
    """One row per non-dominated plan, with the practices it chose."""
    n_eff = int(np.asarray(arrays["efficiency_mask"], dtype=bool).sum())
    rows = []
    for ind in front:
        scale, _, choice_genes = split_genome(list(ind), n_eff, len(subs))
        amounts, _ = apply_individual(
            ind, arrays["base_amounts"], arrays["scale_mask"], arrays["efficiency_mask"]
        )
        choices = decode_choices(choice_genes, subs)
        amounts, unit_costs, trees, names = apply_substitutions(
            amounts, arrays["costs"], arrays["materials"],
            arrays["baseline_trees"] * scale, arrays["efficiency_mask"], choices, subs,
        )
        row = {
            "Trees": int(round(trees)),
            "Cost/Tree": float(amounts @ unit_costs) / max(trees, 1e-9),
            "Total Cost": float(amounts @ unit_costs),
        }
        for col in obj_cols:
            j = list(arrays["traci_impact_cols"]).index(col)
            total = float(amounts @ np.asarray(arrays["impact_matrix"], dtype=float)[:, j])
            row[f"{col} /tree"] = total / max(trees, 1e-9)
        for sub, name in zip(subs, names):
            row[sub.label] = name
        row["Individual"] = list(ind)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df if df.empty else df.sort_values("Cost/Tree").reset_index(drop=True)


def _change_table(amounts):
    return material_change_table(
        materials,
        arrays["base_amounts"],
        amounts,
        arrays["costs"],
        gwp_u,
        scale_mask,
        efficiency_mask,
    )



def _emission_split_chart(amounts, title="Upstream vs direct emissions"):
    """Stacked split of each impact category into upstream and direct shares.

    Upstream is the ecoinvent cradle-to-gate factor; direct is the IPCC field
    layer added in build_optimizer_arrays. Rows flagged "combustion" contribute
    only upstream, because their activity already includes emission to air --
    adding a direct layer there would double count.
    """
    amounts = np.asarray(amounts, dtype=float)
    total_m = np.asarray(arrays["impact_matrix"], dtype=float)
    direct_m = np.asarray(arrays.get("direct_matrix", np.zeros_like(total_m)), dtype=float)
    cols = list(arrays["traci_impact_cols"])

    rows = []
    for j, col in enumerate(cols):
        total = float(amounts @ total_m[:, j])
        direct = float(amounts @ direct_m[:, j])
        if total == 0 and direct == 0:
            continue
        rows.append({
            "Category": col,
            "Upstream": total - direct,
            "Direct (field)": direct,
            "Total": total,
            "Direct share": (direct / total) if total else 0.0,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)

    st.markdown(f"### {title}")
    norm = st.checkbox(
        "Show as share of each category", value=True, key=f"norm_{title}",
        help="Categories span many orders of magnitude, so absolute values are "
             "hard to compare on one axis.",
    )
    plot = df.set_index("Category")[["Upstream", "Direct (field)"]]
    if norm:
        denom = plot.sum(axis=1).replace(0, np.nan)
        plot = plot.div(denom, axis=0).fillna(0.0)
    st.bar_chart(plot, stack=True, height=340)

    shown = df[["Category", "Upstream", "Direct (field)", "Total", "Direct share"]].copy()
    shown["Direct share"] = shown["Direct share"].map(lambda v: f"{v:.0%}")
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.caption(
        "Direct emissions are field N2O, ammonia and nitrate from fertilizer, "
        "plus CO2 from liming. Diesel shows no direct share by design: its "
        "ecoinvent activity already includes combustion."
    )
    return df


def _show_card_and_hotspots(amounts, title="Selected plan"):
    st.markdown(f"### {title}")
    st.markdown(decision_card(materials, arrays["base_amounts"], amounts, arrays["costs"], gwp_u))
    cost_h, gwp_h = hotspot_tables(materials, amounts, arrays["costs"], gwp_u)
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Top cost drivers**")
        st.dataframe(cost_h, use_container_width=True)
    with h2:
        st.markdown("**Top carbon drivers**")
        st.dataframe(gwp_h, use_container_width=True)
    return cost_h, gwp_h


if st.button("Run optimization", type="primary"):
    with st.spinner("Searching farm plans…"):
        try:
            if goal == "practices":
                min_trees = base["trees"] * keep_pct / 100.0
                obj_cols = [GWP_COL] + ([third_axis] if third_axis else [])
                front = run_nsga2_constrained(
                    ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                    arrays["costs"], arrays["impact_matrix"], traci_cols,
                    arrays["base_amounts"], arrays["baseline_trees"],
                    scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                    seed=seed_arg,
                    n_obj=1 + len(obj_cols),
                    n_choice=len(subs),
                    eval_func=evaluate_with_substitutions,
                    materials=materials,
                    substitutions=subs,
                    objective_cols=obj_cols,
                    min_trees=min_trees,
                    per_tree=False,
                    carbon_params=carbon,
                )
                df_p = practice_dataframe(front, arrays, subs, obj_cols)
                st.session_state.last_run = {
                    "kind": "practices", "goal": goal_label, "df": df_p,
                    "obj_cols": obj_cols, "fresh": True,
                }
            elif goal == "tradeoff":
                min_trees = base["trees"] * keep_pct / 100.0
                obj_cols = [GWP_COL] + ([third_axis] if third_axis else [])
                if carbon is not None:
                    front = run_nsga2_constrained(
                        ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                        arrays["costs"], arrays["impact_matrix"], traci_cols,
                        arrays["base_amounts"], arrays["baseline_trees"],
                        scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                        seed=seed_arg, n_obj=1 + len(obj_cols), n_choice=0,
                        eval_func=evaluate_with_substitutions,
                        materials=materials, substitutions=[],
                        objective_cols=obj_cols, per_tree=False,
                        carbon_params=carbon, min_trees=min_trees,
                    )
                elif third_axis:
                    front = run_nsga2_constrained(
                        ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                        arrays["costs"], arrays["impact_matrix"], traci_cols,
                        arrays["base_amounts"], arrays["baseline_trees"],
                        scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                        seed=seed_arg, n_obj=3,
                        eval_func=evaluate_cost_gwp_impact_constrained,
                        third_impact_col=third_axis, min_trees=min_trees,
                    )
                else:
                    front = run_nsga2_constrained(
                        ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                        arrays["costs"], arrays["impact_matrix"], traci_cols,
                        arrays["base_amounts"], arrays["baseline_trees"],
                        scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                        seed=seed_arg, min_trees=min_trees,
                    )
                df_p = pareto_dataframe(front, arrays)
                if len(df_p) < 3:
                    st.info(
                        "This goal returned very few distinct plans. That is "
                        "expected: with all-positive cost and impact factors, "
                        "cutting any input improves both objectives at once, so "
                        "there is nothing to trade. Use **Compare practice "
                        "options** for a real tradeoff curve."
                    )
                st.session_state.last_run = {
                    "kind": "pareto", "goal": goal_label, "df": df_p, "budget": None, "fresh": True,
                }
            elif goal == "budget":
                front = run_nsga2_constrained(
                    ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                    arrays["costs"], arrays["impact_matrix"], traci_cols,
                    arrays["base_amounts"], arrays["baseline_trees"],
                    scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                    eval_func=evaluate_budget_constrained,
                    budget_limit=budget_limit,
                    seed=seed_arg,
                )
                df_p = pareto_dataframe(front, arrays)
                df_p = df_p[df_p["Total Cost"] <= budget_limit + 1e-6].reset_index(drop=True)
                st.session_state.last_run = {
                    "kind": "budget", "goal": goal_label, "df": df_p, "budget": budget_limit, "fresh": True,
                }
            elif goal == "compliance":
                target = base["total_gwp"] * (1 - gwp_reduction_pct / 100.0)
                best = run_single_constrained(
                    evaluate_compliance_constrained, ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                    arrays["base_amounts"], arrays["baseline_trees"],
                    scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                    arrays["costs"], arrays["impact_matrix"], traci_cols,
                    gwp_target=target, seed=seed_arg,
                )
                st.session_state.last_run = {
                    "kind": "single", "goal": goal_label, "ind": list(best),
                    "target": target, "pct": gwp_reduction_pct, "fresh": True,
                }
            elif goal == "cost":
                best = run_single_constrained(
                    evaluate_cost_only_constrained, ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                    arrays["base_amounts"], arrays["baseline_trees"],
                    scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                    arrays["costs"], seed=seed_arg,
                )
                st.session_state.last_run = {"kind": "single", "goal": goal_label, "ind": list(best), "fresh": True}
            else:
                best = run_single_constrained(
                    evaluate_single_impact_constrained, ga["popsize"], ga["ngen"], ga["cxpb"], ga["mutpb"],
                    arrays["base_amounts"], arrays["baseline_trees"],
                    scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                    arrays["impact_matrix"], selected_impact, traci_cols,
                    seed=seed_arg,
                )
                st.session_state.last_run = {
                    "kind": "single", "goal": goal_label, "ind": list(best), "impact": selected_impact, "fresh": True,
                }
        except Exception as exc:
            st.exception(exc)
            st.stop()

run = st.session_state.get("last_run")
if not run:
    st.caption("Choose a goal and click **Run optimization**.")
    st.stop()

st.subheader(run["goal"])

if run["kind"] == "practices":
    df_p = run["df"]
    if df_p is None or df_p.empty:
        st.error("No feasible plans found. Lower the tree floor or widen Advanced bounds.")
        st.stop()
    obj_cols = run["obj_cols"]
    st.dataframe(df_p.drop(columns=["Individual"], errors="ignore"), use_container_width=True)

    sub_labels = [c for c in df_p.columns if c not in
                  ["Trees", "Cost/Tree", "Total Cost", "Individual"]
                  and not c.endswith("/tree")]
    st.markdown("### What wins on each objective")
    cols = st.columns(1 + len(obj_cols))
    picks = [("Cheapest", "Cost/Tree")] + [(c, f"{c} /tree") for c in obj_cols]
    for (title, key), col in zip(picks, cols):
        i = int(df_p[key].idxmin())
        with col:
            st.metric(f"Best: {title}", f"{df_p.loc[i, key]:.4g}")
            for lab in sub_labels:
                st.caption(f"**{lab}:** {df_p.loc[i, lab]}")

    best_sets = {tuple(df_p.loc[int(df_p[k].idxmin()), sub_labels]) for _, k in picks}
    if len(best_sets) > 1:
        st.success(
            f"{len(best_sets)} different practice combinations win on different "
            "objectives — this is burden shifting, and it is why a single-objective "
            "run would mislead."
        )
    else:
        st.info("One combination wins on every objective here, so there is no tradeoff to make.")

    labels = [
        f"Plan {i+1}: {int(r.Trees)} trees, ${r['Cost/Tree']:.2f}/tree"
        for i, r in df_p.iterrows()
    ]
    pick = st.selectbox("Inspect a plan", range(len(df_p)), format_func=lambda i: labels[i])
    row = df_p.iloc[pick]
    for lab in sub_labels:
        st.write(f"**{lab}:** {row[lab]}")
    amounts, _ = apply_individual(row["Individual"], arrays["base_amounts"], scale_mask, efficiency_mask)
    change = _change_table(amounts)
    st.markdown("### Material changes (before substitution effects)")
    st.dataframe(change, use_container_width=True)
    cost_h, gwp_h = _show_card_and_hotspots(amounts)
    _emission_split_chart(amounts)
    hist_df = change
    pareto_out = df_p
    metrics = {
        "goal": run["goal"],
        "trees": float(row["Trees"]),
        "total_cost": float(row["Total Cost"]),
        "cost_per_tree": float(row["Cost/Tree"]),
    }
    st.download_button(
        "Download Excel",
        data=excel_bytes(st.session_state.inventory_df, change, metrics, pareto_out, cost_h, gwp_h),
        file_name="tea_lca_practices.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if run.pop("fresh", False):
        st.session_state.history.append(
            {"scenario": run["goal"], "results": df_p.drop(columns=["Individual"], errors="ignore")}
        )
    st.stop()

if run["kind"] in ("pareto", "budget"):
    df_p = run["df"]
    if df_p is None or df_p.empty:
        st.error("No feasible plans found. Loosen the budget or Advanced bounds.")
        st.stop()
    show = df_p.drop(columns=["Individual"], errors="ignore")
    st.dataframe(show, use_container_width=True)
    if run["kind"] == "budget":
        fig = plot_budget(df_p, run["budget"], base["total_cost"], base["trees"], base["total_gwp"])
        best_i = int(df_p["Trees"].idxmax())
        st.success(
            f"Most trees on budget: **{int(df_p.loc[best_i, 'Trees']):,}** "
            f"for ${df_p.loc[best_i, 'Total Cost']:,.0f}."
        )
    else:
        fig = plot_cost_gwp(df_p, base["total_cost"], base["total_gwp"], base["trees"])
        best_i = knee_index(df_p)
        st.success(
            f"Balanced plan (knee): **{int(df_p.loc[best_i, 'Trees']):,} trees**, "
            f"${df_p.loc[best_i, 'Total Cost']:,.0f}, {df_p.loc[best_i, 'Total GWP']:,.0f} kg CO2-eq."
        )
    st.pyplot(fig)

    labels = [
        f"Plan {i+1}: {int(r.Trees)} trees, ${r['Total Cost']:,.0f}, {r['Total GWP']:,.0f} kg"
        for i, r in df_p.iterrows()
    ]
    pick = st.selectbox("Inspect a plan", range(len(df_p)), index=int(best_i), format_func=lambda i: labels[i])
    row = df_p.iloc[pick]
    amounts, _ = apply_individual(row["Individual"], arrays["base_amounts"], scale_mask, efficiency_mask)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trees", f"{int(row['Trees']):,}")
    k2.metric("Cost", f"${row['Total Cost']:,.0f}", f"{(row['Total Cost']/base['total_cost']-1)*100:+.1f}%")
    k3.metric("GWP", f"{row['Total GWP']:,.0f} kg", f"{(row['Total GWP']/max(base['total_gwp'],1e-9)-1)*100:+.1f}%")
    k4.metric("Per tree", f"${row['Cost/Tree']:.2f} | {row['GWP/Tree']:.2f} kg")
    change = _change_table(amounts)
    st.markdown("### Material changes")
    st.dataframe(change, use_container_width=True)
    cost_h, gwp_h = _show_card_and_hotspots(amounts)
    _emission_split_chart(amounts)
    hist_df = change
    pareto_out = df_p
else:
    ind = run["ind"]
    m = metrics_for_ind(ind, arrays)
    amounts = m["amounts"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trees", f"{int(m['trees']):,}", f"{(m['trees']/base['trees']-1)*100:+.1f}%")
    k2.metric("Cost", f"${m['total_cost']:,.0f}", f"{(m['total_cost']/base['total_cost']-1)*100:+.1f}%")
    k3.metric("GWP", f"{m['total_gwp']:,.0f} kg", f"{(m['total_gwp']/max(base['total_gwp'],1e-9)-1)*100:+.1f}%")
    k4.metric("Per tree", f"${m['cost_per_tree']:.2f} | {m['gwp_per_tree']:.2f} kg")
    if "target" in run:
        if m["total_gwp"] <= run["target"] * 1.01:
            st.success(f"Carbon target met ({run['pct']}% cut). GWP {m['total_gwp']:,.1f} ≤ {run['target']:,.1f}.")
        else:
            st.warning(f"Best attempt {m['total_gwp']:,.1f} kg vs target {run['target']:,.1f}. Raise generations or efficiency ±%.")
        extra = m["total_cost"] - base["total_cost"]
        saved = base["total_gwp"] - m["total_gwp"]
        if saved > 0:
            st.caption(f"Cost of carbon cut: ${extra:,.0f} for {saved:,.1f} kg → ${extra/saved:.2f} per kg CO2-eq.")
    change = _change_table(amounts)
    st.markdown("### Material changes")
    st.dataframe(change, use_container_width=True)
    cost_h, gwp_h = _show_card_and_hotspots(amounts)
    _emission_split_chart(amounts)
    hist_df = change
    pareto_out = None

if run["kind"] in ("pareto", "budget"):
    metrics = {
        "goal": run["goal"],
        "trees": float(row["Trees"]),
        "total_cost": float(row["Total Cost"]),
        "total_gwp": float(row["Total GWP"]),
        "cost_per_tree": float(row["Cost/Tree"]),
        "gwp_per_tree": float(row["GWP/Tree"]),
    }
else:
    metrics = {k: m[k] for k in ("trees", "total_cost", "total_gwp", "cost_per_tree", "gwp_per_tree")}
    metrics["goal"] = run["goal"]

st.download_button(
    "Download Excel",
    data=excel_bytes(st.session_state.inventory_df, change, metrics, pareto_out, cost_h, gwp_h),
    file_name="tea_lca_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if run.pop("fresh", False):
    st.session_state.history.append(
        {"scenario": run["goal"], "results": hist_df.drop(columns=["Individual"], errors="ignore")}
    )

with st.expander("Optimization history"):
    if st.session_state.history:
        for i, rec in enumerate(st.session_state.history, 1):
            st.write(f"**Run {i}: {rec['scenario']}**")
            st.dataframe(rec["results"], use_container_width=True)
    else:
        st.caption("No runs recorded yet.")
