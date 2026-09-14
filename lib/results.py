"""Result tables, plots, and Excel export for the Streamlit pages."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .defaults import GWP_COL
from .optimize import apply_individual


def gwp_vector(arrays):
    cols = list(arrays["traci_impact_cols"])
    matrix = np.asarray(arrays["impact_matrix"], dtype=float)
    return matrix[:, cols.index(GWP_COL)] if GWP_COL in cols else np.zeros(len(arrays["materials"]))


def metrics_for_ind(ind, arrays):
    amounts, scale = apply_individual(
        ind, arrays["base_amounts"], arrays["scale_mask"], arrays["efficiency_mask"]
    )
    trees = float(arrays["baseline_trees"]) * scale
    cost = float(amounts @ np.asarray(arrays["costs"], dtype=float))
    gwp = float(amounts @ gwp_vector(arrays))
    denominator = max(trees, 1e-9)
    return {
        "amounts": amounts, "trees": trees, "total_cost": cost, "total_gwp": gwp,
        "cost_per_tree": cost / denominator, "gwp_per_tree": gwp / denominator,
    }


def pareto_dataframe(front, arrays):
    rows = []
    for ind in front:
        m = metrics_for_ind(ind, arrays)
        rows.append({
            "Trees": m["trees"], "Total Cost": m["total_cost"], "Total GWP": m["total_gwp"],
            "Cost/Tree": m["cost_per_tree"], "GWP/Tree": m["gwp_per_tree"],
            "Individual": list(ind),
        })
    return pd.DataFrame(rows).sort_values("Total Cost").reset_index(drop=True) if rows else pd.DataFrame()


def knee_index(df):
    if df.empty:
        raise ValueError("cannot select a knee from an empty table")
    x = df["Total Cost"].to_numpy(float)
    y = df["Total GWP"].to_numpy(float)
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return int(np.argmin(np.hypot(xn, yn)))


def material_change_table(materials, baseline, selected, costs, gwp, scale_mask, efficiency_mask):
    base = np.asarray(baseline, float)
    chosen = np.asarray(selected, float)
    kind = np.where(scale_mask, "Scale", np.where(efficiency_mask, "Efficiency", "Fixed"))
    return pd.DataFrame({
        "Material": materials, "Type": kind, "Baseline Amount": base,
        "Selected Amount": chosen,
        "Change (%)": np.where(base != 0, 100 * (chosen / base - 1), 0),
        "Cost Change ($)": (chosen - base) * np.asarray(costs, float),
        "GWP Change (kg CO2-eq)": (chosen - base) * np.asarray(gwp, float),
    })


def hotspot_tables(materials, amounts, costs, gwp, n=8):
    amount = np.asarray(amounts, float)
    cost = pd.DataFrame({"Material": materials, "Contribution": amount * np.asarray(costs, float)})
    carbon = pd.DataFrame({"Material": materials, "Contribution": amount * np.asarray(gwp, float)})
    return (
        cost.sort_values("Contribution", ascending=False).head(n).reset_index(drop=True),
        carbon.sort_values("Contribution", ascending=False).head(n).reset_index(drop=True),
    )


def decision_card(materials, baseline, selected, costs, gwp):
    base = np.asarray(baseline, float)
    chosen = np.asarray(selected, float)
    changed = np.where(np.abs(chosen - base) > 1e-9)[0]
    if len(changed) == 0:
        return "No material amounts changed from baseline."
    ranked = sorted(changed, key=lambda i: abs((chosen[i] - base[i]) * costs[i]), reverse=True)[:5]
    return "\n".join(f"- **{materials[i]}:** {base[i]:,.3g} → {chosen[i]:,.3g}" for i in ranked)


def plot_cost_gwp(df, base_cost, base_gwp, base_trees):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["Total Cost"], df["Total GWP"], color="#1f6f5b")
    ax.scatter([base_cost], [base_gwp], marker="*", s=150, color="#c44e52", label="Baseline")
    ax.set(xlabel="Total cost ($/acre-rotation)", ylabel="GWP (kg CO2-eq/acre-rotation)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_budget(df, budget, base_cost, base_trees, base_gwp):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["Total Cost"], df["Trees"], c=df["Total GWP"], cmap="viridis")
    ax.axvline(budget, color="black", linestyle="--", label="Budget")
    ax.set(xlabel="Total cost ($)", ylabel="Harvested trees")
    ax.legend()
    fig.tight_layout()
    return fig


def excel_bytes(inventory, changes, metrics, pareto=None, cost_hotspots=None, gwp_hotspots=None):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        inventory.to_excel(writer, "Inventory", index=False)
        changes.to_excel(writer, "Selected plan", index=False)
        pd.DataFrame([metrics]).to_excel(writer, "Metrics", index=False)
        if pareto is not None:
            pareto.drop(columns=["Individual"], errors="ignore").to_excel(writer, "Pareto front", index=False)
        if cost_hotspots is not None:
            cost_hotspots.to_excel(writer, "Cost hotspots", index=False)
        if gwp_hotspots is not None:
            gwp_hotspots.to_excel(writer, "GWP hotspots", index=False)
    return buffer.getvalue()
