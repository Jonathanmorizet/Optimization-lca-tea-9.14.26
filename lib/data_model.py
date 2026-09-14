"""Farm workbook ↔ optimizer arrays, plus DEAP .xlsm import.

Does not import Streamlit. Session keys used by pages:
    farm_config, inventory_df, opt_ready, source ("workbook"|"excel")
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, BinaryIO, Optional, Union

import numpy as np
import pandas as pd

from .defaults import (
    DEFAULT_BASIS,
    DEFAULT_STAGE,
    default_basis_for,
    default_stage_for,
    EDITOR_COLUMNS,
    GWP_COL,
    SCALE_KEYWORDS,
    TRACI_IMPACT_COLS,
    default_farm_config as _default_farm_config,
    default_inventory_rows,
)

UploadedFile = Union[str, BytesIO, BinaryIO]


def default_farm_config() -> dict[str, Any]:
    return _default_farm_config()


def inventory_table_for_editor(rows: Optional[list[dict[str, Any]]] = None) -> pd.DataFrame:
    """DataFrame shaped for st.data_editor (Material, Unit, Amount, costs, TRACI, Type)."""
    data = rows if rows is not None else default_inventory_rows()
    df = pd.DataFrame(data)
    for col in EDITOR_COLUMNS:
        if col not in df.columns:
            if col == "Stage":
                df[col] = df["Material"].map(default_stage_for)
            elif col == "Basis":
                df[col] = df["Material"].map(default_basis_for)
            elif col in ("Material", "Unit", "Type", "Notes"):
                df[col] = ""
            else:
                df[col] = 0.0
    return df[EDITOR_COLUMNS].copy()


def _is_scale_name(name: str) -> bool:
    low = str(name).lower()
    return any(k in low for k in SCALE_KEYWORDS)


def infer_type_series(materials: pd.Series, existing: Optional[pd.Series] = None) -> pd.Series:
    inferred = materials.map(lambda m: "Scale" if _is_scale_name(m) else "Efficiency")
    if existing is None:
        return inferred
    out = existing.copy()
    blank = out.isna() | (out.astype(str).str.strip() == "") | (out.astype(str).str.lower() == "nan")
    out[blank] = inferred[blank]
    return out


def baseline_trees(farm_config: dict[str, Any]) -> float:
    acres = float(farm_config.get("acres", 1.0))
    harvested = float(farm_config.get("harvested_per_acre", 1900))
    return acres * harvested


def scale_inventory_amounts(inventory_df: pd.DataFrame, farm_config: dict[str, Any]) -> np.ndarray:
    """Return amounts in the same units as the optimizer baseline.

    per_acre_rotation: Amount column is per acre → multiply by acres.
    workbook_total: Amount column is already the modeled total (Excel path).
    """
    amounts = pd.to_numeric(inventory_df["Amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    basis = farm_config.get("amount_basis", "per_acre_rotation")
    if basis == "workbook_total":
        return amounts
    acres = float(farm_config.get("acres", 1.0))
    return amounts * acres


def traci_columns_in(df: pd.DataFrame) -> list[str]:
    skip = {"Material", "Unit", "Year", "Amount", "Unit Cost ($)", "Type", "Notes"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if c == GWP_COL or c in TRACI_IMPACT_COLS:
            cols.append(c)
            continue
        # Excel impact sheet often has extra TRACI headers; keep numeric extras.
        if df[c].dtype.kind in "fc":
            cols.append(c)
    # GWP first if present.
    if GWP_COL in cols:
        cols = [GWP_COL] + [c for c in cols if c != GWP_COL]
    return cols


def build_optimizer_arrays(
    farm_config: dict[str, Any],
    inventory_df: pd.DataFrame,
) -> dict[str, Any]:
    """materials, base_amounts, costs, impact_matrix, masks, baseline_trees."""
    df = inventory_df.copy()
    df = df[df["Material"].notna() & (df["Material"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("Inventory is empty. Add at least one material on Farm Setup.")

    if "Type" not in df.columns:
        df["Type"] = infer_type_series(df["Material"])
    else:
        df["Type"] = infer_type_series(df["Material"], df["Type"])

    for col, default in (("Stage", DEFAULT_STAGE), ("Basis", DEFAULT_BASIS)):
        if col not in df.columns:
            df[col] = df["Material"].map(
                default_stage_for if col == "Stage" else default_basis_for
            )
        else:
            blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
            filler = default_stage_for if col == "Stage" else default_basis_for
            df.loc[blank, col] = df.loc[blank, "Material"].map(filler)

    materials = df["Material"].astype(str).tolist()
    units = df["Unit"].astype(str).tolist() if "Unit" in df.columns else [""] * len(df)
    base_amounts = scale_inventory_amounts(df, farm_config)
    costs = pd.to_numeric(df.get("Unit Cost ($)", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    impact_cols = [c for c in traci_columns_in(df) if c in df.columns]
    if not impact_cols:
        df[GWP_COL] = 0.0
        impact_cols = [GWP_COL]
    impact_matrix = df[impact_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    types = df["Type"].astype(str).str.strip().str.title()
    scale_mask = (types == "Scale").to_numpy(dtype=bool, copy=True)
    efficiency_mask = (types == "Efficiency").to_numpy(dtype=bool, copy=True)

    trees = baseline_trees(farm_config)
    return {
        "materials": materials,
        "units": units,
        "base_amounts": base_amounts,
        "costs": costs,
        "impact_matrix": impact_matrix,
        "traci_impact_cols": impact_cols,
        "scale_mask": scale_mask,
        "efficiency_mask": efficiency_mask,
        "stages": df["Stage"].astype(str).tolist(),
        "bases": df["Basis"].astype(str).tolist(),
        "baseline_trees": float(trees),
        "farm_config": dict(farm_config),
        "inventory_df": df,
    }


def load_deap_excel(uploaded_file: UploadedFile) -> dict[str, Any]:
    """Same merge as the original Streamlit app: sheet0 inputs, sheet1 cost, sheet2 impact.

    Joins on Material + Unit. Amount comes from the inputs sheet. Missing values → 0.
    """
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    names = list(sheets.keys())
    if len(names) < 3:
        raise ValueError(
            f"Expected at least 3 sheets (inputs, cost, impact); found {len(names)}: {names}"
        )
    inputs_df = sheets[names[0]]
    cost_df = sheets[names[1]]
    impact_df = sheets[names[2]]

    for required in ("Material", "Unit"):
        for label, frame in (("inputs", inputs_df), ("cost", cost_df), ("impact", impact_df)):
            if required not in frame.columns:
                raise ValueError(f"{label} sheet is missing column '{required}'")

    merged = inputs_df.merge(cost_df, on=["Material", "Unit"], how="left")
    merged = merged.merge(impact_df, on=["Material", "Unit"], how="left")
    merged = merged.fillna(0)

    if "Amount" not in merged.columns:
        raise ValueError("Inputs sheet must include an 'Amount' column.")
    if "Unit Cost ($)" not in merged.columns:
        # tolerate a few header aliases
        for alt in ("Unit Cost", "UnitCost", "Cost", "unit cost ($)"):
            if alt in merged.columns:
                merged = merged.rename(columns={alt: "Unit Cost ($)"})
                break
        else:
            merged["Unit Cost ($)"] = 0.0

    impact_columns = [
        c
        for c in impact_df.columns
        if c in merged.columns and c not in ("Material", "Unit", "Year")
    ]
    if GWP_COL not in merged.columns:
        merged[GWP_COL] = 0.0
        if GWP_COL not in impact_columns:
            impact_columns = [GWP_COL] + impact_columns

    merged["Type"] = infer_type_series(merged["Material"])
    if "Notes" not in merged.columns:
        merged["Notes"] = ""

    editor = inventory_table_for_editor()
    for col in editor.columns:
        if col not in merged.columns:
            merged[col] = "" if col in ("Type", "Notes", "Material", "Unit") else 0.0
    keep = list(EDITOR_COLUMNS)
    for c in impact_columns:
        if c not in keep:
            keep.append(c)
    keep = [c for c in keep if c in merged.columns]
    inventory = merged[keep].copy()

    farm = default_farm_config()
    farm["amount_basis"] = "workbook_total"
    return {
        "merged_df": merged,
        "inventory_df": inventory,
        "farm_config": farm,
        "traci_impact_cols": impact_columns,
        "sheet_names": names,
    }


def apply_factors_csv(inventory_df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge unit costs and TRACI factors onto the current inventory (Material+Unit)."""
    out = inventory_df.copy()
    key = ["Material", "Unit"]
    for k in key:
        if k not in factors_df.columns:
            raise ValueError(f"Factors CSV must include '{k}'")
    factor_cols = [c for c in factors_df.columns if c not in key]
    slim = factors_df[key + factor_cols].copy()
    merged = out.merge(slim, on=key, how="left", suffixes=("", "_new"))
    for c in factor_cols:
        new_c = f"{c}_new" if f"{c}_new" in merged.columns else c
        if new_c not in merged.columns:
            continue
        if c in merged.columns and new_c != c:
            merged[c] = merged[new_c].combine_first(merged[c])
            merged = merged.drop(columns=[new_c])
        elif c not in out.columns:
            merged[c] = merged[new_c] if new_c != c else merged[c]
    keep = [c for c in EDITOR_COLUMNS if c in merged.columns]
    extra = [c for c in merged.columns if c not in keep and c not in key]
    return merged[keep + extra]


def baseline_totals(arrays: dict[str, Any]) -> dict[str, float]:
    amounts = np.asarray(arrays["base_amounts"], dtype=float)
    costs = np.asarray(arrays["costs"], dtype=float)
    total_cost = float(np.dot(amounts, costs))
    gwp = 0.0
    cols = arrays["traci_impact_cols"]
    matrix = np.asarray(arrays["impact_matrix"], dtype=float)
    if GWP_COL in cols:
        gwp = float(np.dot(amounts, matrix[:, cols.index(GWP_COL)]))
    trees = max(float(arrays["baseline_trees"]), 1e-9)
    return {
        "total_cost": total_cost,
        "total_gwp": gwp,
        "cost_per_tree": total_cost / trees,
        "gwp_per_tree": gwp / trees,
        "trees": trees,
    }
