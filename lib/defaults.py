"""NC Fraser fir plantation defaults for the farm workbook.

Amounts are **per acre, per rotation** (baseline 8 years) unless noted.
Density: 2,150 planted / acre, ~88.5% survival, 1,900 harvested / acre.

Unit costs are grower-facing NC enterprise-budget *examples* (USD, ~2024–26),
editable in the Farm Setup page — not a published Extension circular.

TRACI columns are **placeholders** (order-of-magnitude, not ecoinvent).
Replace via data/sample_factors.csv or a DEAP .xlsm upload once real
ecoinvent/TRACI factors are linked. The impact column name
``kg CO2-Eq/Unit`` matches the user's original optimizer.
"""

from __future__ import annotations

from typing import Any

# Impact column used by evaluate_* functions (do not rename without updating optimize.py).
GWP_COL = "kg CO2-Eq/Unit"

# Extra TRACI stubs so "minimize one environmental impact" has more than GWP.
ACID_COL = "kg SO2-Eq/Unit"
EUTRO_COL = "kg N-Eq/Unit"
ECOTOX_COL = "CTUe/Unit"

TRACI_IMPACT_COLS = [GWP_COL, ACID_COL, EUTRO_COL, ECOTOX_COL]

# Materials whose amounts scale with harvested-tree count (production scale gene).
SCALE_KEYWORDS = ("transplant", "seedling", "plant", "tree")

# How a row's impact factor was sourced. This decides whether the direct-
# emissions engine may fire on it, and getting it wrong is silent.
#
#   "production"  cradle-to-gate only (e.g. market for NPK). The factor stops
#                 at the factory gate, so field emissions must be ADDED.
#   "combustion"  the activity already includes emission to air (e.g. diesel,
#                 burned in agricultural machinery). Adding a direct layer
#                 DOUBLE COUNTS.
#   "none"        no direct pathway modelled for this material.
STAGE_VALUES = ("production", "combustion", "none")
DEFAULT_STAGE = "none"

# What one unit of the Amount column physically is. Pesticide labels state
# active ingredient as a SALT, while ecoinvent pesticide datasets are per kg of
# the ACID. Feeding a salt-basis mass into an acid-basis dataset overstates the
# burden by 18-26% for glyphosate.
BASIS_VALUES = ("as_supplied", "product", "active_ingredient", "acid_equivalent")
DEFAULT_BASIS = "as_supplied"

EDITOR_COLUMNS = [
    "Material",
    "Unit",
    "Amount",
    "Unit Cost ($)",
    GWP_COL,
    ACID_COL,
    EUTRO_COL,
    ECOTOX_COL,
    "Type",  # Scale | Efficiency | Fixed
    "Stage",  # production | combustion | none
    "Basis",  # as_supplied | product | active_ingredient | acid_equivalent
    "Notes",
]


def default_farm_config() -> dict[str, Any]:
    """Farm-level settings. harvested_per_acre is an override of planted × survival."""
    planted = 2150
    survival_pct = 88.5
    return {
        "acres": 1.0,
        "rotation_years": 8,
        "planted_per_acre": planted,
        "survival_pct": survival_pct,
        "harvested_per_acre": 1900,  # manuscript / OpenLCA plantation output
        "functional_unit": "acre_rotation",  # or "harvested_tree"
        # Amounts in the inventory table are per acre per rotation unless Excel loaded.
        "amount_basis": "per_acre_rotation",
    }


def auto_harvested(planted_per_acre: float, survival_pct: float) -> float:
    return planted_per_acre * (survival_pct / 100.0)


# --- Starter inventory (Table S5 / S6 plantation roll-up, 8-yr rotation) -----
# GWP stubs: order-of-magnitude so NPK and diesel remain the usual hotspots.
# Acidification / eutrophication / ecotoxicity are rough ratios of GWP, not TRACI.
_INV = [
    # material, unit, amount/acre/rotation, unit_cost, gwp, acid, eutro, eco, type, notes
    (
        "Transplants (initial)",
        "transplants",
        2150.0,
        1.75,
        0.50,
        0.0015,
        0.0008,
        2.0,
        "Scale",
        "Plug+2; scales with trees planted",
    ),
    (
        "Transplants (replant)",
        "transplants",
        215.0,
        1.75,
        0.50,
        0.0015,
        0.0008,
        2.0,
        "Scale",
        "10% replacement in year 2",
    ),
    (
        "Fertilizer (19-19-19)",
        "lb",
        4210.56,
        0.55,
        0.76,
        0.0040,
        0.0035,
        8.0,
        "Efficiency",
        "8 × 526.32 lb/acre; ecoinvent proxy NPK 15-15-15",
    ),
    (
        "Roundup (Glyphosate)",
        "oz",
        1386.0,
        0.22,
        0.10,
        0.0004,
        0.0009,
        1.5,
        "Efficiency",
        "154 oz/acre × 9 years (0–8)",
    ),
    (
        "Roundup – Low Dose (Mow)",
        "oz",
        216.0,
        0.22,
        0.10,
        0.0004,
        0.0009,
        1.5,
        "Efficiency",
        "24 oz/acre × 9 years",
    ),
    (
        "Crossbow (2,4-D + triclopyr)",
        "oz",
        1152.0,
        0.35,
        0.12,
        0.0005,
        0.0007,
        2.0,
        "Efficiency",
        "128 oz/acre × 9 years",
    ),
    (
        "Trico Pro (sheep fat 6.4%)",
        "gal",
        52.5,
        95.00,
        2.40,
        0.008,
        0.004,
        5.0,
        "Efficiency",
        "Deer browse repellent; 7.5 gal/acre × 7 years",
    ),
    (
        "TriStar 30 SG (acetamiprid)",
        "oz",
        224.0,
        2.50,
        0.18,
        0.0006,
        0.0005,
        3.0,
        "Efficiency",
        "Insecticide; unspecified pesticide proxy",
    ),
    (
        "Envidor 2SC (spirodiclofen)",
        "oz",
        148.2,
        1.80,
        0.16,
        0.0005,
        0.0004,
        2.5,
        "Efficiency",
        "Miticide",
    ),
    (
        "Sivanto 200 SL (flupyradifurone)",
        "oz",
        70.0,
        2.20,
        0.16,
        0.0005,
        0.0004,
        2.5,
        "Efficiency",
        "Insecticide",
    ),
    (
        "Sniper EC (bifenthrin)",
        "oz",
        25.6,
        0.90,
        0.20,
        0.0007,
        0.0004,
        3.5,
        "Efficiency",
        "Pyrethroid",
    ),
    (
        "12 HP Tractor Use",
        "PMH",
        12.69,
        32.00,
        0.40,
        0.002,
        0.0005,
        1.0,
        "Fixed",
        "Machine hours; diesel listed separately",
    ),
    (
        "Diesel – Tractor (Deer Mgmt)",
        "MJ",
        285.0,
        0.027,
        0.095,
        0.0006,
        0.00015,
        0.4,
        "Efficiency",
        "38.5 MJ/L diesel; burned in ag machinery",
    ),
    (
        "Chainsaw (Shearing)",
        "PMH",
        120.06,
        28.00,
        0.30,
        0.0015,
        0.0004,
        0.8,
        "Fixed",
        "Labor + saw hours; fuel below",
    ),
    (
        "Diesel Equivalent – Chainsaw Shearing",
        "MJ",
        5870.0,
        0.027,
        0.095,
        0.0006,
        0.00015,
        0.4,
        "Efficiency",
        "1.27 L/PMH × 38.5 MJ/L",
    ),
    (
        "Chainsaw (Harvest)",
        "PMH",
        44.0,
        28.00,
        0.30,
        0.0015,
        0.0004,
        0.8,
        "Fixed",
        "Year-8 harvest cutting",
    ),
    (
        "Diesel Equivalent – Chainsaw Harvest",
        "MJ",
        2151.0,
        0.027,
        0.095,
        0.0006,
        0.00015,
        0.4,
        "Efficiency",
        "Final cutting operations",
    ),
    (
        "Baling net (polypropylene)",
        "kg",
        16.34,
        4.50,
        2.10,
        0.006,
        0.001,
        4.0,
        "Scale",
        "Post-harvest packaging; scales with trees sold",
    ),
]


def default_inventory_rows() -> list[dict[str, Any]]:
    rows = []
    keys = [
        "Material",
        "Unit",
        "Amount",
        "Unit Cost ($)",
        GWP_COL,
        ACID_COL,
        EUTRO_COL,
        ECOTOX_COL,
        "Type",
        "Notes",
    ]
    for tup in _INV:
        rows.append(dict(zip(keys, tup)))
    return rows


def sample_factors_csv_rows() -> list[dict[str, Any]]:
    """Material + unit factors farmers can replace (no amounts)."""
    rows = []
    for r in default_inventory_rows():
        rows.append(
            {
                "Material": r["Material"],
                "Unit": r["Unit"],
                "Unit Cost ($)": r["Unit Cost ($)"],
                GWP_COL: r[GWP_COL],
                ACID_COL: r[ACID_COL],
                EUTRO_COL: r[EUTRO_COL],
                ECOTOX_COL: r[ECOTOX_COL],
            }
        )
    return rows

def default_stage_for(material: str) -> str:
    """Stage flag for a material name.

    Fertilizers are production-only: ecoinvent stops at the factory gate and
    field N2O, NH3 and nitrate must be added by lib.fertilizer. Diesel is a
    combustion activity that already includes the tailpipe, so nothing may be
    added. Everything else has no direct pathway modelled.
    """
    low = str(material).lower()
    if any(k in low for k in ("fertilizer", "fertiliser", "npk", "urea", "litter", "manure")):
        return "production"
    if any(k in low for k in ("lime", "limestone", "dolomit", "caco3")):
        return "production"
    if any(k in low for k in ("diesel", "gasoline", "petrol", "fuel", "propane")):
        return "combustion"
    return "none"


def default_basis_for(material: str) -> str:
    """Basis flag. Pesticides need an explicit declaration; nothing else does."""
    low = str(material).lower()
    pesticides = ("roundup", "glyphosate", "crossbow", "2,4-d", "triclopyr",
                  "tristar", "sivanto", "envidor", "sniper", "acetamiprid",
                  "bifenthrin", "spirodiclofen", "flupyradifurone")
    if any(k in low for k in pesticides):
        return "active_ingredient"
    return "as_supplied"
