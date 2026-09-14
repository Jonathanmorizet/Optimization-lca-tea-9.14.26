# Implementation guide

## 1. Install the files

Copy into your repo, preserving paths. New files: `lib/carbon.py`,
`lib/fertilizer.py`, `lib/substitutions.py`, `scripts/build_public_factors.py`,
everything under `tests/`. Modified: `lib/optimize.py`, `lib/defaults.py`,
`lib/data_model.py`, `lib/__init__.py`, `pages/2_Optimize.py`, `.gitignore`.

```bash
pip install pytest
python -m pytest tests/ -v      # expect 51 passing
streamlit run Home.py
```

If tests fail before you have changed anything, the copy is incomplete —
`lib/defaults.py` and `lib/data_model.py` must both be updated or the new
`Stage` and `Basis` columns will not exist.

## 2. Load your real factors

The app ships with placeholder TRACI values. Until you replace them its numbers
will not match the manuscript.

1. In openLCA, build a one-unit process per material, run TRACI 2.1, export the
   nine indicator results.
2. Fill `data/factors_internal_TEMPLATE.csv`, save it as
   `data/factors_internal.csv` (gitignored).
3. `python scripts/build_public_factors.py`
4. Load `data/factors_public.csv` in Farm Setup → Load factors CSV.

Before exporting, confirm your openLCA database is bound to the **cut-off**
system model. Databases are installed per system model; if the project was set
up as APOS the export will not be cut-off regardless of what the template says.

## 3. Set the Stage column correctly

This is the single most important thing to get right, because the failure mode
is silent. A double count produces no error — the number is just too high.

| Stage | Meaning | Direct engine |
|---|---|---|
| `production` | Factor is cradle-to-gate (market for NPK, lime) | **fires** |
| `combustion` | Factor already includes emission to air (diesel, burned in agricultural machinery) | never fires |
| `none` | No direct pathway modelled | never fires |

Defaults are assigned by material name and are almost certainly right for the
shipped inventory. Check them anyway after loading your own data.

**How to tell which diesel dataset you used.** A combustion-inclusive factor is
around 0.13 kg CO2-eq/MJ; a production-only factor is around 0.02. If yours is
near 0.02 you used `market for diesel`, combustion is missing entirely, and the
row should be `production` with a combustion layer added — the opposite of the
usual correction.

## 4. Set the Basis column for pesticides

Pesticide labels state active ingredient as a **salt**. ecoinvent pesticide
datasets are per kg of the **acid**. Feeding a salt-basis mass into an
acid-basis dataset overstates the burden by 18% (potassium salt of glyphosate)
or 26% (isopropylamine salt).

```python
from lib.fertilizer import product_to_active_ingredient, to_acid_equivalent

# 1,386 fl oz/acre/rotation of a 47.4% concentrate
ai = product_to_active_ingredient(1386, 0.474)     # 22.73 kg a.i.
ae = to_acid_equivalent(ai, "glyphosate_potassium")  # 18.64 kg acid equivalent
```

Two things to check on the actual product label: whether the percentage is
stated as the salt or as acid equivalent, and which salt it is. Then confirm
the basis of the ecoinvent activity you selected. Same question applies to
Crossbow (2,4-D and triclopyr butoxyethyl esters).

This matters beyond bookkeeping: glyphosate is the second-worst nitrogen burden
per dollar in your factor table, so an overstated herbicide figure makes mowing
look better than it is.

## 5. Run the optimizer

Choose **Compare practice options** or **Show cost vs carbon tradeoffs**, leave
"Credit carbon stored in the harvested trees" on, and set the tree floor.

Three settings decide whether the results mean anything.

**Optimise per acre, report per tree.** Sequestration is 18.3 kg CO2 per 7-ft
tree by definition of the functional unit. On a per-tree basis it is an
additive constant identical for every plan and cannot change a ranking; on a
per-acre basis it multiplies tree count and dominates. The code enforces this
with an explicit `basis` argument — do not work around it.

**Set a real production-scale ceiling.** The optimizer now buys carbon by
growing more trees, indefinitely. The ±20% slider is arbitrary. Bound it by
your actual spacing, acreage, and what the market absorbs, or the
lowest-carbon plan is just "plant more" every time.

**Results are farm gate.** Cradle-to-gate is −15.31 kg CO2-eq/tree;
cradle-to-grave is +6.18. The sign flips on consumer pickup and end-of-life,
neither of which a grower controls. Never optimise against cradle-to-grave.

## 6. Direct field emissions

```python
from lib.fertilizer import direct_emissions, emissions_for_plan, default_products

e = direct_emissions(synthetic_n=324.05, frac_gasf=0.05, climate="wet")
per_tree = e.scaled(1 / 1900)
```

Two method points that will come up in review.

**EF1 for mixed applications.** IPCC 2019 Table 11.1 footnote 5: the synthetic
value (0.016 wet) applies to synthetic **and** mixed synthetic-organic
applications. The 0.006 organic value comes from trials applying organic
amendments alone (Annex 11A.2, Wet O, n=110) and must not be split out of a
blend. The code does this automatically.

**Frac_GASF is fertiliser-specific** and spans fifteen-fold: urea 0.15,
ammonium 0.08, ammonium-nitrate 0.05, nitrate 0.01. Table 7A.2 treats NPK
compounds as a 50/50 AN/CAN mix, so 19-19-19 defaults to **0.05**, not the
0.11 aggregate. The accompanying workbook uses 0.11 as specified; the module
uses 0.05. Pick one and state it — the choice roughly halves acidification, and
field NH3 already exceeds your published cradle-to-grave acidification total.

## 7. Before anything is published

- Email ecoinvent describing the exact use — aggregated per-material LCIA
  results embedded in an academic tool — and keep the reply on file. Git history
  is hard to scrub.
- Replace the placeholder numbers in `lib/substitutions.py`: the mow-dominant
  ratios (herbicide 0.35, diesel 1.85, labour 1.30) and the $2.50/transplant
  grafted premium. `unsourced_options()` surfaces them in the UI.
- Reconcile the two farm descriptions. Enterprise budget: 4×4.5 ft, 2,400
  planted, 2,040 harvested, 85% survival. Manuscript: 4.5×4.5 ft, 2,150
  planted, 1,900 harvested, 88.4%. Pick one basis and rescale the other.
- Reconcile the acre-year normalisation: −15.31/tree × 1,900 = −29,089 per
  acre-rotation, but Table S14 reports −3,111 per acre-year, which over 8 years
  is −24,888. The ratio implies 9.35 years.
- Add plantation-stage liming if it happens. It is absent from both the
  inventory and the enterprise budget, and it is both an upstream input and a
  direct CO2 emission (IPCC Ch.11 s.11.3).

## Still not built

**NPV and discounting.** The last Tier 1 item. The enterprise budget already
has year-by-year cash flows and a Required Rate of Return field sitting at 0.
Until this exists, `rotation_years` still does not enter any calculation.

**Wreath allocation beyond carbon.** `lib/carbon.py` applies the 0.893 factor
to GWP. Cost and the other TRACI categories are still unallocated, so they are
whole-plantation figures while carbon is tree-only.

## What changed in the engine, and why

Bounds used to be enforced by a fixed penalty of 10,000 plus clipping inside
the mutation operator. `cxBlend` extrapolates beyond its parents, so crossover
children could leave the box and were held in only by that penalty. Once
sequestration entered the objective the penalty became cheap relative to the
carbon "gained" by inflating production scale, and the search escaped entirely
— scale reached 63× baseline.

Bounds are now hard, via repair after crossover and mutation
(`optimize._repair_factory`). A penalty cannot bound a variable that appears in
a large negative term.

One consequence worth knowing: the substitution-only front is narrower than it
first appeared. Earlier runs reported ~22 non-dominated plans with 22% cost
spread; with hard bounds it is ~9 plans and 1.3%. The extra width was
infeasible individuals. Substitution is a real but thin tradeoff. Biogenic
sequestration is the load-bearing fix — it alone produces ~120 plans with 33%
cost spread, cheapest at scale 0.80 and lowest-carbon at 1.20.
