"""Direct field emissions from fertiliser and lime.

Upstream vs direct, and why only some inputs need this
-------------------------------------------------------
ecoinvent activities come in two shapes and the distinction decides everything:

  * PRODUCTION datasets ("market for NPK") stop at the factory gate. What the
    product does once spread on a field is foreground emission that no
    background dataset supplies. These need the equations below.

  * TREATMENT datasets ("diesel, burned in agricultural machinery") already
    include the emission to air -- that is the point of the activity. Adding a
    combustion layer on top DOUBLE COUNTS. The app's diesel factor of 0.1335 kg
    CO2-eq/MJ is ~7x a production-only factor and ~56% tailpipe CO2, confirming
    it is combustion-inclusive. So diesel gets nothing from this module.

Every inventory row therefore carries a stage flag, and this engine fires only
on production-only rows. The failure mode is silent: nobody notices a double
count, they just report a number that is too high.

Scale of the correction
-----------------------
Fertiliser field emissions are ~46% of fertiliser GWP, and on acidification
they dominate: field NH3 at the aggregate Frac_GASF gives 0.0428 kg SO2-eq per
tree against 0.0084 upstream. The manuscript's whole cradle-to-grave
acidification is 0.0309, which is less than field NH3 alone -- so that figure
does not currently include these emissions.

Method
------
IPCC 2019 Refinement to the 2006 Guidelines, Volume 4, Chapter 11. All default
factors are from Tables 11.1 and 11.3 and are editable.

EF1 treatment follows Table 11.1 footnote 5 exactly: the synthetic value (0.016
in wet climates) applies to synthetic fertiliser AND to "fertiliser mixtures
that include both synthetic and organic forms of N". The organic value (0.006)
comes from trials applying organic amendments ALONE -- Annex 11A.2 Table 2A.2,
Wet O, n=110 -- and must not be split out of a blend. Annex 11A.2 confirms the
grouping was tested: synthetic and mixed were statistically indistinguishable,
organic alone was significantly lower.

Frac_GASF is fertiliser-specific (Table 11.3) and spans fifteen-fold from
nitrate-based (0.01) to urea (0.15). Per Table 7A.2, IPCC treats NPK compound
fertilisers as a 50/50 ammonium-nitrate / CAN mix, i.e. AN-based at 0.05 -- not
the 0.11 aggregate. This single parameter moves acidification more than any
farm decision in the model, which is what makes product choice a real
substitution built entirely from published defaults.

GWP for N2O is 298 (AR4), matching TRACI 2.1. Do not switch to the AR6 value
without re-characterising the whole LCA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Climate = Literal["wet", "dry"]

# --- IPCC Table 11.1 / 11.3 defaults -------------------------------------

EF1_SYNTHETIC_AND_MIXED_WET = 0.016
EF1_ORGANIC_ONLY_WET = 0.006
EF1_ALL_DRY = 0.005
EF4_WET = 0.014
EF4_DRY = 0.005
EF5 = 0.011
FRAC_GASM = 0.21
FRAC_LEACH_WET = 0.24
FRAC_LEACH_DRY = 0.0

# Table 11.3 disaggregation of Frac_GASF by fertiliser chemistry
FRAC_GASF_BY_FORM = {
    "aggregate": 0.11,
    "urea": 0.15,
    "ammonium": 0.08,
    "ammonium_nitrate": 0.05,
    "nitrate": 0.01,
}

# Stoichiometry
N2O_N_TO_N2O = 44.0 / 28.0
NH3_N_TO_NH3 = 17.0 / 14.0
NO3_N_TO_NO3 = 62.0 / 14.0
GWP_N2O = 298.0

# IPCC Ch.11 s.11.3 -- CO2 from liming, t C per t carbonate
EF_LIME_LIMESTONE = 0.12
EF_LIME_DOLOMITE = 0.13
C_TO_CO2 = 44.0 / 12.0


@dataclass(frozen=True)
class FertiliserProduct:
    """A fertiliser as a grower buys it.

    ``form`` selects Frac_GASF per Table 11.3. For an NPK compound use
    "ammonium_nitrate" (Table 7A.2), not "aggregate".

    ``pan_fraction`` is the plant-available share of total N, used only for
    organics. Broadcast poultry litter is ~0.50, so replacing a given amount of
    available N requires twice that in total N -- and IPCC applies EF1 to TOTAL
    N, not available N. That is why organic scenarios carry more total nitrogen
    than the synthetic baseline they replace.
    """

    name: str
    n_fraction: float
    is_organic: bool = False
    form: str = "aggregate"
    pan_fraction: float = 1.0
    cost_per_kg: Optional[float] = None
    note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.n_fraction <= 1.0:
            raise ValueError("n_fraction must be a mass fraction in [0, 1]")
        if not 0.0 < self.pan_fraction <= 1.0:
            raise ValueError("pan_fraction must be in (0, 1]")
        if self.form not in FRAC_GASF_BY_FORM:
            raise ValueError(
                f"unknown form {self.form!r}; choose from {sorted(FRAC_GASF_BY_FORM)}"
            )

    @property
    def frac_gasf(self) -> float:
        return FRAC_GASF_BY_FORM[self.form]

    def total_n(self, product_kg: float) -> float:
        return float(product_kg) * self.n_fraction

    def available_n(self, product_kg: float) -> float:
        return self.total_n(product_kg) * self.pan_fraction

    def product_for_available_n(self, target_available_n: float) -> float:
        """Product mass needed to supply a given amount of plant-available N."""
        denom = self.n_fraction * self.pan_fraction
        if denom <= 0:
            raise ValueError(f"{self.name} supplies no available nitrogen")
        return float(target_available_n) / denom


def default_products() -> list[FertiliserProduct]:
    return [
        FertiliserProduct(
            name="NPK 19-19-19 (baseline)",
            n_fraction=0.19,
            form="ammonium_nitrate",
            note="NPK compound; Table 7A.2 treats these as AN/CAN, Frac_GASF 0.05.",
        ),
        FertiliserProduct(
            name="Urea (46-0-0)",
            n_fraction=0.46,
            form="urea",
            note="Cheapest N per kg, but Frac_GASF 0.15 -- highest ammonia loss.",
        ),
        FertiliserProduct(
            name="Ammonium sulphate (21-0-0)",
            n_fraction=0.21,
            form="ammonium",
        ),
        FertiliserProduct(
            name="Calcium nitrate (15.5-0-0)",
            n_fraction=0.155,
            form="nitrate",
            note="Frac_GASF 0.01 -- lowest volatilisation, highest price per kg N.",
        ),
        FertiliserProduct(
            name="Poultry litter (broadcast)",
            n_fraction=0.0289,
            is_organic=True,
            form="aggregate",
            pan_fraction=0.50,
            note=(
                "57.8 lb total N/short ton, 50% broadcast availability "
                "(NC State Extension). n_fraction is total N as a mass fraction."
            ),
        ),
    ]


@dataclass
class DirectEmissions:
    """Field emissions for one stand, before allocation or per-tree division."""

    f_sn: float
    f_on: float
    n2o_direct: float
    n2o_volatilisation: float
    n2o_leaching: float
    nh3: float
    nitrate: float
    lime_co2: float = 0.0
    ef1_used: float = 0.0
    frac_gasf_used: float = 0.0

    @property
    def n2o_total(self) -> float:
        return self.n2o_direct + self.n2o_volatilisation + self.n2o_leaching

    @property
    def gwp(self) -> float:
        """kg CO2-eq from N2O plus liming."""
        return self.n2o_total * GWP_N2O + self.lime_co2

    def scaled(self, factor: float) -> "DirectEmissions":
        """Apply an allocation factor or per-tree division to every flow."""
        f = float(factor)
        return DirectEmissions(
            f_sn=self.f_sn * f,
            f_on=self.f_on * f,
            n2o_direct=self.n2o_direct * f,
            n2o_volatilisation=self.n2o_volatilisation * f,
            n2o_leaching=self.n2o_leaching * f,
            nh3=self.nh3 * f,
            nitrate=self.nitrate * f,
            lime_co2=self.lime_co2 * f,
            ef1_used=self.ef1_used,
            frac_gasf_used=self.frac_gasf_used,
        )


def direct_emissions(
    synthetic_n: float,
    organic_n: float = 0.0,
    frac_gasf: float = FRAC_GASF_BY_FORM["aggregate"],
    climate: Climate = "wet",
    lime_kg: float = 0.0,
    lime_ef: float = EF_LIME_LIMESTONE,
    ef1_override: Optional[float] = None,
) -> DirectEmissions:
    """IPCC Tier 1 direct and indirect field emissions.

    ``synthetic_n`` and ``organic_n`` are kg of TOTAL N applied, not
    plant-available N -- IPCC applies EF1 to total N (Ch.11 footnote 11 notes
    that N inputs are deliberately NOT adjusted for volatilisation).

    EF1 selection follows Table 11.1 footnote 5: the synthetic factor covers
    synthetic and mixed applications; the organic factor applies only when the
    application is exclusively organic.
    """
    f_sn = max(float(synthetic_n), 0.0)
    f_on = max(float(organic_n), 0.0)

    if ef1_override is not None:
        ef1 = float(ef1_override)
    elif climate == "dry":
        ef1 = EF1_ALL_DRY
    elif f_sn > 0:
        # Synthetic or mixed -- footnote 5.
        ef1 = EF1_SYNTHETIC_AND_MIXED_WET
    else:
        ef1 = EF1_ORGANIC_ONLY_WET

    ef4 = EF4_WET if climate == "wet" else EF4_DRY
    frac_leach = FRAC_LEACH_WET if climate == "wet" else FRAC_LEACH_DRY

    total_n = f_sn + f_on
    volatilised_n = f_sn * frac_gasf + f_on * FRAC_GASM
    leached_n = total_n * frac_leach

    return DirectEmissions(
        f_sn=f_sn,
        f_on=f_on,
        n2o_direct=total_n * ef1 * N2O_N_TO_N2O,
        n2o_volatilisation=volatilised_n * ef4 * N2O_N_TO_N2O,
        n2o_leaching=leached_n * EF5 * N2O_N_TO_N2O,
        nh3=volatilised_n * NH3_N_TO_NH3,
        nitrate=leached_n * NO3_N_TO_NO3,
        lime_co2=float(lime_kg) * float(lime_ef) * C_TO_CO2,
        ef1_used=ef1,
        frac_gasf_used=frac_gasf,
    )


def emissions_for_plan(
    products_and_masses: list[tuple[FertiliserProduct, float]],
    climate: Climate = "wet",
    lime_kg: float = 0.0,
) -> DirectEmissions:
    """Direct emissions for a mix of products, each given as (product, kg).

    Frac_GASF is mass-weighted across the synthetic products by their nitrogen
    contribution, so switching products changes ammonia loss as Table 11.3
    intends.
    """
    syn_n = 0.0
    org_n = 0.0
    weighted_gasf = 0.0
    for product, mass in products_and_masses:
        n = product.total_n(mass)
        if product.is_organic:
            org_n += n
        else:
            syn_n += n
            weighted_gasf += n * product.frac_gasf
    gasf = weighted_gasf / syn_n if syn_n > 0 else FRAC_GASF_BY_FORM["aggregate"]
    return direct_emissions(
        synthetic_n=syn_n,
        organic_n=org_n,
        frac_gasf=gasf,
        climate=climate,
        lime_kg=lime_kg,
    )


# TRACI 2.1 characterisation of the direct flows, derived from the baseline
# openLCA process. N2O climate change is the AR4 GWP used by TRACI 2.1.
DIRECT_CHARACTERISATION = {
    "nh3": {
        "kg SO2-Eq/Unit": 1.880158,
        "kg PM2.5-Eq/Unit": 0.0667252,
    },
    "nitrate": {
        "kg N-Eq/Unit": 0.251614,
        "CTUe/Unit": 0.00689579,
    },
    "n2o": {
        "kg CO2-Eq/Unit": GWP_N2O,
    },
}


def characterised_direct(emissions: DirectEmissions) -> dict[str, float]:
    """Map direct flows onto TRACI categories.

    No direct field contributions are assigned to human toxicity, ozone
    depletion or photochemical oxidant formation -- those categories are driven
    by upstream supply chains and combustion, not by the N flows above.
    """
    out: dict[str, float] = {}
    for category, factor in DIRECT_CHARACTERISATION["nh3"].items():
        out[category] = out.get(category, 0.0) + emissions.nh3 * factor
    for category, factor in DIRECT_CHARACTERISATION["nitrate"].items():
        out[category] = out.get(category, 0.0) + emissions.nitrate * factor
    out["kg CO2-Eq/Unit"] = out.get("kg CO2-Eq/Unit", 0.0) + emissions.gwp
    return out


# --- reporting basis for pesticides --------------------------------------
#
# Pesticide labels state active ingredient as a SALT; ecoinvent pesticide
# datasets are per kg of the ACID. Multiplying a salt-basis mass into an
# acid-basis dataset overstates the burden -- 18% for the potassium salt of
# glyphosate, 26% for the isopropylamine salt.
#
# These are molar-mass ratios of acid to salt. Confirm which salt your label
# states before applying one; the factor is not interchangeable between salts.
SALT_TO_ACID = {
    "glyphosate_potassium": 0.82,
    "glyphosate_isopropylamine": 0.74,
    "glyphosate_acid": 1.00,
    "2,4-D_butoxyethyl_ester": 0.68,
    "triclopyr_butoxyethyl_ester": 0.67,
}


def to_acid_equivalent(mass: float, salt: str) -> float:
    """Convert a salt-basis active-ingredient mass to acid equivalent."""
    if salt not in SALT_TO_ACID:
        raise ValueError(
            f"unknown salt {salt!r}; choose from {sorted(SALT_TO_ACID)}"
        )
    return float(mass) * SALT_TO_ACID[salt]


def product_to_active_ingredient(
    volume_fl_oz: float, concentration: float, density_kg_per_l: float = 1.17
) -> float:
    """Fluid ounces of formulated product to kg of active ingredient.

    ``concentration`` is the label percentage as a fraction (0.487 for a 48.7%
    product). ``density_kg_per_l`` defaults to a typical glyphosate concentrate.

    Worked example from this project: 1,386 fl oz/acre/rotation at 47.4%
    reproduces the 22.725 kg in the LCI workbook, confirming that figure is
    active ingredient rather than product. Record the conversion -- otherwise it
    exists only as the ratio between two numbers in two different workbooks.
    """
    litres = float(volume_fl_oz) * 0.0295735
    return litres * float(density_kg_per_l) * float(concentration)
