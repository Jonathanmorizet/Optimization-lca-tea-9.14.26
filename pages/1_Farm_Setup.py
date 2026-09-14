"""Farm configuration and inventory editor."""

from io import BytesIO

import pandas as pd
import streamlit as st

from lib.data_model import (
    apply_factors_csv, default_farm_config, inventory_table_for_editor, load_deap_excel,
)

st.set_page_config(page_title="Farm Setup", page_icon="🚜", layout="wide")
st.title("Farm Setup")

if "farm_config" not in st.session_state:
    st.session_state.farm_config = default_farm_config()
if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = inventory_table_for_editor()

farm = dict(st.session_state.farm_config)
c1, c2, c3 = st.columns(3)
farm["acres"] = c1.number_input("Acres", min_value=0.01, value=float(farm["acres"]))
farm["rotation_years"] = c2.number_input("Rotation (years)", min_value=1, value=int(farm["rotation_years"]))
farm["harvested_per_acre"] = c3.number_input(
    "Harvested trees per acre", min_value=1, value=int(farm["harvested_per_acre"])
)

uploaded = st.file_uploader("Load a three-sheet DEAP workbook", type=["xlsx", "xlsm"])
if uploaded is not None:
    try:
        loaded = load_deap_excel(BytesIO(uploaded.getvalue()))
        st.session_state.inventory_df = loaded["inventory_df"]
        farm.update(loaded["farm_config"])
        st.session_state.source = "excel"
        st.success("Workbook loaded.")
    except Exception as exc:
        st.error(f"Could not load workbook: {exc}")

factors = st.file_uploader("Load factors CSV", type=["csv"], key="factor_csv")
if factors is not None:
    try:
        factor_df = pd.read_csv(BytesIO(factors.getvalue()))
        st.session_state.inventory_df = apply_factors_csv(st.session_state.inventory_df, factor_df)
        st.session_state.source = "factor CSV"
        st.success("Factors loaded for this session.")
    except Exception as exc:
        st.error(f"Could not load factors: {exc}")

edited = st.data_editor(
    st.session_state.inventory_df, num_rows="dynamic",
    use_container_width=True, hide_index=True,
)

if st.button("Save farm inventory", type="primary"):
    st.session_state.farm_config = farm
    st.session_state.inventory_df = edited
    st.session_state.opt_ready = True
    st.session_state.setdefault("source", "starter inventory")
    st.success("Farm inventory saved. You can now open Optimize.")
    st.page_link("pages/2_Optimize.py", label="Go to Optimize", icon="📈")
