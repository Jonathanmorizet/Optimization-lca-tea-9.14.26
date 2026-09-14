"""Streamlit entry point for the NC Fraser fir LCA–TEA optimizer."""

import streamlit as st

st.set_page_config(page_title="Fraser Fir LCA–TEA", page_icon="🌲", layout="wide")
st.title("NC Fraser Fir LCA–TEA Optimizer")
st.write(
    "Build a farm inventory, load licensed factors locally, and compare "
    "cost and environmental trade-offs without committing private factor data."
)
st.warning(
    "Bundled impact factors are placeholders. Do not use them for publication; "
    "load your reviewed internal factor file from Farm Setup."
)
st.page_link("pages/1_Farm_Setup.py", label="Set up the farm inventory", icon="🚜")
st.page_link("pages/2_Optimize.py", label="Run the optimizer", icon="📈")
