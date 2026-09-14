# NC Fraser Fir LCA–TEA Optimizer

A Streamlit prototype for exploring production, cost, TRACI, biogenic-carbon,
and direct field-emission trade-offs for North Carolina Fraser fir systems.

## Run locally

~~~bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
streamlit run Home.py
~~~

The bundled factors are placeholders. Keep licensed exports in
data/factors_internal.csv; that path is ignored by Git. Read
IMPLEMENTATION.md before publishing any factor table.
