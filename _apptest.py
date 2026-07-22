"""Headless smoke test: run app.py via Streamlit AppTest, assert no exceptions.
Runs all three tabs (st.tabs executes every branch). Network-dependent tabs
degrade gracefully; we only assert the app doesn't crash.
    venv\\Scripts\\python.exe _apptest.py
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py").run(timeout=120)

assert not at.exception, f"app raised: {at.exception}"
titles = [t.value for t in at.title]
assert any("Flood Forecaster" in t for t in titles), f"title missing, got {titles}"
# at least one dataframe rendered in the 'how good' tab (metrics table)
assert len(at.dataframe) >= 1, "expected metrics/table dataframes to render"
print(f"[apptest] OK - {len(titles)} title(s), {len(at.warning)} warning(s), "
      f"{len(at.dataframe)} table(s), no uncaught exceptions")
print("APP SMOKE TEST PASSED.")
