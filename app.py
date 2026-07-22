"""
Streamlit UI -- DAY-BY-DAY 7-day flood risk for USGS gauge 01543000 (v2).

Ensemble uncertainty bands, a live GloFAS operational benchmark, and event-based +
real-forecast evaluation panels. Interactive Plotly charts, a per-day drill-down
(day picker + ensemble-member histogram), an alert-strictness slider (interpolates
between the primary and high-confidence operating points and re-thresholds live), a
basin map, and a recent river-level sparkline. Everything is read from the persisted
bundles / summaries (never hardcoded); every live view degrades to a friendly
message if an API is down.

Run:  venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import live_data as ld

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

st.set_page_config(page_title="Flood Forecaster (Day to Day)", page_icon="🌊", layout="wide")
RISK_CMAP = LinearSegmentedColormap.from_list("risk", ["#1a9850", "#fee08b", "#f46d43", "#a50026"])


# --------------------------------------------------------------------------- caches
@st.cache_resource
def get_bundles():
    return ld.load_bundles()


@st.cache_data(ttl=3600)
def get_weather(lat, lon):
    w = ld.fetch_forecast_weather(lat, lon)
    return w, w.attrs.get("fetched_at")


@st.cache_data(ttl=3600)
def get_river(gauge_id):
    return ld.fetch_current_river(gauge_id)


@st.cache_data(ttl=3600)
def get_ensemble(lat, lon, model):
    return ld.fetch_ensemble_weather(lat, lon, model)


@st.cache_data(ttl=3600)
def get_glofas(lat, lon):
    return ld.fetch_glofas(lat, lon)


@st.cache_data(ttl=3600)
def get_river_history(gauge_id, days=35):
    """Recent USGS observed daily discharge for the context sparkline (None on failure).
    USGS 'daily' collection -- a live context call, not an Open-Meteo eval endpoint."""
    from datetime import date, timedelta
    end = date.today()
    df = ld.fetch_usgs_daily(gauge_id, (end - timedelta(days=days)).isoformat(), end.isoformat())
    return df if df is not None and len(df) else None


@st.cache_data
def _load_json(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def get_test_predictions():
    p = DATA / "test_predictions.csv"
    return pd.read_csv(p, parse_dates=["anchor_date", "target_date"]) if p.exists() else None


def risk_color(p, threshold):
    return RISK_CMAP(min(1.0, p / max(threshold * 2, 1e-6)))


def risk_hex(p, threshold):
    return to_hex(risk_color(p, threshold))


# ------------------------------------------------------------- add-on helpers
@st.cache_data
def train_rain_max():
    """Largest observed daily rain in the 1980-2014 training record (storm-builder cap)."""
    try:
        return float(pd.read_csv(DATA / "openmeteo_historical.csv")["precip_mm"].max())
    except Exception:
        return 120.0


@st.cache_data
def get_q_history():
    """Full 1980-2014 daily discharge (for the historical event replay context)."""
    try:
        q = pd.read_csv(DATA / "01543000_streamflow_qc.txt", sep=r"\s+", header=None,
                        names=["id", "year", "month", "day", "q_cfs", "flag"])
        q["date"] = pd.to_datetime(q[["year", "month", "day"]])
        return q[["date", "q_cfs"]]
    except Exception:
        return None


@st.cache_data
def event_alert_data():
    """Per-event fired leads + month-level warn stats at the PRIMARY operating point,
    derived from the held-out test predictions (perfect-prog -> optimistic; stated in UI).
    Nothing new is invented: thresholds come from the bundle, events from the same
    run-merging rule the evaluation used."""
    tp = get_test_predictions()
    if tp is None:
        return None
    thr = get_bundles()["A"]["thresholds"]
    fired_mask = tp["p_h"].to_numpy() >= np.array([thr[int(h)] for h in tp["horizon"]])
    fired = {(a, int(h)) for a, h, f in zip(tp["anchor_date"], tp["horizon"], fired_mask) if f}
    truth = (tp[["target_date", "flood_on_day"]].drop_duplicates("target_date")
             .set_index("target_date")["flood_on_day"])
    from evaluate_v2 import find_events  # reuse the evaluation's event definition
    rows = []
    for evd in find_events(list(truth[truth == 1].index)):
        D = evd["onset"]
        leads = [L for L in range(1, 8) if (D - pd.Timedelta(days=L), L) in fired]
        rows.append({"onset": D, "end": evd["end"], "length": evd["length"],
                     "fired_leads": leads, "caught": bool(leads),
                     "max_lead": max(leads) if leads else 0})
    evdf = pd.DataFrame(rows)
    anchors = pd.Index(sorted(tp["anchor_date"].unique()))
    warn = {a: any((a, L) in fired for L in range(1, 8)) for a in anchors}
    winf = {a: any(truth.get(a + pd.Timedelta(days=L), 0) == 1 for L in range(1, 8)) for a in anchors}
    ms = pd.DataFrame({"anchor": anchors})
    ms["warn"] = [warn[a] for a in anchors]
    ms["false"] = [bool(warn[a] and not winf[a]) for a in anchors]
    ms["month"] = ms["anchor"].dt.to_period("M").astype(str)
    month_stats = (ms.groupby("month").agg(warn_days=("warn", "sum"),
                                           false_days=("false", "sum")).to_dict("index"))
    return {"events": evdf, "month_stats": month_stats}


def plain_odds(p):
    """'about 1 in 8 — days like this flooded about 12 times out of 100 historically'."""
    if p < 0.005:
        return "well under 1 in 100 — days like this almost never flooded historically"
    n = max(1, round(1 / p))
    return (f"about 1 in {n} — days like this flooded about {max(1, round(p*100))} "
            f"time{'s' if round(p*100) != 1 else ''} out of 100 historically")


# --------------------------------------------------------------------------- load
bundles = get_bundles()
basin = bundles["A"]["basin"]
meta = bundles["A"]["metadata"]
summary = _load_json("metrics_summary.json")
v2 = _load_json("v2_summary.json")

st.title("🌊 Flood Forecaster (Day to Day)")
st.caption(f"{basin['name']} · USGS {basin['id']} · {basin['lat']:.4f}, {basin['lon']:.4f} · "
           f"drainage {basin['area_km2']} km²")

# ---------------------------------------------------------------- introduction
_fit_y0 = int(meta["fit_period"][0][:4])
_val_y1 = int(meta["validation_period"][1][:4])
_thrA = bundles["A"]
st.markdown(
    f"Estimates the chance of a flood on **each of the next seven days** for one creek in "
    f"Pennsylvania — {basin['name'].split(' at ')[0]}, about {basin['area_km2']} km². "
    f"It deliberately reads **only rainfall and temperature** (not the river's current level), "
    f"because weather services forecast rain a week ahead — that buys **seven days of warning "
    f"instead of one**, at a cost in accuracy. Most flagged days will **not** flood: treat it as "
    f"a *keep-an-eye-on-the-forecast* signal.")
with st.expander("❓ What is this, and why does it matter?"):
    st.markdown(
        f"**What this does.** Estimates the chance of a flood — the river topping "
        f"**{_thrA['flood_threshold_cfs']:,.0f} cfs** (cubic feet per second, "
        f"≈ {_thrA.get('flood_threshold_m3s', 0):.0f} m³/s; only ~2% of days ever get that high) — "
        f"on each of the next seven days at {basin['name']} (~{basin['area_km2']} km²).\n\n"
        f"**The interesting trade.** Most flood models read today's river level. That works well "
        f"but only sees about one day ahead — predicting two days out would need *tomorrow's* "
        f"river level, which doesn't exist yet. This model deliberately ignores the river and uses "
        f"only rainfall and temperature, because weather services forecast rain a week ahead. That "
        f"buys seven days of warning instead of one, and costs accuracy. **That trade is the "
        f"project**, and it's stated everywhere rather than hidden. (An optional variant, Model B, "
        f"adds *today's* — and only today's — river reading to sharpen the next day or two.)\n\n"
        f"**How it works.** Each day combines how wet the ground already is (the past 30 days of "
        f"rain) with how much rain is coming (the 7-day forecast). Saturated ground plus a big "
        f"storm is dangerous; the same storm on dry ground often isn't. It learned this from "
        f"**{_val_y1 - _fit_y0 + 1} years** of records ({_fit_y0}–{_val_y1}). To show uncertainty "
        f"it runs across many weather forecasts at once (an **ensemble** — the same storm "
        f"simulated dozens of times with slightly different starting conditions). Forecasts "
        f"agreeing → narrow range; disagreeing → wide range, and trust the number less.\n\n"
        f"**What it does not do.** It is **not an official flood warning** — for real warnings use "
        f"**weather.gov / the National Weather Service**. Most flagged days will not flood: it's a "
        f"*keep an eye on the forecast* signal. It's least reliable furthest out, because the rain "
        f"forecast itself degrades. And it knows nothing about soil type, snowpack, or upstream "
        f"dams — it's statistical, not physical.")

tab_main, tab_good, tab_method = st.tabs(
    ["📊 This week, day by day", "🎯 How good is this really?", "🧪 Model & method"])

# ===========================================================================
# MAIN
# ===========================================================================
with tab_main:
    try:
        weather, fetched_at = get_weather(basin["lat"], basin["lon"])
    except ld.LiveDataError as e:
        st.warning(f"⚠️ Live weather is unavailable right now, so today's forecast can't be "
                   f"produced. ({e}) The model and its evaluation are still in the other tabs.")
        st.stop()

    river = get_river(basin["id"])

    # --- interactive options ---
    with st.expander("⚙️ Options — model, ensemble system & what-if scenario", expanded=False):
        oc1, oc2, oc3 = st.columns(3)
        model_choice = oc1.radio(
            "Model", ["Auto", "Rain-only (A)", "Rain + today's discharge (B)"], index=0,
            help="Auto = Model B (adds today's USGS discharge) when USGS is up, else rain-only (A).")
        ens_labels = {"gfs_seamless": "GFS / GEFS (NOAA)", "icon_seamless": "ICON (DWD)",
                      "ecmwf_ifs025": "ECMWF IFS"}
        ens_model = oc2.selectbox("Ensemble system", list(ens_labels), index=0,
                                  format_func=lambda m: ens_labels[m],
                                  help="Which weather ensemble drives the uncertainty band.")
        rain_scale = oc3.slider("What-if: scale forecast rain ×", 0.0, 3.0, 1.0, 0.1,
                                help="Hypothetical — multiply the forecast rain to test the model's "
                                     "rain sensitivity. 1.0 = the real forecast.")

    ensemble = get_ensemble(basin["lat"], basin["lon"], ens_model)
    if ensemble is None and ens_model != "gfs_seamless":
        st.caption(f"'{ens_labels[ens_model]}' ensemble unavailable right now — trying GFS/GEFS.")
        ensemble = get_ensemble(basin["lat"], basin["lon"], "gfs_seamless")
    glofas = get_glofas(basin["lat"], basin["lon"])
    weather = weather.copy(); weather.attrs["fetched_at"] = fetched_at

    # what-if: scale ONLY the forecast-window rain (never the observed antecedent)
    if abs(rain_scale - 1.0) > 1e-9:
        m = ~weather["is_observed"]
        weather.loc[m, "precip_mm"] = weather.loc[m, "precip_mm"] * rain_scale
        if ensemble:
            ensemble = dict(ensemble)
            md = ensemble["member_daily"].copy()
            md["precip_mm"] = md["precip_mm"] * rain_scale
            ensemble["member_daily"] = md
        st.info(f"🧪 **Hypothetical scenario:** forecast rain scaled ×{rain_scale:.1f} — a what-if to "
                "probe the model, **not** the real forecast.")

    allow_b = model_choice != "Rain-only (A)"
    if model_choice == "Rain + today's discharge (B)" and not (river and river.get("discharge")):
        st.caption("Model B needs today's USGS discharge, which is unavailable — showing rain-only (A).")

    # score at PRIMARY thresholds; the strictness slider re-thresholds live below
    try:
        res = ld.predict_week(weather_df=weather, bundles=bundles, river=river,
                              ensemble=ensemble, allow_model_b=allow_b)
    except Exception as e:
        st.warning(f"⚠️ Could not score today's forecast: {e}")
        st.stop()
    if res["staleness"]["stale"]:
        st.warning(f"⏳ {res['staleness']['message']} Treat today's window with caution.")

    thr_cfs, thr_mm = res["threshold_cfs"], res["threshold_mm_day"]
    model_used = res["model_used"]
    per = pd.DataFrame(res["per_day"])
    band = np.array(res["band_matrix"]) if res["band_matrix"] else None
    prim, hc = res["thresholds_primary"], res["thresholds_high_confidence"]

    # --- alert-strictness slider (0 = catch more / primary, 1 = high-confidence ~2%) ---
    _ev = v2.get("event") if v2 else None
    strict = st.slider(
        "🎚️ Alert strictness — left catches more floods (more false alarms); right alerts only on "
        "the strongest signals", 0.0, 1.0, 0.0, 0.05)
    if hc:  # interpolate the per-horizon threshold between the two operating points
        per["threshold"] = [(1 - strict) * prim[hz] + strict * hc[hz] for hz in per["horizon"]]
    per["alert"] = per["p"] >= per["threshold"]
    if band is not None:
        per["n_exceed"] = [int((band[:, i] >= per["threshold"].iloc[i]).sum()) for i in range(len(per))]
    head = res["headline"]
    w_thr = ((1 - strict) * head["threshold_primary"] + strict * head["threshold_high_confidence"]
             if head.get("threshold_high_confidence") else head["threshold_primary"])
    head_alert = head["p"] >= w_thr
    if _ev:
        af = (1 - strict) * _ev["primary"]["alert_frequency"] + strict * _ev["high_confidence"]["alert_frequency"]
        st.caption(f"At this strictness the model historically alerts on ~**{af*100:.0f}% of days** "
                   f"(slide: primary {_ev['primary']['alert_frequency']*100:.0f}% → high-confidence "
                   f"{_ev['high_confidence']['alert_frequency']*100:.0f}%). Fewer alerts = fewer false "
                   "alarms but more missed/late floods.")
        st.caption(f"↳ At the two ends (held-out 2005–14): **primary** caught "
                   f"{_ev['primary']['POD_7day']*100:.0f}% of floods but {_ev['primary']['false_alarm_ratio']*100:.0f}% "
                   f"of its warn-days were false · **high-confidence** caught "
                   f"{_ev['high_confidence']['POD_7day']*100:.0f}% with {_ev['high_confidence']['false_alarm_ratio']*100:.0f}% "
                   f"false — always read detection and false alarms together.")

    # --- lead-time trust filter: "I only act on alerts up to N days ahead" ---
    lead_trust = st.slider("🕐 I only act on alerts up to this many days ahead", 1, 7, 7,
                           help="Days beyond this are greyed out below. The readout shows what "
                                "that cut-off buys and costs, at the primary operating point.")
    _ead = event_alert_data()
    if lead_trust < 7 and _ead is not None and summary:
        pod_n = float(_ead["events"]["fired_leads"].apply(
            lambda ls: any(L <= lead_trust for L in ls)).mean())
        prec_n = [m["precision"] for m in summary["metricsA"]
                  if m["horizon"] != "ALL" and int(m["horizon"]) <= lead_trust]
        far_n = 1 - float(np.mean(prec_n)) if prec_n else float("nan")
        st.caption(f"↳ Acting only on leads ≤ **{lead_trust} d**: detects "
                   f"**{pod_n*100:.0f}%** of historical flood events; ~**{far_n*100:.0f}%** of those "
                   f"per-day alerts were false (held-out, primary operating point, perfect-prog → "
                   f"optimistic). Shorter leads = fewer false alarms, less warning time.")

    # --- forecast change tracker (vs the previous real-forecast run; skips hypotheticals) ---
    if abs(rain_scale - 1.0) < 1e-9:
        try:
            _hist_f = DATA / "_cache" / "run_history.json"
            _prev = json.loads(_hist_f.read_text()) if _hist_f.exists() else None
            _cur = {"ts": res.get("fetched_at") or "", "day_t": str(res["day_t"]),
                    "p": {str(d["date"]): d["p"] for d in res["per_day"]}}
            if _prev and _prev.get("p") and _prev != _cur:
                _moves = []
                for dstr, pnew in _cur["p"].items():
                    pold = _prev["p"].get(dstr)
                    if pold is not None and abs(pnew - pold) >= 0.005:
                        _moves.append(f"{pd.Timestamp(dstr):%a %m/%d}: {pold*100:.0f}% → {pnew*100:.0f}%")
                if _moves:
                    st.caption("📈 **Since the previous check** (" + (_prev.get("day_t") or "?") + "): "
                               + " · ".join(_moves[:5]))
            _hist_f.parent.mkdir(exist_ok=True)
            _hist_f.write_text(json.dumps(_cur))
        except Exception:
            pass  # tracker is best-effort; never break the page

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("At least one flood day in the next 7 days", f"{head['p']*100:.0f}%",
                  delta="ALERT" if head_alert else "no alert",
                  delta_color="inverse" if head_alert else "off")
        st.caption(plain_odds(head["p"]) + " (over the whole week).")
    with c2:
        wm = bundles["windowed"]["metadata"]["metrics"]
        st.markdown(
            f"**Headline** = the separately-calibrated *windowed* model (P ≥ 1 flood day in "
            f"t+1…t+7), ~**{wm['precision']*100:.0f}% precision / {wm['recall']*100:.0f}% recall** "
            f"on held-out years — *most alerts don't verify; “watch the forecast.”* A **flood day** "
            f"= river ≥ **{thr_cfs:,.0f} cfs** ({thr_mm:.1f} mm/day ≈ {res.get('threshold_m3s',0):.0f} "
            f"m³/s), the 98th pct of 1980–2004 flow (~2% of days).")
    st.caption("Headline and daily bars come from *different* models (windowed vs per-day) — they "
               "agree in spirit but the headline is **not** the product of the bars.")

    # ---- day-by-day interactive bars with ensemble bands (Plotly) ----
    labels = [f"t+{r.horizon}<br>{r.date:%a %m/%d}" for r in per.itertuples()]
    colors = [risk_hex(p, t) for p, t in zip(per["p"], per["threshold"])]
    # lead-trust filter: grey out days beyond the lead the user said they'd act on
    colors = [c if int(per["horizon"].iloc[k]) <= lead_trust else "#d0d0d0"
              for k, c in enumerate(colors)]
    hastb = res["has_band"]
    err_plus = ((per["p90"] - per["p"]).clip(lower=0) * 100).tolist() if hastb else None
    err_minus = ((per["p"] - per["p10"]).clip(lower=0) * 100).tolist() if hastb else None
    nex = per["n_exceed"] if "n_exceed" in per else [None] * len(per)
    bartext = [(("🔺 " if a else "") + f"{p*100:.1f}%" + (f"<br>{int(n)}/{res['n_members']}" if hastb else ""))
               for a, p, n in zip(per["alert"], per["p"], nex)]
    hov = []
    for i in range(len(per)):
        extra = (f"<br>P10–P90 = {per['p10'].iloc[i]*100:.1f}–{per['p90'].iloc[i]*100:.1f}%"
                 f"<br>{int(per['n_exceed'].iloc[i])}/{res['n_members']} members flag" if hastb else "")
        hov.append(f"<b>{labels[i].replace('<br>', ' ')}</b><br>median P = {per['p'].iloc[i]*100:.1f}%"
                   f"{extra}<br>alert thr = {per['threshold'].iloc[i]*100:.1f}%"
                   f"<br>forecast rain = {per['fc_rain_mm'].iloc[i]:.1f} mm")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34],
                        vertical_spacing=0.08,
                        subplot_titles=("calibrated P(flood) %", "forecast rain (mm/day)"))
    fig.add_trace(go.Bar(x=labels, y=per["p"]*100, marker_color=colors, marker_line_color="#333",
                         marker_line_width=.6, text=bartext, textposition="outside",
                         hovertext=hov, hoverinfo="text",
                         error_y=dict(type="data", array=err_plus, arrayminus=err_minus,
                                      visible=hastb, color="#222", thickness=1.3, width=5),
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=per["threshold"]*100, mode="lines", name="alert threshold",
                  line=dict(color="#555", dash="dash", width=1), hoverinfo="skip"), row=1, col=1)
    show_rain_band = bool(hastb and per["fc_rain_p10"].notna().any())
    rain_ep = ((per["fc_rain_p90"] - per["fc_rain_mm"]).clip(lower=0)).tolist() if show_rain_band else None
    rain_em = ((per["fc_rain_mm"] - per["fc_rain_p10"]).clip(lower=0)).tolist() if show_rain_band else None
    fig.add_trace(go.Bar(x=labels, y=per["fc_rain_mm"], marker_color="#4575b4", showlegend=False,
                         error_y=dict(type="data", array=rain_ep, arrayminus=rain_em,
                                      visible=show_rain_band, color="#222", thickness=1, width=4),
                         hovertemplate="rain %{y:.1f} mm<extra></extra>"), row=2, col=1)
    fig.add_hline(y=thr_mm, line=dict(color="#a50026", dash="dot", width=1), row=2, col=1,
                  annotation_text=f"1-day flood-equiv ≈ {thr_mm:.1f} mm", annotation_font_size=10)
    ymax = (per["p90"] if hastb else per["p"]).max() * 100
    band_txt = (f"{res['n_members']} {res['ensemble_model']} members" if hastb
                else "ensemble band unavailable — single point estimate")
    fig.update_yaxes(range=[0, max(6, ymax*1.4)], row=1, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=66, b=10), bargap=0.25,
                      title=dict(text=(f"Per-day flood probability · day t = {res['day_t']:%Y-%m-%d} · "
                             f"model {model_used} "
                             f"({'rain + today’s USGS discharge' if model_used=='B' else 'rain-only'})"
                             f" · {band_txt}"), font=dict(size=13)))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("⬇️ Download this week (CSV)",
                       per[["date", "horizon", "p", "p10", "p90", "threshold", "alert",
                            "fc_rain_mm"]].to_csv(index=False),
                       file_name=f"flood_week_{res['day_t']}.csv", mime="text/csv")

    # ---- ⛈️ storm builder: add rain to specific days, watch the week cascade ----
    with st.expander("⛈️ Storm builder — put rain on specific days and watch the whole week "
                     "respond (hypothetical)"):
        _cap = train_rain_max()
        fc_dates = [pd.Timestamp(d).date() for d in
                    weather.loc[~weather["is_observed"], "date"]]
        live_rain = [float(x) for x in weather.loc[~weather["is_observed"], "precip_mm"]]
        for k in range(7):
            st.session_state.setdefault(f"sb_{k}", round(live_rain[k], 1))
        pb1, pb2, pb3, pb4 = st.columns(4)
        if pb1.button("🌧 A wet day (t+3)"):
            for k in range(7):
                st.session_state[f"sb_{k}"] = round(live_rain[k], 1)
            st.session_state["sb_2"] = 35.0
        if pb2.button("⛈ A major storm (t+3/t+4)"):
            for k in range(7):
                st.session_state[f"sb_{k}"] = round(live_rain[k], 1)
            st.session_state["sb_2"] = min(90.0, _cap)
            st.session_state["sb_3"] = 40.0
        if pb3.button("☀️ A dry week"):
            for k in range(7):
                st.session_state[f"sb_{k}"] = 0.0
        if pb4.button("↺ Reset to live forecast"):
            for k in range(7):
                st.session_state[f"sb_{k}"] = round(live_rain[k], 1)
        sb_cols = st.columns(7)
        sb_vals = [sb_cols[k].number_input(f"t+{k+1}\n{fc_dates[k]:%a}", min_value=0.0,
                                           max_value=float(_cap), step=5.0, key=f"sb_{k}",
                                           help=f"{fc_dates[k]:%A %b %d} rain (mm)")
                   for k in range(7)]
        if any(v >= _cap - 1e-9 for v in sb_vals):
            st.warning(f"Capped at **{_cap:.0f} mm/day** — the largest daily rain ever observed in "
                       f"the 1980–2014 training record. Beyond that the model would be "
                       f"extrapolating blind.")
        modified = any(abs(v - lv) > 0.05 for v, lv in zip(sb_vals, live_rain))
        if modified:
            w_storm = weather.copy()
            w_storm.loc[~w_storm["is_observed"], "precip_mm"] = sb_vals
            try:
                res_storm = ld.predict_week(weather_df=w_storm, bundles=bundles, river=river,
                                            ensemble=None, allow_model_b=allow_b)
                storm_p = [d["p_point"] for d in res_storm["per_day"]]
                base_p = per["p_point"].tolist()
                fs = go.Figure()
                fs.add_trace(go.Bar(x=labels, y=[p * 100 for p in base_p], name="live forecast",
                                    marker_color="#9ecae1"))
                fs.add_trace(go.Bar(x=labels, y=[p * 100 for p in storm_p], name="your storm",
                                    marker_color="#d73027"))
                fs.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=34, b=10),
                                 yaxis_title="P(flood) % (point estimate)",
                                 title=dict(text="Live forecast vs your hypothetical storm",
                                            font=dict(size=12)),
                                 legend=dict(orientation="h", y=1.15))
                st.plotly_chart(fs, use_container_width=True)
                st.caption("🧪 **Exploring a hypothetical, not a forecast.** Notice the **cascade**: "
                           "rain added to one day also lifts the days after it, because the model's "
                           "forecast features are cumulative — incoming rain keeps mattering once "
                           "it has fallen. Point estimates (no ensemble band) for both bars.")
            except Exception as e:
                st.caption(f"Could not score the storm scenario: {e}")
        else:
            st.caption("Edit any day's rain (or use a preset) to see the week re-scored. "
                       "This panel explores hypotheticals — it never changes the real forecast above.")

    # ---- per-day drill-down (day-card selector) ----
    st.markdown("#### 🔎 Inspect a single day")
    opts = [f"t+{r.horizon} · {r.date:%A %b %d}" for r in per.itertuples()]
    day_labels = [f"{r.date:%a %b %d}" for r in per.itertuples()]
    pick = st.radio("Pick a day", day_labels, horizontal=True)
    i = day_labels.index(pick)
    sel = opts[i]

    def _risk_tier(p, thr):
        if p >= thr:
            return "High", "#a50026"
        if p >= 0.5 * thr:
            return "Watch", "#e08214"
        return "Low", "#1a9850"

    card_cols = st.columns(7)
    for k, rr in enumerate(per.itertuples()):
        word, colr = _risk_tier(rr.p, rr.threshold)
        border = "3px solid #111" if k == i else "1px solid #d9d9d9"
        card_cols[k].markdown(
            f"<div style='border:{border};border-radius:12px;padding:10px 6px;text-align:center;'>"
            f"<div style='font-size:0.9rem;'>{rr.date:%a}</div>"
            f"<div style='font-size:0.8rem;color:#777;margin-bottom:7px;'>{rr.date:%b %d}</div>"
            f"<div style='background:{colr};color:#fff;border-radius:8px;padding:5px 0;"
            f"font-weight:700;'>{word}</div></div>", unsafe_allow_html=True)

    _w, _c = _risk_tier(per["p"].iloc[i], per["threshold"].iloc[i])
    st.markdown(f"## {per['date'].iloc[i]:%A, %B %d} — "
                f"<span style='color:{_c};'>{_w} risk · {per['p'].iloc[i]*100:.1f}% chance</span>",
                unsafe_allow_html=True)
    st.caption("High = at/above the alert threshold at the current strictness · Watch = within "
               "half of it · Low = below that.")
    r = per.iloc[i]
    d1, d2, d3 = st.columns(3)
    d1.metric(f"P(flood) · {sel.split(' · ')[1]}", f"{r['p']*100:.1f}%",
              delta="ALERT" if r["alert"] else "watch",
              delta_color="inverse" if r["alert"] else "off")
    if hastb:
        d2.metric("Ensemble range (P10–P90)", f"{r['p10']*100:.1f}–{r['p90']*100:.1f}%",
                  help="Spread of the forecast ensemble only — true uncertainty is wider.")
        d3.metric("Members flagging this day", f"{int(r['n_exceed'])} / {res['n_members']}")
    else:
        d2.metric("Ensemble range", "unavailable")
        d3.metric("Forecast rain", f"{r['fc_rain_mm']:.1f} mm")
    lead = int(r["horizon"])
    trust = ("**most trustworthy** lead" if lead <= 2 else
             "**softest** lead — read as directional" if lead >= 6 else "mid-range lead")
    rain_note = (f" (ensemble {r['fc_rain_p10']:.1f}–{r['fc_rain_p90']:.1f} mm)"
                 if hastb and pd.notna(r["fc_rain_p10"]) else "")
    st.caption(f"Lead **t+{lead}** ({r['date']:%A %b %d}) · {trust}. Forecast rain that day: "
               f"**{r['fc_rain_mm']:.1f} mm**{rain_note}. Alert threshold at this strictness: "
               f"{r['threshold']*100:.1f}%. This chance is {plain_odds(float(r['p']))}.")

    # ---- why is this day risky? antecedent (wet ground) vs forecast (incoming rain) ----
    try:
        w_dry = weather.copy()
        w_dry.loc[~w_dry["is_observed"], "precip_mm"] = 0.0
        res_dry = ld.predict_week(weather_df=w_dry, bundles=bundles, river=river,
                                  ensemble=None, allow_model_b=allow_b)
        ground = float(res_dry["per_day"][i]["p_point"])
        full_pt = float(r["p_point"])
        rain_part = max(0.0, full_pt - ground)
        f2 = go.Figure()
        f2.add_trace(go.Bar(y=[""], x=[ground * 100], name="ground already wet (past 30 days)",
                            orientation="h", marker_color="#8c6d31",
                            hovertemplate="wet-ground part: %{x:.1f}%<extra></extra>"))
        f2.add_trace(go.Bar(y=[""], x=[rain_part * 100], name="incoming rain (forecast)",
                            orientation="h", marker_color="#3182bd",
                            hovertemplate="incoming-rain part: %{x:.1f}%<extra></extra>"))
        f2.update_layout(barmode="stack", height=120, margin=dict(l=8, r=8, t=30, b=8),
                         xaxis_title="P(flood) %  (point estimate)",
                         legend=dict(orientation="h", y=2.2),
                         title=dict(text=f"Why is {sel} risky? — the two risk sources",
                                    font=dict(size=12)))
        st.plotly_chart(f2, use_container_width=True)
        if full_pt < 0.01:
            _sent = "essentially negligible from **both** sources today"
        elif rain_part > 2 * ground:
            _sent = "mostly **incoming rain** — the ground itself isn't the problem"
        elif ground > 2 * rain_part:
            _sent = "mostly **ground already wet** — even modest rain lands on a saturated basin"
        else:
            _sent = "a **mix** — wet ground and incoming rain both contribute"
        st.caption(f"{r['date']:%A}'s risk is {_sent}. *Method: the day is re-scored with all "
                   f"forecast rain removed (a leave-block-out re-score) — what's left is the "
                   f"wet-ground part; the difference is attributed to incoming rain. Approximate "
                   f"attribution, not exact; point estimates.*")
    except Exception:
        st.caption("Risk-source split unavailable right now.")

    if band is not None:
        vals = band[:, i] * 100
        fh = go.Figure(go.Histogram(x=vals, nbinsx=12, marker_color="#4575b4",
                                    marker_line_color="#333", marker_line_width=.4))
        fh.add_vline(x=r["threshold"]*100, line=dict(color="#a50026", dash="dash"),
                     annotation_text="alert thr", annotation_font_size=10)
        fh.add_vline(x=r["p"]*100, line=dict(color="#1a9850"),
                     annotation_text="median", annotation_font_size=10)
        fh.update_layout(height=250, margin=dict(l=10, r=10, t=42, b=10),
                         title=dict(text=f"How the {res['n_members']} ensemble members disagree for {sel}",
                                    font=dict(size=12)), xaxis_title="member calibrated P(flood) %",
                         yaxis_title="# members", showlegend=False)
        st.plotly_chart(fh, use_container_width=True)
    else:
        st.caption("No ensemble today → no member spread to show for this day (single estimate above).")

    # ---- 🍝 ensemble explorer: every member's rain + best/worst-case scenarios ----
    with st.expander("🍝 Ensemble explorer — see every forecast member, and the realistic "
                     "best / worst cases"):
        if ensemble and band is not None:
            _md = ensemble["member_daily"]
            _piv = _md.pivot_table(index="date", columns="member", values="precip_mm")
            _fcd = pd.to_datetime(pd.Series([pd.Timestamp(d) for d in
                    weather.loc[~weather["is_observed"], "date"]]))
            _piv = _piv.reindex(pd.DatetimeIndex(_fcd)).dropna(how="all")
            fsp = go.Figure()
            for mcol in _piv.columns:
                fsp.add_trace(go.Scatter(x=_piv.index, y=_piv[mcol], mode="lines",
                                         line=dict(color="rgba(70,117,180,0.22)", width=1),
                                         hoverinfo="skip", showlegend=False))
            fsp.add_trace(go.Scatter(x=_piv.index, y=_piv.median(axis=1), mode="lines+markers",
                                     line=dict(color="#08306b", width=3), name="member median",
                                     hovertemplate="%{x|%a %m/%d}: %{y:.1f} mm<extra>median</extra>"))
            fsp.update_layout(height=260, margin=dict(l=10, r=10, t=34, b=10),
                              yaxis_title="forecast rain (mm/day)",
                              title=dict(text=f"All {res['n_members']} members' 7-day rainfall "
                                              f"({res['ensemble_model']})", font=dict(size=12)))
            st.plotly_chart(fsp, use_container_width=True)
            # scenario toggle -- reads the ALREADY-computed per-member risks (band matrix)
            totals = _piv.sum(axis=0)
            if len(totals) == band.shape[0]:
                scen = st.radio("Scenario", ["Ensemble median (shown above)", "Wettest member",
                                             "Driest member"], horizontal=True)
                if scen != "Ensemble median (shown above)":
                    k = int(np.argmax(totals.to_numpy()) if scen == "Wettest member"
                            else np.argmin(totals.to_numpy()))
                    fsc = go.Figure()
                    fsc.add_trace(go.Bar(x=labels, y=per["p"] * 100, name="ensemble median",
                                         marker_color="#9ecae1"))
                    fsc.add_trace(go.Bar(x=labels, y=band[k, :] * 100,
                                         name=f"{scen.lower()} ({totals.iloc[k]:.0f} mm total)",
                                         marker_color="#d73027" if scen == "Wettest member" else "#74c476"))
                    fsc.update_layout(barmode="group", height=280, margin=dict(l=10, r=10, t=30, b=10),
                                      yaxis_title="P(flood) %", legend=dict(orientation="h", y=1.15),
                                      title=dict(text="Week re-read under one member's rain — a "
                                                      "realistic best/worst case, not a forecast",
                                                 font=dict(size=12)))
                    st.plotly_chart(fsc, use_container_width=True)
            st.caption("🧪 Single-member scenarios are **hypothetical readings of one plausible "
                       "future**, not forecasts. And the whole band is **forecast-input spread "
                       "only** — it excludes model error, so true uncertainty is wider.")
        else:
            st.caption("Ensemble unavailable right now — no member spaghetti or scenarios to show.")

    # ---- GloFAS side-by-side (Plotly) ----
    st.markdown("#### Operational reference — GloFAS (ECMWF, ~5 km) for comparison")
    if glofas and glofas.get("median_cfs"):
        gd = pd.DataFrame({"date": glofas["dates"], "median": glofas["median_cfs"],
                           "p25": glofas.get("p25_cfs"), "p75": glofas.get("p75_cfs")})
        figg = go.Figure()
        if gd["p25"].notna().any():
            figg.add_trace(go.Scatter(x=gd["date"], y=gd["p75"], mode="lines", line=dict(width=0),
                                      hoverinfo="skip", showlegend=False))
            figg.add_trace(go.Scatter(x=gd["date"], y=gd["p25"], mode="lines", line=dict(width=0),
                                      fill="tonexty", fillcolor="rgba(33,102,172,0.2)",
                                      name="GloFAS p25–p75", hoverinfo="skip"))
        figg.add_trace(go.Scatter(x=gd["date"], y=gd["median"], mode="lines+markers",
                                  line=dict(color="#2166ac"), name="GloFAS median",
                                  hovertemplate="%{x|%a %m/%d}: %{y:.0f} cfs<extra></extra>"))
        figg.add_hline(y=thr_cfs, line=dict(color="#a50026", dash="dash", width=1),
                       annotation_text=f"USGS flood thr {thr_cfs:,.0f} cfs", annotation_font_size=10)
        figg.update_layout(height=250, margin=dict(l=10, r=10, t=16, b=10),
                           yaxis_title="discharge (cfs)", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(figg, use_container_width=True)
        note = ("⚠️ GloFAS resolved this basin only *indicatively* (coarse ~5 km grid) — treat as a "
                "rough reference." if glofas.get("indicative") else
                f"GloFAS reach resolved by nudging coords {glofas.get('offset')}° to match the USGS scale.")
        st.caption(note + " GloFAS forecast is the operational standard; this ML model sits beside it for comparison.")
    else:
        st.info("GloFAS operational forecast unavailable right now — comparison hidden.")

    st.markdown(
        "**Reliability by lead day.** t+1/t+2 are the most trustworthy; t+6/t+7 the least — the "
        "model is less certain further out **and** the rain forecast itself degrades with lead time. "
        "The whiskers show **forecast-input spread only** (the weather ensemble), *not* total "
        "uncertainty — the true range is wider.")

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown("#### Current river & basin context")
        if river and river.get("discharge"):
            d = river["discharge"]
            st.metric("Current observed river level (USGS)", f"{float(d['value']):,.0f} {d.get('unit','cfs')}")
            if model_used == "B":
                st.caption("Today's level is feeding the near-horizon prediction (Model B).")
        else:
            st.info("Current river level unavailable (USGS) — serving the **rain-only** Model A.")
        # recent observed river-level sparkline (last ~30 days, USGS daily)
        rh = get_river_history(basin["id"])
        if rh is not None and len(rh) >= 3:
            figr = go.Figure(go.Scatter(x=rh["date"], y=rh["q_cfs"], mode="lines",
                             line=dict(color="#2166ac"),
                             hovertemplate="%{x|%a %m/%d}: %{y:.0f} cfs<extra></extra>"))
            figr.add_hline(y=thr_cfs, line=dict(color="#a50026", dash="dot", width=1),
                           annotation_text=f"flood thr {thr_cfs:,.0f} cfs", annotation_font_size=9)
            figr.update_layout(height=170, margin=dict(l=6, r=6, t=24, b=6),
                               yaxis_title="cfs", title=dict(text="Observed river level, last ~30 days",
                               font=dict(size=11)), showlegend=False)
            st.plotly_chart(figr, use_container_width=True)
        else:
            st.caption("Recent river-level history unavailable right now.")
    with cc2:
        st.markdown("#### Basin location")
        st.map(pd.DataFrame({"lat": [basin["lat"]], "lon": [basin["lon"]]}), zoom=7)
    st.caption(f"Forecast fetched: {res.get('fetched_at','?')} · Weather/ensemble: Open-Meteo "
               f"({res.get('ensemble_model') or 'deterministic'}) · GloFAS: Open-Meteo Flood · "
               f"River: modern USGS API")

    with st.expander("📖 Glossary — the five terms this page uses"):
        st.markdown(
            f"- **Flood threshold** — the river level that counts as a flood here: "
            f"**{thr_cfs:,.0f} cfs** ≈ **{res.get('threshold_m3s', 0):.0f} m³/s** "
            f"(≈ {thr_mm:.1f} mm of runoff/day). It's the 98th percentile of 1980–2004 flow, "
            f"so only ~2 days in 100 ever reach it.\n"
            f"- **Calibrated probability** — a % that means its number: historically, days shown "
            f"as \"12%\" flooded about 12 times out of 100. Not a raw model score.\n"
            f"- **Ensemble** — the same weather forecast run dozens of times "
            f"({res['n_members'] or 'many'} members today) with slightly different starting "
            f"conditions; the spread of answers is the whisker on each bar.\n"
            f"- **Lead time** — how many days ahead a warning comes (t+1 = tomorrow … t+7). "
            f"Longer lead = more time to act, less reliable signal.\n"
            f"- **False alarm** — an alert-day with no flood. At a ~2% base rate most alerts are "
            f"false by design; that's why detection is always shown next to the false-alarm rate.")

# ===========================================================================
# HOW GOOD
# ===========================================================================
with tab_good:
    st.subheader("Per-horizon test metrics (held-out 2005–2014)")
    if summary:
        mdf = pd.DataFrame(summary["metricsA"]); mdf["horizon"] = mdf["horizon"].astype(str)
        show = mdf.rename(columns={"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})[
            ["horizon", "base_rate", "ROC-AUC", "PR-AUC", "recall", "precision", "f1", "brier"]]
        st.dataframe(show.style.format({
            "base_rate": "{:.3%}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "recall": "{:.2f}",
            "precision": "{:.2f}", "f1": "{:.2f}", "brier": "{:.4f}"}, na_rep="—"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        if (FIGURES / "skill_gradient.png").exists():
            st.image(str(FIGURES / "skill_gradient.png"))
        st.caption("Held-out gradient is ~flat *because* the forecast block is trained & tested on "
                   "observed (perfect) rain — flattering the far horizons. Real degradation shows in "
                   "the real-forecast backtest below.")
    with c2:
        if (FIGURES / "reliability_per_horizon.png").exists():
            st.image(str(FIGURES / "reliability_per_horizon.png"))
        st.caption("A displayed daily *X%* ≈ that historical flood rate for days like this. Near "
                   "horizons calibrate tighter than far ones.")

    # ---- event-based (two operating points; POD never shown alone) ----
    st.subheader("Event-based evaluation — does it warn in time?")
    ev = v2.get("event") if v2 else None
    if ev:
        pr, hcp = ev["primary"], ev["high_confidence"]
        st.markdown(
            "> **The model catches almost every historical flood, but only because it alerts on a "
            "large fraction of all days — most alert-days have no flood.** Read a long lead time as "
            "*“this basin is often in a watch state,”* not precise week-ahead prediction. So POD is "
            "always shown beside the **false-alarm ratio** and the **alert frequency**.")
        tbl = pd.DataFrame([
            {"operating point": "primary (recall-targeted)", "POD (7d)": pr["POD_7day"],
             "false-alarm ratio": pr["false_alarm_ratio"], "alert frequency": pr["alert_frequency"],
             "CSI": pr["CSI"], "median lead (d)": pr["median_lead_days"]},
            {"operating point": "high-confidence (~2% target)", "POD (7d)": hcp["POD_7day"],
             "false-alarm ratio": hcp["false_alarm_ratio"], "alert frequency": hcp["alert_frequency"],
             "CSI": hcp["CSI"], "median lead (d)": hcp["median_lead_days"]}])
        st.dataframe(tbl.style.format({"POD (7d)": "{:.2f}", "false-alarm ratio": "{:.2f}",
                     "alert frequency": "{:.2f}", "CSI": "{:.2f}", "median lead (d)": "{:.0f}"}),
                     width="stretch", hide_index=True)
        st.caption(f"{pr['n_events']} flood events "
                   "(2005–2014). **Lead time is capped at 7 days by construction**, so a median of 7 "
                   "means the mass is saturated at the cap (the alert was already on) — not that the "
                   "model resolves timing. The high-confidence column shows the honest cost: far fewer "
                   "false alarms, but lower detection and lead. Both inherit perfect-prog optimism; the "
                   "real-forecast backtest below is the live view.")
        if (FIGURES / "event_evaluation.png").exists():
            st.image(str(FIGURES / "event_evaluation.png"))
    else:
        st.info("Run `evaluate_v2.py` to populate the event-based panel.")

    # ---- 🎞️ historical event replay (all 41 events, including the misses) ----
    st.subheader("Replay a real flood — what would this model have said?")
    _ead2 = event_alert_data()
    tp_all = get_test_predictions()
    qh = get_q_history()
    if _ead2 is not None and tp_all is not None:
        evs = _ead2["events"].sort_values("onset").reset_index(drop=True)
        ev_opts = [f"{r.onset:%Y-%m-%d} · {'✓ caught' if r.caught else '✗ MISSED'} · "
                   f"{r.length} flood day{'s' if r.length > 1 else ''}" for r in evs.itertuples()]
        ev_sel = st.selectbox("Pick a flood event from the held-out 2005–2014 record "
                              "(the list includes the model's misses)", ev_opts,
                              index=len(ev_opts) - 1)
        er = evs.iloc[ev_opts.index(ev_sel)]
        D = pd.Timestamp(er["onset"])
        prim_thr = get_bundles()["A"]["thresholds"]
        rows_r = []
        for L in range(7, 0, -1):
            m = tp_all[(tp_all["anchor_date"] == D - pd.Timedelta(days=L)) &
                       (tp_all["horizon"] == L)]
            if len(m):
                rows_r.append({"lead": L, "p": float(m["p_h"].iloc[0]),
                               "thr": float(prim_thr[L]), "fired": L in er["fired_leads"]})
        rp_df = pd.DataFrame(rows_r)
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            if len(rp_df):
                fr = go.Figure()
                fr.add_trace(go.Scatter(x=rp_df["lead"], y=rp_df["p"] * 100, mode="lines+markers",
                                        name="P(flood on onset day)", line=dict(color="#2166ac"),
                                        marker=dict(size=9, color=["#d73027" if f else "#2166ac"
                                                                   for f in rp_df["fired"]])))
                fr.add_trace(go.Scatter(x=rp_df["lead"], y=rp_df["thr"] * 100, mode="lines",
                                        name="alert threshold", line=dict(color="#555", dash="dash")))
                fr.update_layout(height=280, margin=dict(l=10, r=10, t=34, b=10),
                                 xaxis=dict(title="days before the flood (lead)", autorange="reversed"),
                                 yaxis_title="P(flood) %", legend=dict(orientation="h", y=1.18),
                                 title=dict(text=f"The forecast for {D:%b %d, %Y} as it approached "
                                                 "(red dots = alert fired)", font=dict(size=12)))
                st.plotly_chart(fr, use_container_width=True)
        with rc2:
            if qh is not None:
                win = qh[(qh["date"] >= D - pd.Timedelta(days=10)) &
                         (qh["date"] <= pd.Timestamp(er["end"]) + pd.Timedelta(days=5))]
                fq = go.Figure(go.Scatter(x=win["date"], y=win["q_cfs"], mode="lines",
                                          line=dict(color="#08306b"),
                                          hovertemplate="%{x|%b %d}: %{y:,.0f} cfs<extra></extra>"))
                fq.add_hline(y=thr_cfs, line=dict(color="#a50026", dash="dash", width=1),
                             annotation_text="flood threshold", annotation_font_size=9)
                fq.update_layout(height=280, margin=dict(l=10, r=10, t=34, b=10),
                                 yaxis_title="observed discharge (cfs)",
                                 title=dict(text="What the river actually did", font=dict(size=12)))
                st.plotly_chart(fq, use_container_width=True)
        _mkey = f"{D:%Y-%m}"
        _mstat = _ead2["month_stats"].get(_mkey, {})
        if er["caught"]:
            verdict = (f"**Warned {er['max_lead']} day{'s' if er['max_lead'] > 1 else ''} ahead** "
                       f"(earliest alert that fired for the onset day).")
        else:
            verdict = "**✗ This flood was MISSED** — no alert fired for the onset day at any lead."
        st.markdown(f"{verdict} That same month ({_mkey}) the model issued warnings on "
                    f"**{int(_mstat.get('warn_days', 0))}** days, of which "
                    f"**{int(_mstat.get('false_days', 0))}** had no flood in the following week — "
                    f"detection never comes free of false alarms.")
        st.caption("Replay uses the held-out test predictions at the primary operating point. Those "
                   "were scored with **perfect-prog (observed) rain**, so the real-time forecast "
                   "would have been weaker — most at the longer leads.")
    else:
        st.info("Event replay needs `data/test_predictions.csv` + `data/event_eval.csv` "
                "(run `evaluate_v2.py`).")

    # ---- real-forecast backtest ----
    st.subheader("Real-forecast backtest — the honest “how good live?”")
    rf = v2.get("real_forecast") if v2 else None
    if rf:
        st.markdown("**The money plot — per horizon, not just the aggregate.** Day+1 holds up; the far "
                    "horizons drop most under real (imperfect) forecasts. That declining red curve is "
                    "what a live user actually gets; the flat perfect-prog gradient above is not.")
        rfd = pd.DataFrame(rf["per_horizon"])
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            if (FIGURES / "real_forecast_backtest.png").exists():
                st.image(str(FIGURES / "real_forecast_backtest.png"))
        with cc2:
            st.dataframe(rfd[["horizon", "n_pos", "pr_auc_perfect", "pr_auc_real",
                              "recall_perfect", "recall_real"]].rename(columns={
                "pr_auc_perfect": "PR-AUC perf", "pr_auc_real": "PR-AUC real",
                "recall_perfect": "recall perf", "recall_real": "recall real"}).style.format({
                "PR-AUC perf": "{:.3f}", "PR-AUC real": "{:.3f}", "recall perf": "{:.2f}",
                "recall real": "{:.2f}"}, na_rep="—"), width="stretch", hide_index=True)
        st.caption(f"Separate {rf['window'][0]}…{rf['window'][1]} window · {rf['n_pos']} flood targets · "
                   f"single basin — a realism check, not the primary test. {rf['note']}")
    else:
        st.info("Real-forecast backtest unavailable (needs the Historical-Forecast + USGS-daily APIs). "
                "The synthetic degradation stress test still spans all leads:")
    if summary and summary.get("stress"):
        sdf = pd.DataFrame(summary["stress"])
        st.dataframe(sdf.rename(columns={"pr_auc_perfect": "PR-AUC (perfect)",
                     "pr_auc_degraded": "PR-AUC (noisy QPF)"}).style.format({
                     "PR-AUC (perfect)": "{:.3f}", "PR-AUC (noisy QPF)": "{:.3f}"}),
                     width="stretch", hide_index=True)

    # ---- GloFAS benchmark + matched operating point ----
    st.subheader("External benchmark — GloFAS reanalysis (not a forecast)")
    gb = v2.get("glofas") if v2 else None
    gm = v2.get("glofas_matched") if v2 else None
    if gb:
        st.markdown(
            f"Resolution check: at the exact gauge coords GloFAS reads **{gb['base_coord_mean_m3s']:.2f} "
            f"m³/s** mean — implausible for a {basin['area_km2']} km² basin (USGS mean "
            f"**{gb['usgs_mean_m3s']:.1f} m³/s**). Nudging coords **{gb['offset']}°** gives "
            f"**{gb['chosen_mean_m3s']:.1f} m³/s** ({'still indicative' if gb['indicative'] else 'resolved'}).")
        if gm:
            st.markdown(
                f"**Matched operating point (fair comparison).** Bare CSI numbers are apples-to-oranges "
                f"because the two run at different alert rates, so we compare on **POD-vs-FAR**. At "
                f"GloFAS's false-alarm ratio (**{gm['glofas']['FAR']:.2f}**, alert-freq "
                f"{gm['glofas']['alert_freq']:.2f}), GloFAS-reanalysis detects **POD "
                f"{gm['glofas']['POD']:.2f}** vs the model's shortest-lead (h=1) **POD "
                f"{gm['model_pod_at_glofas_far']:.2f}** at the same FAR. GloFAS *reanalysis* (observed "
                f"forcing) is a strong upper-bound reference — the model does not beat it here, and that's "
                f"the honest read.")
            g1, g2 = st.columns([1, 1])
            with g1:
                if (FIGURES / "glofas_matched.png").exists():
                    st.image(str(FIGURES / "glofas_matched.png"))
            with g2:
                if (FIGURES / "glofas_benchmark.png").exists():
                    st.image(str(FIGURES / "glofas_benchmark.png"))
        elif (FIGURES / "glofas_benchmark.png").exists():
            st.image(str(FIGURES / "glofas_benchmark.png"))
        st.caption("GloFAS is **reanalysis** (observed-forcing) — an upper-bound reference, **not** a "
                   "like-for-like forecast benchmark (that needs GloFAS reforecasts; see README). Units "
                   "converted m³/s → cfs; ~5 km grid needed the coordinate nudge above.")
    else:
        st.info("GloFAS benchmark not available (run `evaluate_v2.py` with network).")

    st.subheader("How this compares (mind the different scales)")
    if summary:
        wm = bundles["windowed"]["metadata"]["metrics"]
        comp = pd.DataFrame([
            {"model": "This windowed headline (≥1 flood day in 7)", "ROC-AUC": wm["roc_auc"],
             "base rate": summary["windowed"]["base_rate"]},
            {"model": "Published 1-day model", "ROC-AUC": summary["one_day_model_roc_auc"],
             "base rate": summary["per_day_base_rate_test"]},
            {"model": "Prior 'flood within 7 days' build", "ROC-AUC": summary["old_windowed_roc_auc"],
             "base rate": summary["windowed"]["base_rate"]}])
        st.dataframe(comp.style.format({"ROC-AUC": "{:.3f}", "base rate": "{:.2%}"}),
                     width="stretch", hide_index=True)

    st.subheader("Offline validation view (2005–2014)")
    tp = get_test_predictions()
    if tp is not None:
        hsel = st.selectbox("Filter by lead day (horizon)", options=list(range(1, 8)), index=0)
        sub = tp[tp["horizon"] == hsel]
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(sub["target_date"], sub["p_h"], lw=.6, color="#2166ac", label="predicted P")
        fl = sub[sub["flood_on_day"] == 1]
        ax.scatter(fl["target_date"], np.full(len(fl), sub["p_h"].max() * 1.02), marker="v",
                   color="#a50026", s=18, label="actual flood day", zorder=3)
        ax.set_title(f"t+{hsel}: predicted daily P(flood) vs actual flood days")
        ax.set_ylabel("P(flood)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ===========================================================================
# MODEL & METHOD
# ===========================================================================
with tab_method:
    st.subheader("Three-block feature taxonomy")
    st.markdown(
        "| Block | Time window | Filled at train | Filled at serve |\n|---|---|---|---|\n"
        "| **A. Antecedent** (≤ t) | rain 1/3/7/14/30, EWMA, temps, season of *t* | observed ERA5 | "
        "Open-Meteo `past_days` |\n"
        "| **B. Forecast** (t+1…t+h) | cum/on-day/prev1-2/max forecast rain, forecast tmax | "
        "**observed** ERA5 (*perfect-prog*) | **real** forecast — each **ensemble member** |\n"
        "| **C. Horizon** | h ∈ 1…7, season of target day | — | — |\n")
    st.markdown(
        "The **forecast block is the only place a forecast-day value enters** — observed rain at "
        "train (perfect-prog), the real forecast at serve. That reversal lets each day move with its "
        "own weather while the antecedent stays strictly backward-looking (no leak; sentinel-tested).")

    st.subheader("How the ensemble band is built")
    st.markdown(
        f"We fetch the **{ld.ENSEMBLE_MODEL}** ensemble (~31 members). The shared observed antecedent "
        "is held fixed; each member's **forecast-window** rain is pushed through the same model + "
        "per-horizon calibrators, giving one calibrated p_h per member. The bar is the **median**, the "
        "whisker the **P10–P90**, and “N/M” counts members above the alert threshold. **It captures "
        "forecast-input spread only** — not model error or antecedent-source uncertainty — so true "
        "uncertainty is wider (honesty fact #4).")

    st.subheader("Design")
    st.markdown(
        f"- **Pooled multi-horizon** HistGradientBoosting (horizon = feature) → all 7 risks; monotone "
        f"+1 on every rain feature.\n"
        f"- **Leakage-safe split by anchor day**, chronological, 7-day embargo; CV = {meta['cv_scheme']}.\n"
        f"- **Per-horizon isotonic calibration** + per-horizon threshold on the validation slice "
        f"({meta['validation_period'][0]}…{meta['validation_period'][1]}) only.\n"
        f"- **Model B** adds today's backward-looking discharge q_t (live USGS), falls back to A when "
        f"USGS is down; the headline is a **separate** windowed model.\n"
        f"- Fit {meta['fit_period'][0]}…{meta['fit_period'][1]}; test {meta['test_period'][0]}…"
        f"{meta['test_period'][1]}. scikit-learn {meta['sklearn']}, Python {meta['python']}.")

    c1, c2 = st.columns(2)
    with c1:
        for f in ("feature_importance.png", "windowed_calibration.png", "event_evaluation.png"):
            if (FIGURES / f).exists():
                st.image(str(FIGURES / f))
    with c2:
        for f in ("forecast_degradation.png", "real_forecast_backtest.png", "glofas_matched.png"):
            if (FIGURES / f).exists():
                st.image(str(FIGURES / f))

    st.subheader("Honesty facts & limitations")
    st.markdown(
        "0. Every per-day number is a **per-horizon-calibrated probability**.\n"
        "1. **Reliability drops with lead day** (model + QPF skill).\n"
        "2. **Held-out metrics are optimistic vs live**, most at far horizons — the real-forecast "
        "backtest measures the gap.\n"
        "3. **~2% per-day base rate → low precision**; bars are “watch,” not certainties.\n"
        "4. **The band is forecast-input spread only** — true uncertainty is wider.\n"
        "5. **GloFAS is coarse (~5 km)** and needed a coordinate nudge; the historical comparison is "
        "**reanalysis, not a forecast**.\n"
        "6. **Model B** uses only backward-looking discharge (day t), live from USGS; falls back to A.\n"
        "- Single basin, single split, daily resolution, per-horizon thresholds — not a general model.")

st.divider()
st.caption("Weather, ensemble & GloFAS by **Open-Meteo** (CC BY 4.0). River data: **USGS** (public "
           "domain).")
