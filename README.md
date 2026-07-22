# Flood Forecaster (Day to Day) — USGS 01543000

Per-day, per-horizon-**calibrated** flood risk for each of the next 7 days with
**ensemble uncertainty bands**, a live **GloFAS operational benchmark**, and
**event-based + real-forecast** evaluation — plus an honest *"at least one flood day
this week"* headline.

**Basin:** Driftwood Branch Sinnemahoning Creek at Sterling Run, PA (Cameron County;
272 mi² ≈ 705 km²), USGS gauge **01543000**, lat/lon **41.4133 / -78.1972**.

## What is this, and why does it matter?

**What this does.** Estimates the chance of a flood — the river topping **2,544 cfs**
(≈ 72 m³/s; only ~2% of days ever get that high) — on each of the next seven days for
one creek in Pennsylvania (~705 km²).

**The interesting trade.** Most flood models read today's river level. That works well
but only sees about one day ahead — predicting two days out would need *tomorrow's*
river level, which doesn't exist yet. This model deliberately ignores the river and
uses only rainfall and temperature, because weather services forecast rain a week
ahead. That buys **seven days of warning instead of one**, and costs accuracy. That
trade is the project, and it's stated everywhere rather than hidden. (An optional
variant, Model B, adds *today's* — and only today's — river reading to sharpen the
next day or two.)

**How it works.** Each day combines how wet the ground already is (the past 30 days of
rain) with how much rain is coming (the 7-day forecast). Saturated ground plus a big
storm is dangerous; the same storm on dry ground often isn't. Learned from **25 years**
of records (1980–2004), tested on the next ten. To show uncertainty it runs across many
weather forecasts at once (an **ensemble** — the same storm simulated dozens of times
with slightly different starting conditions). Forecasts agreeing → narrow range;
disagreeing → wide range, and trust the number less.

**What it does not do.** It is **not an official flood warning** — for real warnings
use **weather.gov / the National Weather Service**. Most flagged days will not flood:
it's a *keep an eye on the forecast* signal. It's least reliable furthest out, because
the rain forecast itself degrades. And it knows nothing about soil type, snowpack, or
upstream dams — it's statistical, not physical.

### v2 additions
1. **Ensemble bands** — each day's risk is scored across the ~31-member GFS ensemble
   (`gfs_seamless`); the app shows the **median + P10–P90** whisker and "N of M members
   flag this day." The band is *forecast-input* uncertainty only.
2. **GloFAS benchmark** — Open-Meteo's operational GloFAS discharge shown live beside the
   model, plus a historical reanalysis comparison. A **mandatory ~5 km resolution check**
   corrects a mis-picked reach (exact coords read 0.35 m³/s; nudging +0.1°lat/−0.1°lon
   gives 11 m³/s ≈ the USGS scale).
3. **Event + real-forecast evaluation** — flood-**event** detection (POD / lead time /
   false-alarm / CSI) and a **real archived-forecast backtest** (Historical Forecast API +
   USGS observed labels) that measures the perfect-prog optimism gap.

---

## What makes this different from a "flood within 7 days" model

A windowed model predicts one label — *a flood on **any** of days t+1…t+7* — from
features that all stop at day *t*. It therefore (1) **cannot tell the days apart**
(the input is identical for every horizon) and (2) **ignores the forecast**. This
build fixes both:

1. **Per-day target** `flood_on_day[t+h]` for each horizon h = 1…7.
2. A **forecast block** of features built from the QPF (forecast precipitation) for
   days t+1…t+h, so each day's risk moves with that day's incoming weather.

One **pooled multi-horizon** gradient-boosted model (horizon *h* is a feature) emits
all 7 per-day risks; a **separate windowed model** supplies the headline so it is
never faked from the daily bars.

### Three feature blocks (the split is load-bearing for leakage)

| Block | Window | Train fill | Serve fill |
|---|---|---|---|
| **A. Antecedent** (≤ t) | rolling rain 1/3/7/14/30, EWMA, tmax/tmin/tmax_3d, season of *t* | observed ERA5 | Open-Meteo `past_days` |
| **B. Forecast** (t+1…t+h) | `fc_rain_cum/on_h/prev1/prev2/max`, `fc_tmax_on_h` | **observed** ERA5 rain (*perfect-prog*) | **real** Open-Meteo `forecast_days` |
| **C. Horizon** | `horizon` h∈1…7, season of target day t+h | — | — |

The forecast block is the **only** place a forecast-day value may enter; the
antecedent block stays strictly backward-looking. One shared row-builder
(`features.make_horizon_rows`) is used for training **and** serving, so features
cannot drift — enforced by a sentinel leakage self-test and a byte-parity self-test.

---

## Results (held-out 2005–2014)

- **Flood threshold** = 98th percentile of 1980–2004 flow = **2,543.8 cfs ≈ 8.83 mm/day**;
  per-day base rate ≈ **2%** (train 2.01% / test 2.25%).
- **Model A (rain-only)** — aggregate **ROC-AUC ≈ 0.92**, PR-AUC ≈ 0.29; near-horizon
  h+1 ROC-AUC ≈ 0.93.
- **Model B (+ today's discharge q_t)** — lifts near-horizon **PR-AUC 0.28 → 0.56**;
  served when USGS is up, else falls back to Model A.
- **Windowed headline** — ROC-AUC ≈ **0.79** (cf. published 1-day model 0.943 and the
  prior "flood within 7 days" build ≈ 0.79; *different targets/base rates — not
  directly comparable*).
- **Event-based** (41 flood events) — POD is **never shown alone** (it's inflated by the alert
  rate). At two operating points:
  | operating point | POD (7d) | false-alarm ratio | alert frequency | CSI |
  |---|---|---|---|---|
  | primary (recall-targeted) | 0.95 | 0.84 | **0.51** | 0.16 |
  | high-confidence (~2% target) | 0.78 | 0.56 | 0.13 | 0.35 |

  The primary POD 0.95 comes at the cost of alerting on **half of all days**; the high-confidence
  threshold (derived on the validation fold) trades detection for far fewer false alarms. Lead time
  is capped at 7 days by construction, so "median lead 7 d" means the alert was already on — not
  timing skill.
- **Real-forecast backtest** (2022–2024, 119 flood targets) — **per horizon**: PR-AUC/recall hold up
  at day+1 and drop most at far horizons under real forecasts (aggregate PR-AUC **0.40 perfect-prog →
  0.32 real**; archived-forecast vs observed rain r≈0.57, genuinely imperfect). This declining curve
  is the honest live view; the flat perfect-prog gradient is not.
- **GloFAS benchmark** (reanalysis, coord-corrected) — compared at a **matched operating point**: at
  GloFAS's false-alarm ratio (0.54), GloFAS-reanalysis detects **POD 0.41** vs the model's shortest-lead
  **POD 0.19**. GloFAS *reanalysis* (observed-forcing) is a strong upper-bound reference and this ML
  build does **not** beat it — the honest read; it is *not* a like-for-like forecast benchmark.

Metrics, calibration curves, the skill-gradient, the ensemble bands and all v2 panels are in
`flood_daybyday.ipynb` and the app's *"How good is this really?"* tab.

## How to read this app (for a non-expert)

- The **headline %** is the chance of *at least one* flood day in the next 7 days.
- The **7 interactive bars** are each day's flood chance; the **whisker** is the weather-ensemble
  spread and "N of M" is how many forecast members flag that day. **Bands show forecast spread only
  — real uncertainty is wider.**
- **Pick a day** in the *Inspect a single day* panel to see its detail card and a **histogram of how
  the 31 ensemble members disagree** for that day.
- The **alert-strictness slider** slides between "catch more floods (more false alarms)" and
  "high-confidence only," and the bars/alerts re-threshold live — with the historical alert-rate shown.
- A **flood day** here is the river topping its 98th-percentile flow (~2% of days), so most alerts
  are *"keep an eye on it,"* not "a flood is coming."
- **GloFAS** is the professional reference shown beside the model; a **basin map** and a **recent
  river-level sparkline** give context.

## Live robustness

Day-t (= last observed day) is resolved in **America/New_York**, never the deploy server's UTC clock;
if the forecast feed lags, the app shows a **staleness warning** instead of silently shifting the
window. The ensemble member count is read from the response (never hardcoded) and falls back to a
single-bar point estimate if too few members return. The **live path only** calls the ensemble/
deterministic forecast + USGS + GloFAS (each cached ~1 h) — **never** the archive or historical-forecast
endpoints, which are eval-only. Every API-down path degrades to a friendly message (verified,
including ensemble+GloFAS both down).

---

## Honesty facts (shown in the notebook and the app)

0. Every displayed per-day number is a **per-horizon-calibrated probability**.
1. **Reliability decreases with lead day** — t+1/t+2 most trustworthy, t+6/t+7 least
   (inherent uncertainty + QPF skill degrading with lead time).
2. **Live skill is optimistically estimated** by the held-out metrics (perfect-prog
   train vs imperfect real forecast), most at far horizons; the stress test bounds it.
3. **~2% per-day base rate → low precision by design**; daily bars are *"watch these
   days,"* not certainties.
4. **The ensemble band is forecast-input spread only** — the weather ensemble propagated
   through the model. It ignores model error and antecedent-source uncertainty, so the true
   range is wider.
5. **GloFAS is coarse (~5 km)** and needed a coordinate nudge to resolve this basin; the
   historical comparison is **reanalysis, not a forecast**.
6. **Model B** uses only *backward-looking* discharge (day *t*), sourced live from USGS;
   never future discharge, and degrades to rain-only when USGS is down.

---

## Project layout

```
features.py          three-block feature engineering + shared row-builder
train_daybyday.py    training pipeline (labels, pooled set, anchor-day CV, Model A/B,
                     LR baseline, windowed model, per-horizon calibration, figures)
evaluate_v2.py       [v2] event-based eval + real-forecast backtest + GloFAS benchmark
                     (writes glofas_config.json with the resolution-corrected offset)
live_data.py         live pipeline: deterministic + ensemble forecast, USGS, GloFAS;
                     build 7 rows, predict_week (with ensemble bands)
app.py               Streamlit day-by-day UI (3 tabs, bands + GloFAS side-by-side)
flood_daybyday.ipynb narrative notebook (regenerate via _build_notebook.py)
models/              flood_daybyday_model.joblib (A), _model_B.joblib, _windowed_model.joblib
data/                cached ERA5 weather, streamflow QC, test_predictions*.csv,
                     metrics_summary.json, [v2] v2_summary.json, event_eval.csv,
                     glofas_benchmark.csv, glofas_config.json, real_forecast_backtest.csv
figures/             skill gradient, per-horizon reliability, windowed calibration,
                     forecast-degradation, feature importance, [v2] event_evaluation,
                     real_forecast_backtest, glofas_benchmark
```

### APIs (all free, no key; Open-Meteo is CC BY 4.0)
`archive-api` (ERA5 training) · `api…/forecast` (deterministic) ·
`ensemble-api…/ensemble` (GFS members) · `flood-api…/flood` (GloFAS) ·
`historical-forecast-api` (real archived forecasts) · modern USGS `api.waterdata.usgs.gov`
(latest + daily). No new Python deps — all are plain `requests` calls.

## Run it

```bash
py -3.12 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt      # deploy deps
venv\Scripts\python train_daybyday.py                        # (re)train + write artifacts
venv\Scripts\python evaluate_v2.py                           # [v2] event/real-forecast/GloFAS eval
venv\Scripts\python live_data.py                             # 3 self-tests (parity + leakage + ensemble)
venv\Scripts\streamlit run app.py                            # the app
```

**Deploy (Streamlit Community Cloud):** pick **Python 3.12** in the Advanced-settings
dropdown; `requirements.txt` holds exact pins. `models/`, the cached CSVs and every
`.joblib` are committed (small; no LFS) so the app loads on first boot.

---

## Deferred (future work)

- **LSTM / seq2seq (MIMO)** ingesting the rain+temp sequence to emit all 7 daily risks
  at once — likely beats the pooled GBM, at the cost of TF/PyTorch deploy weight.
- **GloFAS reforecasts** for a true like-for-like forecast benchmark (vs the reanalysis
  reference used here).
- **Multi-basin generalization (CAMELS)** with basin attributes → ungauged rivers.
- **Conformal prediction** on top of per-horizon calibration for guaranteed coverage.

*(v2 delivered the ensemble probabilistic-rain bands that v1 had deferred.)*

---

*Weather, ensemble & GloFAS data by [Open-Meteo](https://open-meteo.com) (CC BY 4.0).
River data: USGS (public domain).*
