"""
Shared feature engineering for the DAY-BY-DAY 7-day flood forecaster.

Imported by BOTH the training pipeline (train_daybyday.py) and the live app
(live_data.py). Keeping the rolling windows, the EWMA alpha, the forecast-block
maths and the exact column order in ONE place is what guarantees the live app
builds features byte-identically to training -- so train/serve drift cannot
silently occur.

Three clearly-named feature blocks (the split is load-bearing for leakage):

  A) ANTECEDENT  -- strictly <= day t, observed, backward-looking.
        rain_1/3/7/14/30, api_ewma, tmax, tmin, tmax_3d, doy_sin/cos (of day t).
     These use ONLY precip on days <= t (rolling sums look backward), so a
     forecast-day value can never enter them.

  B) FORECAST    -- the QPF for days t+1 .. t+h (this is what makes each of the
        seven days different). fc_rain_cum, fc_rain_on_h, fc_rain_prev1/2,
        fc_rain_max, fc_tmax_on_h.
        * TRAIN time: filled from the OBSERVED ERA5 rain on t+1..t+h -- a
          perfect-forecast (perfect-prog) proxy.
        * SERVE time: filled from the REAL Open-Meteo forecast_days values.
        This is the deliberate reversal of the antecedent rule and applies ONLY
        to this block.

  C) HORIZON     -- horizon h in {1..7} and the seasonality of the TARGET day
        t+h (doy_sin_target, doy_cos_target).

`make_horizon_rows` is the single row-builder used by both sides. At train time
it is handed the full observed ERA5 series and every valid anchor day; at serve
time it is handed the [past_days | forecast_days] window and the one anchor
(today's last observed day). Same code => guaranteed parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- exact feature order for each block; the persisted bundle stores these ---
ANTECEDENT = [
    "rain_1", "rain_3", "rain_7", "rain_14", "rain_30",
    "api_ewma", "tmax", "tmin", "tmax_3d", "doy_sin", "doy_cos",
]
FORECAST = [
    "fc_rain_cum", "fc_rain_on_h", "fc_rain_prev1", "fc_rain_prev2",
    "fc_rain_max", "fc_tmax_on_h",
]
HORIZON = ["horizon", "doy_sin_target", "doy_cos_target"]

# Model A (rain-only) flat feature order.
FEATURES_A = ANTECEDENT + FORECAST + HORIZON
# Model B adds today's (backward-looking) discharge; served only when USGS is up.
DISCHARGE = ["q_t", "q_ewma_3"]
FEATURES_B = FEATURES_A + DISCHARGE

# Antecedent-only feature set for the windowed headline model (old recipe).
FEATURES_WINDOWED = ANTECEDENT

EWMA_ALPHA = 0.1
Q_EWMA_ALPHA = 0.5  # 3-day-ish discharge smoother
HORIZONS = list(range(1, 8))  # t+1 .. t+7


# ---------------------------------------------------------------------------
# Rain features that are +1 monotonic ("more rain -> not less risk") in Model A.
# ---------------------------------------------------------------------------
RAIN_MONOTONIC = {
    "rain_1", "rain_3", "rain_7", "rain_14", "rain_30", "api_ewma",
    "fc_rain_cum", "fc_rain_on_h", "fc_rain_prev1", "fc_rain_prev2", "fc_rain_max",
}


def monotonic_cst(features: list[str]) -> list[int]:
    """+1 on rain features (and discharge), 0 elsewhere -- for HistGradientBoosting."""
    out = []
    for f in features:
        if f in RAIN_MONOTONIC or f in ("q_t", "q_ewma_3"):
            out.append(1)
        else:
            out.append(0)
    return out


# ---------------------------------------------------------------------------
# Block A: antecedent features (backward-looking, <= day t)
# ---------------------------------------------------------------------------
def antecedent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ANTECEDENT columns to a daily frame.

    Required input columns: date (datetime), precip_mm, tmax, tmin.
    Rows must be sorted ascending and one calendar day apart. Rolling features
    with an undefined full window stay NaN (warm-up rows); api_ewma and the doy
    terms are defined from the first row.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    p = df["precip_mm"]
    df["rain_1"] = p.rolling(1, min_periods=1).sum()
    df["rain_3"] = p.rolling(3, min_periods=3).sum()
    df["rain_7"] = p.rolling(7, min_periods=7).sum()
    df["rain_14"] = p.rolling(14, min_periods=14).sum()
    df["rain_30"] = p.rolling(30, min_periods=30).sum()
    df["api_ewma"] = p.ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    df["tmax_3d"] = df["tmax"].rolling(3, min_periods=3).mean()
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _doy_terms(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    return np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


# ---------------------------------------------------------------------------
# The single row-builder: emit one row per (anchor, horizon).
# Used identically at train time (all anchors, perfect-prog block) and serve
# time (one anchor, real-forecast block).
# ---------------------------------------------------------------------------
def make_horizon_rows(
    weather_df: pd.DataFrame,
    anchor_idx: np.ndarray | list[int],
    horizons: list[int] = HORIZONS,
    discharge: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the pooled (anchor x horizon) feature frame.

    Parameters
    ----------
    weather_df : daily frame with ANTECEDENT columns already added
        (call antecedent_features first). Must be sorted ascending, contiguous.
    anchor_idx : positional indices of the anchor days t. Each must satisfy
        anchor+max(horizons) < len(weather_df) so the forecast/target rows exist.
    horizons : list of lead days h.
    discharge : optional aligned q_cfs array (same length as weather_df) used to
        build the Model-B discharge block AND the per-day label. When None (live
        serving with no discharge) the discharge/label columns are omitted.

    Returns one row per (anchor, horizon) with block A/B/C features, anchor_date,
    target_date, and -- if discharge is given -- q_target (raw target discharge).
    """
    w = weather_df.sort_values("date").reset_index(drop=True)
    n = len(w)
    anchor_idx = np.asarray(anchor_idx, dtype=int)
    P = w["precip_mm"].to_numpy(dtype=float)
    TMAX = w["tmax"].to_numpy(dtype=float)
    dates = w["date"].to_numpy()
    # prefix sums for O(1) window sums: pre[k] = sum(P[0..k-1])
    pre = np.concatenate([[0.0], np.cumsum(P)])

    ante = w[ANTECEDENT].to_numpy(dtype=float)  # antecedent block at each row
    dsin_t, dcos_t = _doy_terms(w["date"])

    if discharge is not None:
        q = np.asarray(discharge, dtype=float)
        q_ewma = pd.Series(q).ewm(alpha=Q_EWMA_ALPHA, adjust=False).mean().to_numpy()

    frames = []
    for h in horizons:
        tgt = anchor_idx + h
        valid = tgt < n
        a = anchor_idx[valid]
        t = tgt[valid]

        # --- forecast block over days a+1 .. a+h (perfect-prog / real forecast) ---
        fc_cum = pre[a + h + 1] - pre[a + 1]              # sum P[a+1..a+h]
        fc_on_h = P[t]                                    # P[a+h]
        fc_prev1 = P[a + h - 1] if h >= 2 else np.zeros_like(a, dtype=float)
        fc_prev2 = P[a + h - 2] if h >= 3 else np.zeros_like(a, dtype=float)
        # running max over the window a+1..a+h
        fc_max = P[a + 1].copy()
        for k in range(2, h + 1):
            fc_max = np.maximum(fc_max, P[a + k])
        if h == 0:  # not used (horizons start at 1) but keep safe
            fc_max = np.zeros_like(a, dtype=float)
        fc_tmax = TMAX[t]

        block = {}
        for j, name in enumerate(ANTECEDENT):
            block[name] = ante[a, j]
        block["fc_rain_cum"] = fc_cum
        block["fc_rain_on_h"] = fc_on_h
        block["fc_rain_prev1"] = fc_prev1
        block["fc_rain_prev2"] = fc_prev2
        block["fc_rain_max"] = fc_max
        block["fc_tmax_on_h"] = fc_tmax
        block["horizon"] = np.full(a.shape, h, dtype=float)
        block["doy_sin_target"] = dsin_t[t]
        block["doy_cos_target"] = dcos_t[t]
        block["anchor_date"] = dates[a]
        block["target_date"] = dates[t]
        if discharge is not None:
            block["q_t"] = q[a]
            block["q_ewma_3"] = q_ewma[a]
            block["q_target"] = q[t]
        frames.append(pd.DataFrame(block))

    out = pd.concat(frames, ignore_index=True)
    return out
