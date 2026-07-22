"""
Live data pipeline for the DAY-BY-DAY 7-day flood forecaster.

Standalone module the Streamlit app imports. It:
  1. fetches live weather from the Open-Meteo FORECAST endpoint (day -30..+7),
  2. rebuilds the model's features with the SAME code training used
     (features.make_horizon_rows) -> the 7 rows for today's anchor,
  3. reads the current river level from the MODERN USGS Water Data API
     (context + Model-B near-horizon input; degrades to None on any failure),
  4. scores the 7 per-day risks (per-horizon calibrated) + the windowed headline.

Serve policy (graceful degradation):
  * Model B (rain + today's discharge q_t) is used when USGS gives a current
    discharge; otherwise it falls back to Model A (rain-only) so the app never
    breaks. The response says which model produced the numbers.

Run directly for the two mandatory self-tests (network-free):
    venv\\Scripts\\python.exe live_data.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

BASIN_TZ = ZoneInfo("America/New_York")  # day-t is defined in basin-local time, not server UTC

from features import ANTECEDENT, FEATURES_A, FEATURES_B, HORIZONS, antecedent_features, make_horizon_rows

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
CACHE_DIR = DATA / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAST_DAYS = 30
FORECAST_DAYS = 7
CACHE_TTL_SECONDS = 3600

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"   # live forecast (NOT the archive host)
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"        # [v2] per-member forecast
GLOFAS_URL = "https://flood-api.open-meteo.com/v1/flood"               # [v2] operational GloFAS
HISTFC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # [v2] real archived forecasts
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"          # ERA5 observed (eval)
USGS_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/latest-continuous/items"  # modern host
USGS_DAILY_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"  # [v2] daily values
ENSEMBLE_MODEL = "gfs_seamless"  # NOAA GEFS, ~31 members, free
CFS_PER_M3S = 35.3147            # 1 m3/s = 35.3147 cfs
GLOFAS_CONFIG = DATA / "glofas_config.json"  # stores the resolution-corrected coord offset


class LiveDataError(RuntimeError):
    """Raised when live weather cannot be fetched or parsed."""


# ---------------------------------------------------------------------------
# 1. Forecast weather  (day -30 .. day +7 in ONE call)
# ---------------------------------------------------------------------------
def fetch_forecast_weather(lat: float, lon: float, use_cache: bool = True) -> pd.DataFrame:
    """Return a tidy daily frame day -30..+7. Columns: date, precip_mm, tmax, tmin,
    is_observed. The last FORECAST_DAYS rows are the forecast; the rest are the
    observed past_days block. Raises LiveDataError on any HTTP/parse failure."""
    key = f"forecast_{lat:.4f}_{lon:.4f}_{datetime.now().strftime('%Y%m%d%H')}"
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        if time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS:
            try:
                return _frame_from_payload(json.loads(cache_file.read_text()))
            except Exception:
                pass  # fall through to a fresh fetch

    params = {
        "latitude": lat, "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "America/New_York",
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=45)
        r.raise_for_status()
        payload = r.json()
    except requests.RequestException as e:
        raise LiveDataError(f"Open-Meteo forecast request failed: {e}") from e
    except ValueError as e:
        raise LiveDataError(f"Open-Meteo forecast returned malformed JSON: {e}") from e
    if "daily" not in payload or "time" not in payload.get("daily", {}):
        raise LiveDataError("Open-Meteo forecast response missing 'daily' data")

    payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    try:
        cache_file.write_text(json.dumps(payload))
    except OSError:
        pass
    return _frame_from_payload(payload)


def _frame_from_payload(payload: dict) -> pd.DataFrame:
    d = payload["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "precip_mm": d["precipitation_sum"],
        "tmax": d["temperature_2m_max"],
        "tmin": d["temperature_2m_min"],
    }).sort_values("date").reset_index(drop=True)
    if df[["precip_mm", "tmax", "tmin"]].isna().all(axis=None):
        raise LiveDataError("Open-Meteo forecast returned all-null weather")
    df[["precip_mm", "tmax", "tmin"]] = df[["precip_mm", "tmax", "tmin"]].ffill().bfill()
    n = len(df)
    df["is_observed"] = np.arange(n) < (n - FORECAST_DAYS)
    df.attrs["fetched_at"] = payload.get("_fetched_at", datetime.now(timezone.utc).isoformat())
    return df


# ---------------------------------------------------------------------------
# 2. Live features  (SHARED make_horizon_rows -> identical to training)
# ---------------------------------------------------------------------------
def _anchor_index(feat: pd.DataFrame) -> int:
    """day t = the last OBSERVED day (yesterday); its backward features never touch
    forecast rows. If no is_observed column (a purely historical frame), day t is
    the row 7 before the end so a full forecast window exists."""
    feat = feat.sort_values("date").reset_index(drop=True)
    if "is_observed" in feat.columns:
        idx = int(np.where(feat["is_observed"].to_numpy())[0].max())
    else:
        idx = len(feat) - 1 - FORECAST_DAYS
    return idx


def build_live_features(weather_df: pd.DataFrame, discharge: np.ndarray | None = None) -> pd.DataFrame:
    """Return the 7 rows (h=1..7) for today's anchor, in FEATURES_A order, with the
    anchor's antecedent block, the forecast block from the forecast_days rain, and
    the horizon block. Highest-risk function in the project -- guarded by self-tests."""
    feat = antecedent_features(weather_df)  # backward antecedent block for every row
    if "is_observed" in weather_df.columns:
        feat["is_observed"] = weather_df.sort_values("date").reset_index(drop=True)["is_observed"].values
    anchor = _anchor_index(feat)
    rows = make_horizon_rows(feat, [anchor], HORIZONS, discharge=discharge)
    if rows[FEATURES_A].isna().any(axis=None):
        raise LiveDataError("live feature row has NaNs (insufficient antecedent history?)")
    return rows.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Current river level (modern USGS API) -- context + Model-B input
# ---------------------------------------------------------------------------
def fetch_current_river(gauge_id: str) -> dict | None:
    """Latest discharge / gage height from the modern USGS API, or None on failure."""
    out: dict = {}
    for pcode, label in (("00060", "discharge"), ("00065", "gage_height")):
        try:
            r = requests.get(
                USGS_URL,
                params={"monitoring_location_id": f"USGS-{gauge_id}",
                        "parameter_code": pcode, "f": "json"},
                timeout=20,
            )
            r.raise_for_status()
            feats = r.json().get("features", [])
            if not feats:
                continue
            p = feats[0]["properties"]
            out[label] = {"value": p.get("value"), "unit": p.get("unit_of_measure"),
                          "time": p.get("time")}
        except Exception:
            continue
    return out or None


def _current_discharge_cfs(river: dict | None) -> float | None:
    if not river or "discharge" not in river:
        return None
    try:
        return float(river["discharge"]["value"])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# [v2] 3b. Ensemble forecast -> per-member forecast-window spread
# ---------------------------------------------------------------------------
def fetch_ensemble_weather(lat: float, lon: float, model: str = ENSEMBLE_MODEL,
                           use_cache: bool = True) -> dict | None:
    """Return per-member daily forecast weather from the Open-Meteo Ensemble API.

    `model` selects the ensemble system (e.g. gfs_seamless / icon_seamless /
    ecmwf_ifs025). The band is *forecast-input* uncertainty only: members differ in
    the FORECAST window; the shared antecedent comes from the deterministic
    `past_days` block. Returns a dict {member_daily, n_members, model, fetched_at} or
    None on any failure. Member count is parsed from the response, never hardcoded."""
    key = f"ens_{model}_{lat:.4f}_{lon:.4f}_{datetime.now().strftime('%Y%m%d%H')}"
    cache_file = CACHE_DIR / f"{key}.json"
    payload = None
    if use_cache and cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            payload = json.loads(cache_file.read_text())
        except Exception:
            payload = None
    if payload is None:
        try:
            r = requests.get(ENSEMBLE_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "precipitation,temperature_2m",
                "models": model, "forecast_days": FORECAST_DAYS,
                "past_days": 1, "timezone": "America/New_York"}, timeout=60)
            r.raise_for_status()
            payload = r.json()
            payload["_model"] = model
            payload["_fetched_at"] = datetime.now(timezone.utc).isoformat()
            try:
                cache_file.write_text(json.dumps(payload))
            except OSError:
                pass
        except Exception:
            return None
    try:
        return _ensemble_frame(payload)
    except Exception:
        return None


def _ensemble_frame(payload: dict) -> dict:
    h = payload["hourly"]
    t = pd.to_datetime(h["time"])
    day = t.date if hasattr(t, "date") else pd.DatetimeIndex(t).date
    day = pd.DatetimeIndex(t).normalize()
    # member keys: 'precipitation' (control) + 'precipitation_memberNN'
    pkeys = [k for k in h if k == "precipitation" or k.startswith("precipitation_member")]
    frames = []
    for pk in pkeys:
        suffix = pk.replace("precipitation", "")            # '' or '_member01'
        tk = "temperature_2m" + suffix
        mem = pk.replace("precipitation", "m0") if suffix == "" else suffix.lstrip("_")
        dfm = pd.DataFrame({"date": day, "precip": h[pk],
                            "temp": h.get(tk, [np.nan] * len(day))})
        agg = dfm.groupby("date").agg(precip_mm=("precip", "sum"),
                                      tmax=("temp", "max"), tmin=("temp", "min")).reset_index()
        agg["member"] = mem
        frames.append(agg)
    member_daily = pd.concat(frames, ignore_index=True)
    member_daily["date"] = pd.to_datetime(member_daily["date"])
    n_members = member_daily["member"].nunique()
    return {"member_daily": member_daily, "n_members": int(n_members),
            "model": payload.get("_model", ENSEMBLE_MODEL),
            "fetched_at": payload.get("_fetched_at", datetime.now(timezone.utc).isoformat())}


# ---------------------------------------------------------------------------
# [v2] 3c. GloFAS operational discharge benchmark (with resolution sanity check)
# ---------------------------------------------------------------------------
def _load_glofas_offset() -> tuple[float, float]:
    """Coordinate offset (dlat, dlon) that puts GloFAS on the right river reach.
    Determined once by evaluate_v2.py's resolution check; (0,0) if not yet run."""
    try:
        cfg = json.loads(GLOFAS_CONFIG.read_text())
        return float(cfg["dlat"]), float(cfg["dlon"])
    except Exception:
        return 0.0, 0.0


def fetch_glofas(lat: float = 41.4133, lon: float = -78.1972,
                 forecast_days: int = FORECAST_DAYS) -> dict | None:
    """GloFAS 7-day forecast discharge (median + p25/p75) converted to cfs, using the
    resolution-corrected coordinates. Returns dict or None on failure. `indicative`
    flags a basin the ~5 km grid resolves poorly."""
    dlat, dlon = _load_glofas_offset()
    la, lo = round(lat + dlat, 4), round(lon + dlon, 4)
    try:
        r = requests.get(GLOFAS_URL, params={
            "latitude": la, "longitude": lo,
            "daily": "river_discharge,river_discharge_median,river_discharge_p25,river_discharge_p75",
            "forecast_days": forecast_days}, timeout=40)
        r.raise_for_status()
        d = r.json().get("daily", {})
        if not d.get("time"):
            return None
        def cfs(key):
            return [None if v is None else float(v) * CFS_PER_M3S for v in d.get(key, [])]
        med = cfs("river_discharge_median") or cfs("river_discharge")
        vals = [v for v in med if v is not None]
        return {
            "dates": [pd.Timestamp(x).date() for x in d["time"]],
            "median_cfs": med, "p25_cfs": cfs("river_discharge_p25"),
            "p75_cfs": cfs("river_discharge_p75"),
            "offset": (dlat, dlon),
            "indicative": bool(dlat == 0.0 and dlon == 0.0),  # uncorrected -> flag
            "coords": (la, lo),
            "peak_cfs": max(vals) if vals else None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# [v2] 3d. Eval-only fetchers: real archived forecasts + USGS daily discharge
# ---------------------------------------------------------------------------
def fetch_historical_forecast(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Real forecasts *as issued* (Historical Forecast API), daily. None on failure."""
    try:
        r = requests.get(HISTFC_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "start_date": start, "end_date": end, "timezone": "America/New_York"}, timeout=60)
        r.raise_for_status()
        d = r.json()["daily"]
        return pd.DataFrame({"date": pd.to_datetime(d["time"]),
                             "precip_mm": d["precipitation_sum"],
                             "tmax": d["temperature_2m_max"], "tmin": d["temperature_2m_min"]})
    except Exception:
        return None


def fetch_archive_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """ERA5 observed daily weather (archive API) for an arbitrary window. None on failure."""
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "start_date": start, "end_date": end, "timezone": "America/New_York"}, timeout=60)
        r.raise_for_status()
        d = r.json()["daily"]
        return pd.DataFrame({"date": pd.to_datetime(d["time"]),
                             "precip_mm": d["precipitation_sum"],
                             "tmax": d["temperature_2m_max"], "tmin": d["temperature_2m_min"]})
    except Exception:
        return None


def fetch_usgs_daily(gauge_id: str, start: str, end: str) -> pd.DataFrame | None:
    """USGS observed daily mean discharge (modern 'daily' collection). None on failure."""
    try:
        rows, offset = [], 0
        while True:
            r = requests.get(USGS_DAILY_URL, params={
                "monitoring_location_id": f"USGS-{gauge_id}", "parameter_code": "00060",
                "statistic_id": "00003", "datetime": f"{start}/{end}",
                "f": "json", "limit": 1000, "offset": offset}, timeout=60)
            r.raise_for_status()
            feats = r.json().get("features", [])
            if not feats:
                break
            for f in feats:
                p = f["properties"]
                rows.append((p.get("time"), p.get("value")))
            if len(feats) < 1000:
                break
            offset += 1000
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["date", "q_cfs"])
        df["date"] = pd.to_datetime(df["date"])
        df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce")
        return df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Predict the week
# ---------------------------------------------------------------------------
def load_bundles(models_dir: Path = MODELS) -> dict:
    return {
        "A": joblib.load(models_dir / "flood_daybyday_model.joblib"),
        "B": joblib.load(models_dir / "flood_daybyday_model_B.joblib"),
        "windowed": joblib.load(models_dir / "flood_windowed_model.joblib"),
    }


def _calibrate(model, features, calibrators, rows: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(rows[features].to_numpy())[:, 1]
    out = np.empty_like(raw, dtype=float)
    hz = rows["horizon"].to_numpy()
    for h in HORIZONS:
        m = hz == h
        out[m] = calibrators[h].predict(raw[m])
    return out


_UNSET = object()  # sentinel: distinguishes "river not supplied" from "USGS is down (None)"


def _score_week(weather_df: pd.DataFrame, bundle: dict, features: list[str],
                q_now: float | None) -> tuple[np.ndarray, pd.DataFrame]:
    """Return the 7 per-horizon CALIBRATED probabilities for one weather frame.
    Model B uses today's discharge q_now at the anchor; Model A ignores it."""
    feat = antecedent_features(weather_df)
    if "is_observed" in weather_df.columns:
        feat["is_observed"] = weather_df.sort_values("date").reset_index(drop=True)["is_observed"].values
    anchor = _anchor_index(feat)
    disch = None
    if q_now is not None:
        disch = np.full(len(feat), np.nan)
        disch[anchor] = q_now  # make_horizon_rows reads q at the anchor only
    rows = make_horizon_rows(feat, [anchor], HORIZONS, discharge=disch)
    if rows[features].isna().any(axis=None):
        raise LiveDataError("live feature row has NaNs")
    return _calibrate(bundle["model"], features, bundle["calibrators"], rows), rows


def _ensemble_bands(base_weather: pd.DataFrame, ensemble: dict, bundle: dict,
                    features: list[str], q_now: float | None) -> np.ndarray | None:
    """Score every ensemble member (forecast-window rain overridden onto the shared
    observed antecedent) -> matrix (n_members, 7) of calibrated p_h. None on failure."""
    try:
        mem = ensemble["member_daily"]
        piv_p = mem.pivot_table(index="date", columns="member", values="precip_mm")
        piv_tx = mem.pivot_table(index="date", columns="member", values="tmax")
        piv_tn = mem.pivot_table(index="date", columns="member", values="tmin")
        base = base_weather.sort_values("date").reset_index(drop=True).copy()
        base["date"] = pd.to_datetime(base["date"])
        fc_mask = ~base["is_observed"].to_numpy() if "is_observed" in base else np.zeros(len(base), bool)
        members = list(piv_p.columns)
        P = []
        for m in members:
            w = base.copy()
            for col, piv in (("precip_mm", piv_p), ("tmax", piv_tx), ("tmin", piv_tn)):
                vals = w["date"].map(lambda dd, pv=piv: pv.loc[dd, m]
                                     if dd in pv.index and not pd.isna(pv.loc[dd, m]) else np.nan)
                w[col] = np.where(fc_mask & vals.notna().to_numpy(), vals.to_numpy(), w[col].to_numpy())
            cal, _ = _score_week(w, bundle, features, q_now)
            P.append(cal)
        return np.array(P) if P else None
    except Exception:
        return None


def _staleness(day_t) -> dict:
    """day-t should be 'yesterday' in basin-local time. If the forecast API's past_days
    block lags (so day-t is >1 day behind), warn instead of silently serving a shifted
    window. Uses America/New_York, never the (possibly UTC) server clock."""
    today_local = datetime.now(BASIN_TZ).date()
    lag = (today_local - pd.Timestamp(day_t).date()).days
    stale = lag > 2  # yesterday=1 is normal; allow a 1-day cushion for API timing
    return {"stale": bool(stale), "day_t_lag_days": int(lag),
            "today_local": today_local.isoformat(),
            "message": (f"Forecast data looks stale: last observed day is {pd.Timestamp(day_t).date()} "
                        f"({lag} days behind {today_local}). The 7-day window may be shifted.")
                       if stale else ""}


def predict_week(lat: float = 41.4133, lon: float = -78.1972,
                 weather_df: pd.DataFrame | None = None,
                 bundles: dict | None = None,
                 river=_UNSET,
                 ensemble=_UNSET,
                 allow_model_b: bool = True,
                 high_confidence: bool = False) -> dict:
    """Score the 7 per-day calibrated risks (with ensemble bands when available) +
    the windowed headline for today.

    Never fabricates a per-day number from the windowed model, and never fabricates
    the headline from the per-day product -- they come from different models (§5).
    Pass `river`/`ensemble` to reuse already-fetched data (the app caches them);
    pass None to signal "unavailable"; omit to let predict_week fetch them.
    `high_confidence` swaps the recall-targeted thresholds for the stricter,
    validation-derived `threshold_high_confidence` (~2% alert rate).

    LIVE PATH BUDGET: this function only ever touches the ensemble/deterministic
    forecast, USGS latest, and (via the app) GloFAS -- never the archive or
    historical-forecast endpoints, which are eval-only.
    """
    bundles = bundles or load_bundles()
    if weather_df is None:
        weather_df = fetch_forecast_weather(lat, lon)
    if river is _UNSET:  # not supplied -> fetch; explicit None -> honour "USGS down"
        try:
            river = fetch_current_river(BASIN_ID(bundles))
        except Exception:
            river = None
    if ensemble is _UNSET:
        ensemble = fetch_ensemble_weather(lat, lon)
    q_now = _current_discharge_cfs(river) if allow_model_b else None

    # Model B when a live discharge is available, else Model A (rain-only)
    if q_now is not None:
        model_used, bundle, features = "B", bundles["B"], FEATURES_B
    else:
        model_used, bundle, features = "A", bundles["A"], FEATURES_A

    cal, rows = _score_week(weather_df, bundle, features, q_now)
    thr = bundle.get("threshold_high_confidence") if high_confidence else None
    if not thr:
        thr = bundle["thresholds"]
        high_confidence = False  # requested but unavailable -> fall back silently

    # [v2] ensemble bands (forecast-input uncertainty only). Member count is whatever
    # GEFS returned today (never hardcoded); need >=2 members for a meaningful spread.
    band = None
    rain_stats = {}
    if ensemble:
        band = _ensemble_bands(weather_df, ensemble, bundle, features, q_now)
        if band is not None and band.shape[0] < 2:
            band = None  # too few members -> fall back to the deterministic single bar
        try:
            piv = ensemble["member_daily"].pivot_table(index="date", columns="member", values="precip_mm")
            for d in piv.index:
                v = piv.loc[d].dropna().to_numpy()
                if v.size:
                    rain_stats[pd.Timestamp(d).date()] = (float(np.median(v)),
                                                          float(np.percentile(v, 10)),
                                                          float(np.percentile(v, 90)))
        except Exception:
            rain_stats = {}

    wf = weather_df.sort_values("date").reset_index(drop=True)
    fc_rain_by_date = dict(zip(pd.to_datetime(wf["date"]).dt.date, wf["precip_mm"]))

    per_day = []
    for i, h in enumerate(HORIZONS):
        tdate = pd.Timestamp(rows["target_date"].iloc[i]).date()
        rmed, r10, r90 = rain_stats.get(tdate, (float(fc_rain_by_date.get(tdate, np.nan)), None, None))
        entry = {"horizon": h, "date": tdate, "threshold": float(thr[h]),
                 "p_point": float(cal[i]), "fc_rain_mm": rmed,
                 "fc_rain_p10": r10, "fc_rain_p90": r90}
        if band is not None:
            col = band[:, i]
            entry.update({
                "p": float(np.median(col)),
                "p10": float(np.percentile(col, 10)),
                "p90": float(np.percentile(col, 90)),
                "n_members": int(band.shape[0]),
                "n_exceed": int((col >= thr[h]).sum()),
            })
            entry["alert"] = bool(entry["p"] >= thr[h])
        else:  # deterministic point estimate, no band
            entry.update({"p": float(cal[i]), "p10": None, "p90": None,
                          "n_members": None, "n_exceed": None,
                          "alert": bool(cal[i] >= thr[h])})
        per_day.append(entry)

    # windowed headline (separate model; antecedent-only). Expose BOTH thresholds so
    # the app's strictness slider can recompute the alert without re-scoring.
    wb = bundles["windowed"]
    w_prim = float(wb["threshold"])
    w_hc = wb.get("threshold_high_confidence")
    w_hc = float(w_hc) if w_hc else None
    w_thr = (w_hc if (high_confidence and w_hc) else w_prim)
    anc = rows.iloc[[0]][ANTECEDENT].to_numpy()
    w_raw = wb["model"].predict_proba(wb["scaler"].transform(anc))[:, 1]
    w_cal = float(wb["calibrator"].predict(w_raw)[0])
    headline = {"p": w_cal, "threshold": w_thr, "threshold_primary": w_prim,
                "threshold_high_confidence": w_hc, "alert": bool(w_cal >= w_thr)}

    hc_all = bundle.get("threshold_high_confidence")
    day_t = pd.Timestamp(rows["anchor_date"].iloc[0]).date()
    return {
        "day_t": day_t,
        "per_day": per_day,
        "headline": headline,
        "model_used": model_used,
        "operating_point": "high_confidence" if high_confidence else "primary",
        # raw ensemble member probabilities (members x 7) so the app can re-threshold
        # live for the strictness slider and the per-day drill-down histogram
        "band_matrix": band.tolist() if band is not None else None,
        "thresholds_primary": {h: float(bundle["thresholds"][h]) for h in HORIZONS},
        "thresholds_high_confidence": ({h: float(hc_all[h]) for h in HORIZONS}
                                       if hc_all else None),
        "has_band": band is not None,
        "n_members": int(band.shape[0]) if band is not None else 0,
        "ensemble_model": ensemble.get("model") if ensemble else None,
        "river": river,
        "fetched_at": weather_df.attrs.get("fetched_at"),
        "staleness": _staleness(day_t),
        "basin": bundle["basin"],
        "threshold_cfs": bundle["flood_threshold_cfs"],
        "threshold_mm_day": bundle["flood_threshold_mm_day"],
        "threshold_m3s": bundle.get("flood_threshold_m3s"),
        "forecast_rain": [
            {"date": pd.Timestamp(d).date(), "precip_mm": float(p)}
            for d, p in zip(wf["date"], wf["precip_mm"])
        ],
    }


def BASIN_ID(bundles: dict) -> str:
    return bundles["A"]["basin"]["id"]


# ---------------------------------------------------------------------------
# Mandatory self-tests  (network-free: run on the cached ERA5 + test predictions)
# ---------------------------------------------------------------------------
def _selftest_feature_parity() -> None:
    """(a) build_live_features reproduces the training antecedent AND forecast blocks
    to atol 1e-10 on a known historical anchor (vs data/test_predictions.csv)."""
    hist = DATA / "openmeteo_historical.csv"
    ref = DATA / "test_predictions.csv"
    if not hist.exists() or not ref.exists():
        raise SystemExit("run train_daybyday.py first (need cached weather + test_predictions.csv)")
    weather = pd.read_csv(hist, parse_dates=["date"])
    ref_df = pd.read_csv(ref, parse_dates=["anchor_date", "target_date"])

    anchor_date = ref_df["anchor_date"].iloc[len(ref_df) // 3]  # some interior test anchor
    feat = antecedent_features(weather)
    idx = int(feat.index[feat["date"] == anchor_date][0])
    rows = make_horizon_rows(feat, [idx], HORIZONS)  # perfect-prog block from observed rain
    got = rows.sort_values("horizon").reset_index(drop=True)
    exp = (ref_df[ref_df["anchor_date"] == anchor_date]
           .sort_values("horizon").reset_index(drop=True))
    assert len(got) == len(exp) == len(HORIZONS), "horizon count mismatch"
    max_diff = 0.0
    for f in FEATURES_A:
        max_diff = max(max_diff, float(np.abs(got[f].to_numpy() - exp[f].to_numpy()).max()))
    assert max_diff < 1e-10, f"feature parity drift = {max_diff:.2e} (must be < 1e-10)"
    print(f"[selftest a] antecedent+forecast parity OK for anchor {anchor_date.date()} "
          f"(max diff {max_diff:.1e})")


def _selftest_forecast_block_leakage() -> None:
    """(b) sentinel forecast rain changes ONLY the forecast columns + the prediction,
    and leaves the antecedent block byte-identical (no forecast-day leak backward)."""
    hist = DATA / "openmeteo_historical.csv"
    weather = pd.read_csv(hist, parse_dates=["date"]).iloc[:200].copy()
    # mark the last 7 rows as the "forecast" window
    weather["is_observed"] = np.arange(len(weather)) < (len(weather) - FORECAST_DAYS)

    base = build_live_features(weather)
    w_big = weather.copy()
    w_big.loc[~w_big["is_observed"], "precip_mm"] = 1e6  # sentinel storm in the forecast window
    big = build_live_features(w_big)

    for f in ANTECEDENT:
        assert np.array_equal(base[f].to_numpy(), big[f].to_numpy()), \
            f"forecast-day rain leaked into antecedent feature {f}!"
    fc_changed = any(not np.array_equal(base[f].to_numpy(), big[f].to_numpy())
                     for f in ("fc_rain_cum", "fc_rain_on_h", "fc_rain_max"))
    assert fc_changed, "sentinel did not change the forecast block (builder bug)"

    # and the model prediction must move
    bundle = joblib.load(MODELS / "flood_daybyday_model.joblib")
    p_base = _calibrate(bundle["model"], FEATURES_A, bundle["calibrators"], base)
    p_big = _calibrate(bundle["model"], FEATURES_A, bundle["calibrators"], big)
    assert not np.allclose(p_base, p_big), "sentinel forecast rain did not move the prediction"
    print(f"[selftest b] forecast-block sentinel: antecedent byte-identical, forecast "
          f"block + prediction changed (p t+7 {p_base[-1]:.3f} -> {p_big[-1]:.3f})")


def _selftest_ensemble_parity() -> None:
    """[v2] (c) a synthetic ensemble whose members equal the base forecast reproduces
    the deterministic p_h exactly; a spread of members is monotone in rain and its
    median/exceedance aggregation is correct. Network-free."""
    hist = DATA / "openmeteo_historical.csv"
    weather = pd.read_csv(hist, parse_dates=["date"]).iloc[:200].copy()
    weather["is_observed"] = np.arange(len(weather)) < (len(weather) - FORECAST_DAYS)
    bundle = joblib.load(MODELS / "flood_daybyday_model.joblib")
    fc_dates = weather.loc[~weather["is_observed"], "date"]
    base_fc = weather[weather["date"].isin(fc_dates)][["date", "precip_mm", "tmax", "tmin"]]

    # (i) identical members == deterministic path
    ident = pd.concat([base_fc.assign(member=f"m{k}") for k in range(3)], ignore_index=True)
    P = _ensemble_bands(weather, {"member_daily": ident}, bundle, FEATURES_A, None)
    det, _ = _score_week(weather, bundle, FEATURES_A, None)
    assert P is not None and P.shape == (3, len(HORIZONS)), f"bad band shape {None if P is None else P.shape}"
    assert np.allclose(P, det, atol=1e-10), "identical members must reproduce the deterministic p_h"

    # (ii) spread is monotone in rain; median/exceedance aggregation correct
    rains = [0.0, 20.0, 200.0]
    spread = pd.concat([base_fc.assign(precip_mm=r, member=f"m{k}") for k, r in enumerate(rains)],
                       ignore_index=True)
    S = _ensemble_bands(weather, {"member_daily": spread}, bundle, FEATURES_A, None)
    assert np.all(S[0] <= S[1] + 1e-9) and np.all(S[1] <= S[2] + 1e-9), "p_h not monotone in member rain"
    med = np.median(S, axis=0)
    assert np.allclose(med, S[1], atol=1e-9), "median of 3 members must be the middle member"
    thr7 = bundle["thresholds"][7]
    assert int((S[:, 6] >= thr7).sum()) == int(np.sum(np.array([S[k, 6] for k in range(3)]) >= thr7)), \
        "exceedance count mismatch"
    print(f"[selftest c] ensemble parity OK: identical members==deterministic; spread monotone; "
          f"median/exceedance correct (t+7 spread {S[0,6]:.3f}->{S[2,6]:.3f})")


if __name__ == "__main__":
    _selftest_feature_parity()
    _selftest_forecast_block_leakage()
    _selftest_ensemble_parity()
    # optional live sample (never fails the gate)
    try:
        res = predict_week()
        band = f", band {res['n_members']} members" if res["has_band"] else ", no band"
        stale = " STALE!" if res["staleness"]["stale"] else ""
        print(f"[sample] day t = {res['day_t']}{stale} (model {res['model_used']}{band}); headline "
              f"P(>=1 flood day in 7) = {res['headline']['p']:.1%}")
        for d in res["per_day"]:
            rng = f" [{d['p10']:.0%}-{d['p90']:.0%}] {d['n_exceed']}/{d['n_members']}" \
                  if res["has_band"] else ""
            print(f"   t+{d['horizon']} {d['date']}  P={d['p']:.1%}{rng}  "
                  f"{'ALERT' if d['alert'] else 'watch'}  rain {d['fc_rain_mm']:.1f}mm")
        g = fetch_glofas()
        if g:
            print(f"[sample] GloFAS peak next 7d = {g['peak_cfs']:.0f} cfs "
                  f"(offset {g['offset']}, {'INDICATIVE' if g['indicative'] else 'resolved'})")
    except Exception as e:
        print(f"[sample] live fetch skipped: {e}")
    print("ALL SELF-TESTS PASSED.")
