"""
[v2 + final] Extra evaluation for the day-by-day flood forecaster.

Final (go-live) pass hardens how three already-measured results are presented:
  F1.1 Event eval reported at TWO operating points (recall-targeted "primary" and a
       stricter validation-derived "high-confidence" ~2% alert rate), always beside the
       day-level false-alarm ratio AND the alert frequency; lead-time as a histogram
       with the 7-day cap stated.
  F1.2 Real-forecast backtest reported PER HORIZON (PR-AUC + recall), perfect-prog vs
       real archived forecast -- the money plot.
  F1.3 GloFAS compared at a MATCHED operating point (POD-vs-FAR, equal-FAR read-off).

It also derives+persists `threshold_high_confidence` (per horizon) into the bundles --
on the VALIDATION fold only, never the test set. The core model is NOT retrained.

Run:  venv\\Scripts\\python.exe evaluate_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from features import (ANTECEDENT, FEATURES_A, FEATURES_B, FEATURES_WINDOWED, HORIZONS,
                      antecedent_features, make_horizon_rows)
from train_daybyday import load_data, build_pooled, split_masks, BASIN
import live_data as ld

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIG = ROOT / "figures"
CFS_PER_M3S = 35.3147
TARGET_ALERT_HC = 0.02  # high-confidence operating point: alert on ~2% of days (matches
#                         the flood base rate and GloFAS's 98th-pct exceedance)


def _calibrate_rows(bundle, rows, features=None):
    features = features or bundle["features"]
    raw = bundle["model"].predict_proba(rows[features].to_numpy())[:, 1]
    out = np.empty_like(raw, dtype=float)
    hz = rows["horizon"].to_numpy()
    for h in HORIZONS:
        m = hz == h
        out[m] = bundle["calibrators"][h].predict(raw[m])
    return out


# ===========================================================================
#  Derive + persist the high-confidence thresholds (VALIDATION fold only)
# ===========================================================================
def patch_high_confidence_thresholds(pooled, masks):
    """Add `threshold_high_confidence` (per horizon, ~2% validation alert rate) to
    bundles A & B, and a scalar to the windowed bundle. Validation-only; no retrain."""
    val = pooled[masks["val"]]
    for name, feats in (("flood_daybyday_model.joblib", FEATURES_A),
                        ("flood_daybyday_model_B.joblib", FEATURES_B)):
        b = joblib.load(MODELS / name)
        cal = _calibrate_rows(b, val, feats)
        hz = val["horizon"].to_numpy()
        thr_hc = {}
        for h in HORIZONS:
            s = cal[hz == h]
            thr_hc[h] = float(np.quantile(s, 1 - TARGET_ALERT_HC)) if s.size else 1.0
        b["threshold_high_confidence"] = thr_hc
        joblib.dump(b, MODELS / name)
    # windowed headline
    wb = joblib.load(MODELS / "flood_windowed_model.joblib")
    win = (pooled.groupby("anchor_date").agg(
        flood_within7=("flood_on_day", "max"),
        **{c: (c, "first") for c in FEATURES_WINDOWED}).reset_index())
    wmask = split_masks(win["anchor_date"])
    wv = win[wmask["val"]]
    wraw = wb["model"].predict_proba(wb["scaler"].transform(wv[FEATURES_WINDOWED].to_numpy()))[:, 1]
    wcal = wb["calibrator"].predict(wraw)
    wb["threshold_high_confidence"] = float(np.quantile(wcal, 1 - TARGET_ALERT_HC))
    joblib.dump(wb, MODELS / "flood_windowed_model.joblib")
    print(f"[patch] threshold_high_confidence added (val ~{TARGET_ALERT_HC:.0%} alert rate) to A/B/windowed")
    return joblib.load(MODELS / "flood_daybyday_model.joblib")


# ===========================================================================
# 6.2  EVENT-BASED EVALUATION (offline) -- two operating points
# ===========================================================================
def find_events(flood_dates, gap: int = 1):
    if not flood_dates:
        return []
    d = sorted(pd.Timestamp(x) for x in flood_dates)
    events, start, prev = [], d[0], d[0]
    for cur in d[1:]:
        if (cur - prev).days <= gap + 1:
            prev = cur
        else:
            events.append({"onset": start, "end": prev, "length": (prev - start).days + 1})
            start = prev = cur
    events.append({"onset": start, "end": prev, "length": (prev - start).days + 1})
    return events


def _event_metrics(tp, thr_map, truth, events):
    fired = tp["p_h"].to_numpy() >= np.array([thr_map[int(h)] for h in tp["horizon"]])
    alert = {(a, int(h)): bool(f) for a, h, f in zip(tp["anchor_date"], tp["horizon"], fired)}
    rows = []
    for ev in events:
        D = ev["onset"]
        caught = [L for L in HORIZONS if alert.get((D - pd.Timedelta(days=L), L), False)]
        rows.append({"onset": D.date(), "length": ev["length"], "caught": bool(caught),
                     "max_lead": max(caught) if caught else 0,
                     "caught_le3": any(L <= 3 for L in caught)})
    ev_df = pd.DataFrame(rows)
    n = len(ev_df)
    leads = ev_df.loc[ev_df["caught"], "max_lead"]
    warn = pd.Series({a: any(alert.get((a, L), False) for L in HORIZONS)
                      for a in tp["anchor_date"].unique()})
    win_flood = pd.Series({a: any(truth.get(a + pd.Timedelta(days=L), 0) == 1 for L in HORIZONS)
                           for a in warn.index})
    hits = int((warn & win_flood).sum()); miss = int((~warn & win_flood).sum())
    fa = int((warn & ~win_flood).sum())
    return {
        "n_events": n,
        "POD_7day": float(ev_df["caught"].mean()) if n else float("nan"),
        "POD_lead<=3": float(ev_df["caught_le3"].mean()) if n else float("nan"),
        "median_lead_days": float(leads.median()) if len(leads) else float("nan"),
        "alert_frequency": float(warn.mean()),
        "false_alarm_ratio": fa / (hits + fa) if (hits + fa) else float("nan"),
        "CSI": hits / (hits + miss + fa) if (hits + miss + fa) else float("nan"),
        "hits": hits, "misses": miss, "false_alarms": fa,
        "lead_hist": {int(L): int((leads == L).sum()) for L in HORIZONS},
    }, ev_df


def event_evaluation(bundle):
    tp = pd.read_csv(DATA / "test_predictions.csv", parse_dates=["anchor_date", "target_date"])
    truth = (tp[["target_date", "flood_on_day"]].drop_duplicates("target_date")
             .set_index("target_date")["flood_on_day"])
    events = find_events(list(truth[truth == 1].index))
    prim, ev_df = _event_metrics(tp, bundle["thresholds"], truth, events)
    hc, _ = _event_metrics(tp, bundle["threshold_high_confidence"], truth, events)
    ev_df.to_csv(DATA / "event_eval.csv", index=False)
    print(f"[event] {prim['n_events']} events | PRIMARY POD(7d)={prim['POD_7day']:.2f} "
          f"FAR={prim['false_alarm_ratio']:.2f} alert-freq={prim['alert_frequency']:.2f} "
          f"CSI={prim['CSI']:.2f}")
    print(f"[event] HIGH-CONF POD(7d)={hc['POD_7day']:.2f} FAR={hc['false_alarm_ratio']:.2f} "
          f"alert-freq={hc['alert_frequency']:.2f} CSI={hc['CSI']:.2f} "
          f"median-lead={hc['median_lead_days']:.0f}d")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(HORIZONS))
    a1.bar(x - .2, [prim["lead_hist"][L] for L in HORIZONS], .4, label="primary", color="#4575b4")
    a1.bar(x + .2, [hc["lead_hist"][L] for L in HORIZONS], .4, label="high-confidence", color="#d73027")
    a1.set_xticks(x); a1.set_xticklabels([str(L) for L in HORIZONS])
    a1.set_xlabel("earliest warning lead (days, capped at 7)"); a1.set_ylabel("# events")
    a1.set_title("Lead-time distribution (mass at 7 = alert already on)"); a1.legend(fontsize=8)
    a1.grid(axis="y", alpha=.3)
    labels = ["POD(7d)", "false-alarm\nratio", "alert\nfrequency", "CSI"]
    pv = [prim["POD_7day"], prim["false_alarm_ratio"], prim["alert_frequency"], prim["CSI"]]
    hv = [hc["POD_7day"], hc["false_alarm_ratio"], hc["alert_frequency"], hc["CSI"]]
    a2.bar(np.arange(4) - .2, pv, .4, label="primary (recall-targeted)", color="#4575b4")
    a2.bar(np.arange(4) + .2, hv, .4, label="high-confidence (~2% alert)", color="#d73027")
    a2.set_xticks(range(4)); a2.set_xticklabels(labels, fontsize=8); a2.set_ylim(0, 1.05)
    a2.set_title("Detection vs false alarms: the operating-point tradeoff"); a2.legend(fontsize=8)
    a2.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "event_evaluation.png", dpi=130); plt.close(fig)
    return {"primary": prim, "high_confidence": hc}


# ===========================================================================
# 6.4  GLOFAS BENCHMARK (network) + matched operating point
# ===========================================================================
def glofas_benchmark(df, threshold_cfs):
    import requests
    lat, lon = BASIN["lat"], BASIN["lon"]
    usgs_mean_m3s = float(df.loc[df["date"].dt.year.between(2005, 2014), "q_cfs"].mean() / CFS_PER_M3S)

    def fetch(la, lo):
        try:
            r = requests.get(ld.GLOFAS_URL, params={
                "latitude": round(la, 4), "longitude": round(lo, 4), "daily": "river_discharge",
                "start_date": "2005-01-01", "end_date": "2014-12-31"}, timeout=90)
            r.raise_for_status(); d = r.json()["daily"]
            return pd.DataFrame({"date": pd.to_datetime(d["time"]),
                                 "q_m3s": pd.to_numeric(pd.Series(d["river_discharge"]), errors="coerce")})
        except Exception:
            return None

    base = fetch(lat, lon)
    if base is None:
        print("[glofas] reanalysis fetch failed -> benchmark skipped"); return None
    base_mean = float(base["q_m3s"].mean())
    grid = []
    for dla in (0.0, 0.1, -0.1):
        for dlo in (0.0, 0.1, -0.1):
            g = base if (dla == 0 and dlo == 0) else fetch(lat + dla, lon + dlo)
            if g is not None:
                grid.append((abs(g["q_m3s"].mean() - usgs_mean_m3s), dla, dlo, float(g["q_m3s"].mean()), g))
    grid.sort(key=lambda x: x[0])
    _, dlat, dlon, chosen_mean, gdf = grid[0]
    indicative = (dlat == 0 and dlon == 0) or not (0.3 <= chosen_mean / max(usgs_mean_m3s, 1e-6) <= 3.0)
    (DATA / "glofas_config.json").write_text(json.dumps(
        {"dlat": dlat, "dlon": dlon, "usgs_mean_m3s": usgs_mean_m3s,
         "base_coord_mean_m3s": base_mean, "chosen_mean_m3s": chosen_mean,
         "indicative": bool(indicative)}, indent=2))
    print(f"[glofas] resolution: base {base_mean:.2f} m3/s vs USGS {usgs_mean_m3s:.2f} -> "
          f"offset ({dlat:+.1f},{dlon:+.1f}) {chosen_mean:.2f} m3/s "
          f"({'INDICATIVE' if indicative else 'resolved'})")

    g = gdf.dropna().copy()
    g_thr = float(np.percentile(g["q_m3s"], 98))
    g["glofas_flood"] = (g["q_m3s"] >= g_thr).astype(int)
    lab = df[["date", "q_cfs"]].copy(); lab["usgs_flood"] = (lab["q_cfs"] >= threshold_cfs).astype(int)
    m = g.merge(lab, on="date", how="inner"); m["glofas_q_cfs"] = m["q_m3s"] * CFS_PER_M3S
    m[["date", "glofas_q_cfs", "glofas_flood", "usgs_flood"]].to_csv(DATA / "glofas_benchmark.csv", index=False)
    pod = recall_score(m["usgs_flood"], m["glofas_flood"], zero_division=0)
    far = 1 - precision_score(m["usgs_flood"], m["glofas_flood"], zero_division=0)
    alert_freq = float(m["glofas_flood"].mean())
    print(f"[glofas] reanalysis-vs-USGS (per-day): POD={pod:.2f} FAR={far:.2f} "
          f"alert-freq={alert_freq:.2f} (reanalysis, NOT a forecast)")

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(m["date"], m["glofas_q_cfs"], lw=.5, color="#2166ac", label="GloFAS reanalysis (cfs)")
    ax.axhline(threshold_cfs, color="#a50026", ls="--", lw=1, label=f"USGS flood thr {threshold_cfs:.0f} cfs")
    ax.axhline(g_thr * CFS_PER_M3S, color="#f46d43", ls=":", lw=1,
               label=f"GloFAS 98th pct {g_thr*CFS_PER_M3S:.0f} cfs")
    ax.set_yscale("log"); ax.set_title("GloFAS reanalysis vs USGS flood days (2005-2014, coarse ~5km)")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "glofas_benchmark.png", dpi=130); plt.close(fig)
    return {"offset": [dlat, dlon], "indicative": bool(indicative), "usgs_mean_m3s": usgs_mean_m3s,
            "base_coord_mean_m3s": base_mean, "chosen_mean_m3s": chosen_mean,
            "glofas_thr_cfs": g_thr * CFS_PER_M3S,
            "vs_usgs": {"POD": pod, "FAR": far, "alert_freq": alert_freq}, "flavor": "reanalysis"}


def glofas_matched(bundle):
    """F1.3 -- compare the model vs GloFAS-reanalysis at a MATCHED operating point.
    Uses the model's shortest-lead (h=1) per-day flood detector so it is comparable to
    GloFAS's per-day discharge. Plots POD-vs-FAR and reads model POD at GloFAS's FAR."""
    gb = DATA / "glofas_benchmark.csv"
    tp = pd.read_csv(DATA / "test_predictions.csv", parse_dates=["target_date"])
    if not gb.exists():
        return None
    gdf = pd.read_csv(gb, parse_dates=["date"])
    h1 = tp[tp["horizon"] == 1][["target_date", "p_h", "flood_on_day"]].rename(columns={"target_date": "date"})
    m = h1.merge(gdf[["date", "glofas_flood", "usgs_flood"]], on="date", how="inner").dropna()
    y = m["usgs_flood"].to_numpy()
    # GloFAS operating point
    g_pod = recall_score(y, m["glofas_flood"], zero_division=0)
    g_far = 1 - precision_score(y, m["glofas_flood"], zero_division=0)
    # model POD-FAR sweep on h=1 calibrated score
    ts = np.unique(np.quantile(m["p_h"], np.linspace(0, 1, 200)))
    pods, fars = [], []
    for t in ts:
        pred = (m["p_h"].to_numpy() >= t).astype(int)
        pods.append(recall_score(y, pred, zero_division=0))
        fars.append(1 - precision_score(y, pred, zero_division=0))
    pods, fars = np.array(pods), np.array(fars)
    # model POD at GloFAS's FAR (equal-FAR read-off)
    order = np.argsort(fars)
    model_pod_at_gfar = float(np.interp(g_far, fars[order], pods[order]))
    # model operating points
    def pt(thr):
        pred = (m["p_h"].to_numpy() >= thr).astype(int)
        return (float(recall_score(y, pred, zero_division=0)),
                float(1 - precision_score(y, pred, zero_division=0)), float(pred.mean()))
    prim_pod, prim_far, prim_af = pt(bundle["thresholds"][1])
    hc_pod, hc_far, hc_af = pt(bundle["threshold_high_confidence"][1])
    print(f"[glofas-match] equal-FAR: GloFAS POD={g_pod:.2f}@FAR={g_far:.2f}(af={m['glofas_flood'].mean():.2f}) "
          f"vs model POD={model_pod_at_gfar:.2f}@same FAR | model high-conf POD={hc_pod:.2f} "
          f"FAR={hc_far:.2f} af={hc_af:.2f}")

    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.plot(fars[order], pods[order], "-", color="#2166ac", label="model (h=1) sweep")
    ax.scatter([g_far], [g_pod], color="#000", zorder=5, s=60, marker="D",
               label=f"GloFAS reanalysis (af {m['glofas_flood'].mean():.2f})")
    ax.scatter([prim_far], [prim_pod], color="#4575b4", zorder=5, s=50,
               label=f"model primary (af {prim_af:.2f})")
    ax.scatter([hc_far], [hc_pod], color="#d73027", zorder=5, s=50,
               label=f"model high-conf (af {hc_af:.2f})")
    ax.axvline(g_far, color="#888", ls=":", lw=1)
    ax.set_xlabel("false-alarm ratio  (1 − precision)"); ax.set_ylabel("POD (recall)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=7)
    ax.set_title("Matched operating point: model (h=1) vs GloFAS reanalysis")
    fig.tight_layout(); fig.savefig(FIG / "glofas_matched.png", dpi=130); plt.close(fig)
    return {"n_days": int(len(m)), "glofas": {"POD": g_pod, "FAR": g_far,
            "alert_freq": float(m["glofas_flood"].mean())},
            "model_pod_at_glofas_far": model_pod_at_gfar,
            "model_primary": {"POD": prim_pod, "FAR": prim_far, "alert_freq": prim_af},
            "model_high_conf": {"POD": hc_pod, "FAR": hc_far, "alert_freq": hc_af}}


# ===========================================================================
# 6.3  REAL-FORECAST BACKTEST (network) -- per-horizon PR-AUC + recall
# ===========================================================================
def real_forecast_backtest(bundle, threshold_cfs, start="2022-01-01", end="2024-12-24"):
    lat, lon = BASIN["lat"], BASIN["lon"]
    obs_start = (pd.Timestamp(start) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    obs_end = (pd.Timestamp(end) + pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    obs = ld.fetch_archive_weather(lat, lon, obs_start, obs_end)
    fc = ld.fetch_historical_forecast(lat, lon, start, obs_end)
    usgs = ld.fetch_usgs_daily(BASIN["id"], start, obs_end)
    if obs is None or fc is None or usgs is None:
        print(f"[realfc] skipped (obs={obs is not None}, fc={fc is not None}, usgs={usgs is not None})")
        return None
    obs = obs.dropna().sort_values("date").reset_index(drop=True)
    feat_obs = antecedent_features(obs)
    fc_p = {d: v for d, v in zip(fc["date"], fc["precip_mm"]) if not pd.isna(v)}
    fc_t = {d: v for d, v in zip(fc["date"], fc["tmax"]) if not pd.isna(v)}
    obs_p = dict(zip(feat_obs["date"], feat_obs["precip_mm"]))
    obs_t = dict(zip(feat_obs["date"], feat_obs["tmax"]))
    feat_real = feat_obs.copy()
    feat_real["precip_mm"] = feat_real["date"].map(lambda d: fc_p.get(d, obs_p.get(d)))
    feat_real["tmax"] = feat_real["date"].map(lambda d: fc_t.get(d, obs_t.get(d)))
    n = len(feat_obs); ad = pd.to_datetime(feat_obs["date"])
    valid = np.where((ad >= pd.Timestamp(start)) & (ad <= pd.Timestamp(end)))[0]
    anchors = np.array([i for i in valid if i >= 29 and i + max(HORIZONS) < n])
    rp = make_horizon_rows(feat_obs, anchors, HORIZONS)
    rr = make_horizon_rows(feat_real, anchors, HORIZONS)
    qmap = dict(zip(usgs["date"], usgs["q_cfs"]))
    rp["q_target"] = pd.to_datetime(rp["target_date"]).map(qmap)
    y = (rp["q_target"] >= threshold_cfs).astype(float)
    ok = (~rr[FEATURES_A].isna().any(axis=1).to_numpy() & ~rp[FEATURES_A].isna().any(axis=1).to_numpy()
          & rp["q_target"].notna().to_numpy())
    rp, rr, yy = rp[ok].copy(), rr[ok].copy(), y[ok].to_numpy()
    p_perf = _calibrate_rows(bundle, rp, FEATURES_A); p_real = _calibrate_rows(bundle, rr, FEATURES_A)
    thr = bundle["thresholds"]
    out = []
    for h in HORIZONS:
        mh = rp["horizon"].to_numpy() == h; yh = yy[mh]
        rec = lambda p: float(recall_score(yh, (p[mh] >= thr[h]).astype(int), zero_division=0))
        rec_ok = 0 < yh.sum() < len(yh)
        out.append({"horizon": h, "n": int(mh.sum()), "n_pos": int(yh.sum()),
                    "pr_auc_perfect": float(average_precision_score(yh, p_perf[mh])) if rec_ok else np.nan,
                    "pr_auc_real": float(average_precision_score(yh, p_real[mh])) if rec_ok else np.nan,
                    "recall_perfect": rec(p_perf) if yh.sum() else np.nan,
                    "recall_real": rec(p_real) if yh.sum() else np.nan})
    bt = pd.DataFrame(out); bt.to_csv(DATA / "real_forecast_backtest.csv", index=False)
    common = fc.merge(obs, on="date", suffixes=("_fc", "_obs")).dropna()
    rain_corr = float(np.corrcoef(common["precip_mm_fc"], common["precip_mm_obs"])[0, 1]) if len(common) > 10 else np.nan
    n_pos = int(yy.sum())
    print(f"[realfc] {start}..{end}: {int(ok.sum())} rows, {n_pos} flood targets | agg PR-AUC "
          f"perfect {np.nanmean(bt['pr_auc_perfect']):.3f} vs real {np.nanmean(bt['pr_auc_real']):.3f} "
          f"| rain r={rain_corr:.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    a1.plot(bt["horizon"], bt["pr_auc_perfect"], "s-", color="#1a9850", label="perfect-prog")
    a1.plot(bt["horizon"], bt["pr_auc_real"], "o--", color="#d73027", label="real forecast")
    a1.set_ylabel("PR-AUC"); a1.set_title("PR-AUC per lead day"); a1.set_ylim(0, 1)
    a1.legend(fontsize=8); a1.grid(alpha=.3)
    a2.plot(bt["horizon"], bt["recall_perfect"], "s-", color="#1a9850", label="perfect-prog")
    a2.plot(bt["horizon"], bt["recall_real"], "o--", color="#d73027", label="real forecast")
    a2.set_ylabel("recall @ primary thr"); a2.set_title("Recall per lead day"); a2.set_ylim(0, 1.02)
    a2.legend(fontsize=8); a2.grid(alpha=.3)
    for a in (a1, a2):
        a.set_xlabel("lead day (t+h)")
    fig.suptitle(f"Real-forecast backtest {start[:4]}-{end[:4]} (n_pos={n_pos}; single basin; indicative)")
    fig.tight_layout(); fig.savefig(FIG / "real_forecast_backtest.png", dpi=130); plt.close(fig)
    return {"window": [start, end], "n_rows": int(ok.sum()), "n_pos": n_pos,
            "archived_fc_vs_obs_rain_corr": rain_corr, "per_horizon": bt.to_dict(orient="records"),
            "note": (f"Real archived forecasts (rain corr {rain_corr:.2f} with observed) vs the "
                     "perfect-prog proxy on the same anchors/labels. Small, single-basin, recent "
                     "sample -> indicative; the synthetic degradation test spans all leads on 2005-2014.")}


# ===========================================================================
def main():
    print("=" * 74); print("v2 + FINAL EXTRA EVALUATION"); print("=" * 74)
    df = load_data()
    bundle0 = joblib.load(MODELS / "flood_daybyday_model.joblib")
    thr_cfs = bundle0["flood_threshold_cfs"]
    pooled = build_pooled(df, thr_cfs); masks = split_masks(pooled["anchor_date"])
    bundle = patch_high_confidence_thresholds(pooled, masks)

    summary = {"event": event_evaluation(bundle),
               "target_alert_high_confidence": TARGET_ALERT_HC}
    summary["glofas"] = glofas_benchmark(df, thr_cfs)
    summary["glofas_matched"] = glofas_matched(bundle) if summary["glofas"] else None
    summary["real_forecast"] = real_forecast_backtest(bundle, thr_cfs)
    (DATA / "v2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n[v2] wrote v2_summary.json + event/glofas/glofas_matched/real_forecast CSVs & figures")
    print("v2 + FINAL EVALUATION DONE.")


if __name__ == "__main__":
    main()
