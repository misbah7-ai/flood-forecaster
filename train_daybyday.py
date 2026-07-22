"""
Train the DAY-BY-DAY 7-day flood forecaster for USGS gauge 01543000.

Products
--------
* Model A  -- pooled multi-horizon HistGradientBoosting -> per-day P(flood) for
              each of t+1..t+7. Forecast block trained on OBSERVED rain
              (perfect-prog); served on the real Open-Meteo forecast.
* Model B  -- Model A + today's (backward-looking) discharge q_t; sharper near
              horizons; served only when USGS is up (falls back to A).
* Windowed -- balanced LogisticRegression on the retained flood_within7 label,
              isotonic-calibrated; supplies the HONEST ">=1 flood day in 7" headline.
* LR pooled baseline for coefficient interpretability.

Discipline (all enforced/asserted below)
  - Flood threshold = 98th pct of q on TRAIN YEARS 1980-2004 only.
  - Per-day label flood_on_day[t+h] = q[t+h] >= threshold.
  - Split by ANCHOR DAY, chronologically, with a 7-day embargo (drop 2004-12-25..31).
  - Anchor-day-grouped time CV (all 7 rows of a day stay in one fold).
  - Per-HORIZON isotonic calibration + per-horizon operating threshold, both on
    the VALIDATION slice only (last 3 training years), never on test.

Run:  venv\\Scripts\\python.exe train_daybyday.py
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import (
    ANTECEDENT, FORECAST, HORIZON, FEATURES_A, FEATURES_B, FEATURES_WINDOWED,
    HORIZONS, EWMA_ALPHA, Q_EWMA_ALPHA,
    antecedent_features, make_horizon_rows, monotonic_cst,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
MODELS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET_RECALL = 0.80  # per-horizon operating point chosen on calibrated validation
ONE_DAY_MODEL_ROC_AUC = 0.943  # published 1-day model, for honest comparison
OLD_WINDOWED_ROC_AUC = 0.79    # prior "flood within 7 days" build

BASIN = {
    "id": "01543000",
    "name": "Driftwood Branch Sinnemahoning Creek at Sterling Run, PA",
    "lat": 41.4133, "lon": -78.1972, "area_km2": 705,
}

# chronological split boundaries (by anchor day = day t)
FIT_END = pd.Timestamp("2001-12-31")        # model fit set: 1980-01-30 .. 2001-12-31
VAL_START = pd.Timestamp("2002-01-01")      # validation (calibration + thresholds)
VAL_END = pd.Timestamp("2004-12-24")        # last train anchor before the embargo
EMBARGO_START = pd.Timestamp("2004-12-25")  # 7-day embargo (== max horizon)
EMBARGO_END = pd.Timestamp("2004-12-31")
TEST_START = pd.Timestamp("2005-01-01")


# ---------------------------------------------------------------------------
def cfs_to_mm_per_day(cfs: float, area_km2: float) -> float:
    """Convert a discharge in cubic feet/s to a basin-average runoff in mm/day."""
    m3s = cfs * 0.028316846592
    area_m2 = area_km2 * 1e6
    return m3s * 86400.0 / area_m2 * 1000.0


def load_data() -> pd.DataFrame:
    """Return a daily frame: date, precip_mm, tmax, tmin, q_cfs (aligned, gap-free)."""
    weather = pd.read_csv(DATA / "openmeteo_historical.csv", parse_dates=["date"])
    q = pd.read_csv(
        DATA / "01543000_streamflow_qc.txt", sep=r"\s+", header=None,
        names=["id", "year", "month", "day", "q_cfs", "flag"],
    )
    q["date"] = pd.to_datetime(q[["year", "month", "day"]])
    df = weather.merge(q[["date", "q_cfs"]], on="date", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    # integrity checks
    gaps = df["date"].diff().dropna().dt.days
    assert (gaps == 1).all(), "weather/discharge series is not gap-free daily"
    assert df[["precip_mm", "tmax", "tmin", "q_cfs"]].isna().sum().sum() == 0, "missing values"
    return df


# ---------------------------------------------------------------------------
def build_pooled(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build the pooled (anchor x horizon) dataset with the per-day flood label."""
    feat = antecedent_features(df)  # adds ANTECEDENT columns
    n = len(feat)
    max_h = max(HORIZONS)
    # valid anchor: full 30-day antecedent window (idx>=29) AND t+7 in range
    first_valid = 29
    anchors = np.arange(first_valid, n - max_h)
    pooled = make_horizon_rows(feat, anchors, HORIZONS, discharge=df["q_cfs"].to_numpy())
    pooled["flood_on_day"] = (pooled["q_target"] >= threshold).astype(int)
    pooled["anchor_date"] = pd.to_datetime(pooled["anchor_date"])
    pooled["target_date"] = pd.to_datetime(pooled["target_date"])
    return pooled


def split_masks(anchor_date: pd.Series) -> dict:
    """Chronological split by anchor day, with the 7-day 2004/2005 embargo."""
    d = pd.to_datetime(anchor_date)
    fit = d <= FIT_END
    val = (d >= VAL_START) & (d <= VAL_END)
    embargo = (d >= EMBARGO_START) & (d <= EMBARGO_END)
    test = d >= TEST_START
    return {"fit": fit.to_numpy(), "val": val.to_numpy(),
            "embargo": embargo.to_numpy(), "test": test.to_numpy()}


# ---------------------------------------------------------------------------
def anchor_day_cv_score(pooled: pd.DataFrame, mask_fit: np.ndarray,
                        features: list[str], params: dict, n_splits: int = 4) -> float:
    """Expanding-window time CV on UNIQUE ANCHOR DAYS (all 7 rows of a day stay
    together). Returns mean average_precision. Asserts no anchor straddles a split."""
    sub = pooled[mask_fit]
    days = np.sort(sub["anchor_date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aps = []
    for tr_idx, va_idx in tscv.split(days):
        tr_days, va_days = set(days[tr_idx]), set(days[va_idx])
        assert tr_days.isdisjoint(va_days), "anchor day leaked across a CV split!"
        tr = sub[sub["anchor_date"].isin(tr_days)]
        va = sub[sub["anchor_date"].isin(va_days)]
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, monotonic_cst=monotonic_cst(features), **params)
        sw = compute_sample_weight("balanced", tr["flood_on_day"].to_numpy())
        clf.fit(tr[features].to_numpy(), tr["flood_on_day"].to_numpy(), sample_weight=sw)
        p = clf.predict_proba(va[features].to_numpy())[:, 1]
        aps.append(average_precision_score(va["flood_on_day"].to_numpy(), p))
    return float(np.mean(aps))


def fit_hgb(pooled: pd.DataFrame, mask: np.ndarray, features: list[str], params: dict):
    sub = pooled[mask]
    X, y = sub[features].to_numpy(), sub["flood_on_day"].to_numpy()
    clf = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, monotonic_cst=monotonic_cst(features), **params)
    sw = compute_sample_weight("balanced", y)
    clf.fit(X, y, sample_weight=sw)
    return clf


def fit_per_horizon_calibrators(scores: np.ndarray, y: np.ndarray,
                                horizon: np.ndarray) -> dict:
    cals = {}
    for h in HORIZONS:
        m = horizon == h
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores[m], y[m])
        cals[h] = iso
    return cals


def apply_calibration(scores: np.ndarray, horizon: np.ndarray, cals: dict) -> np.ndarray:
    out = np.empty_like(scores, dtype=float)
    for h in HORIZONS:
        m = horizon == h
        out[m] = cals[h].predict(scores[m])
    return out


def choose_threshold(cal_scores: np.ndarray, y: np.ndarray, target_recall: float) -> float:
    """Highest threshold on calibrated scores that still yields recall >= target."""
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, cal_scores)
    ok = rec[:-1] >= target_recall  # rec[:-1] aligns with thr
    if not ok.any():
        return float(np.min(cal_scores))
    return float(np.max(thr[ok]))


def per_horizon_metrics(cal_scores: np.ndarray, y: np.ndarray, horizon: np.ndarray,
                        thresholds: dict) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        m = horizon == h
        yh, ph = y[m], cal_scores[m]
        pred = (ph >= thresholds[h]).astype(int)
        rows.append({
            "horizon": h,
            "base_rate": float(yh.mean()),
            "roc_auc": float(roc_auc_score(yh, ph)) if yh.sum() else np.nan,
            "pr_auc": float(average_precision_score(yh, ph)) if yh.sum() else np.nan,
            "recall": float(recall_score(yh, pred, zero_division=0)),
            "precision": float(precision_score(yh, pred, zero_division=0)),
            "f1": float(f1_score(yh, pred, zero_division=0)),
            "brier": float(brier_score_loss(yh, ph)),
            "n": int(m.sum()), "n_pos": int(yh.sum()),
        })
    agg = {
        "horizon": "ALL", "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, cal_scores)),
        "pr_auc": float(average_precision_score(y, cal_scores)),
        "recall": np.nan, "precision": np.nan, "f1": np.nan,
        "brier": float(brier_score_loss(y, cal_scores)),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }
    return pd.DataFrame(rows + [agg])


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("DAY-BY-DAY 7-DAY FLOOD FORECASTER -- gauge 01543000")
    print("=" * 78)

    df = load_data()
    print(f"[data] {len(df)} daily rows {df['date'].min().date()}..{df['date'].max().date()}; "
          f"missing days: 0")

    # --- threshold on TRAIN YEARS 1980-2004 only ---
    train_years = df[df["date"].dt.year <= 2004]
    threshold = float(np.percentile(train_years["q_cfs"], 98))
    thr_mm = cfs_to_mm_per_day(threshold, BASIN["area_km2"])
    thr_m3s = threshold / 35.3147  # cfs -> m3/s (for the GloFAS comparison, v2)
    base_rate_full = float((df["q_cfs"] >= threshold).mean())
    print(f"[label] flood threshold = 98th pct(1980-2004) = {threshold:.1f} cfs "
          f"= {thr_mm:.2f} mm/day = {thr_m3s:.1f} m3/s; "
          f"1-day base rate over full record = {base_rate_full:.4%}")

    # --- pooled dataset ---
    pooled = build_pooled(df, threshold)
    m = split_masks(pooled["anchor_date"])
    n_anchor = pooled["anchor_date"].nunique()
    print(f"[pooled] {len(pooled)} rows from {n_anchor} anchor days x {len(HORIZONS)} horizons")

    # --- embargo assertion ---
    embargo_days = pd.to_datetime(pooled.loc[m["embargo"], "anchor_date"]).nunique()
    assert embargo_days == 7, f"embargo should drop 7 anchor days, got {embargo_days}"
    print(f"[embargo] confirmed exactly {embargo_days} anchor days dropped "
          f"({EMBARGO_START.date()}..{EMBARGO_END.date()})")

    fit_days = pd.to_datetime(pooled.loc[m['fit'], 'anchor_date'])
    val_days = pd.to_datetime(pooled.loc[m['val'], 'anchor_date'])
    test_days = pd.to_datetime(pooled.loc[m['test'], 'anchor_date'])
    print(f"[split] fit {fit_days.min().date()}..{fit_days.max().date()} "
          f"({fit_days.nunique()} days) | val {val_days.min().date()}..{val_days.max().date()} "
          f"({val_days.nunique()} days) | test {test_days.min().date()}..{test_days.max().date()} "
          f"({test_days.nunique()} days)")
    pr_train = float(pooled.loc[m["fit"] | m["val"], "flood_on_day"].mean())
    pr_test = float(pooled.loc[m["test"], "flood_on_day"].mean())
    print(f"[label] per-day flood rate: train {pr_train:.4%} | test {pr_test:.4%}")

    # --- anchor-day-grouped CV to pick HGB config (leakage-safe) ---
    configs = {
        "hgb_a": dict(learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
                      min_samples_leaf=60, l2_regularization=1.0, early_stopping=False),
        "hgb_b": dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
                      min_samples_leaf=100, l2_regularization=1.0, early_stopping=False),
    }
    cv_scores = {name: anchor_day_cv_score(pooled, m["fit"], FEATURES_A, p)
                 for name, p in configs.items()}
    best_name = max(cv_scores, key=cv_scores.get)
    best_params = configs[best_name]
    print(f"[cv] anchor-day time-fold AP: " +
          ", ".join(f"{k}={v:.4f}" for k, v in cv_scores.items()) +
          f"  -> chose {best_name}")

    # --- fit Model A on the FIT slice; calibrate + threshold on VALIDATION ---
    modelA = fit_hgb(pooled, m["fit"], FEATURES_A, best_params)
    val = pooled[m["val"]]
    val_scoreA = modelA.predict_proba(val[FEATURES_A].to_numpy())[:, 1]
    calsA = fit_per_horizon_calibrators(val_scoreA, val["flood_on_day"].to_numpy(),
                                        val["horizon"].to_numpy())
    val_calA = apply_calibration(val_scoreA, val["horizon"].to_numpy(), calsA)
    thresholdsA = {h: choose_threshold(val_calA[val["horizon"].to_numpy() == h],
                                       val["flood_on_day"].to_numpy()[val["horizon"].to_numpy() == h],
                                       TARGET_RECALL) for h in HORIZONS}

    # --- LR pooled baseline (standardized) for coefficients ---
    fit = pooled[m["fit"]]
    scaler = StandardScaler().fit(fit[FEATURES_A].to_numpy())
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(fit[FEATURES_A].to_numpy()), fit["flood_on_day"].to_numpy())
    test = pooled[m["test"]]
    lr_test_ap = float(average_precision_score(
        test["flood_on_day"].to_numpy(),
        lr.predict_proba(scaler.transform(test[FEATURES_A].to_numpy()))[:, 1]))
    lr_coefs = dict(sorted(zip(FEATURES_A, lr.coef_[0]), key=lambda kv: -abs(kv[1])))

    # --- Model B (adds q_t, q_ewma_3) ---
    modelB = fit_hgb(pooled, m["fit"], FEATURES_B, best_params)
    val_scoreB = modelB.predict_proba(val[FEATURES_B].to_numpy())[:, 1]
    calsB = fit_per_horizon_calibrators(val_scoreB, val["flood_on_day"].to_numpy(),
                                        val["horizon"].to_numpy())
    val_calB = apply_calibration(val_scoreB, val["horizon"].to_numpy(), calsB)
    thresholdsB = {h: choose_threshold(val_calB[val["horizon"].to_numpy() == h],
                                       val["flood_on_day"].to_numpy()[val["horizon"].to_numpy() == h],
                                       TARGET_RECALL) for h in HORIZONS}

    # --- evaluate on TEST (calibrated) ---
    test_scoreA = modelA.predict_proba(test[FEATURES_A].to_numpy())[:, 1]
    test_calA = apply_calibration(test_scoreA, test["horizon"].to_numpy(), calsA)
    metricsA = per_horizon_metrics(test_calA, test["flood_on_day"].to_numpy(),
                                   test["horizon"].to_numpy(), thresholdsA)
    test_scoreB = modelB.predict_proba(test[FEATURES_B].to_numpy())[:, 1]
    test_calB = apply_calibration(test_scoreB, test["horizon"].to_numpy(), calsB)
    metricsB = per_horizon_metrics(test_calB, test["flood_on_day"].to_numpy(),
                                   test["horizon"].to_numpy(), thresholdsB)

    print("\n[Model A -- rain-only, per-horizon test metrics]")
    print(metricsA.to_string(index=False,
          formatters={c: "{:.4f}".format for c in
                      ["base_rate", "roc_auc", "pr_auc", "recall", "precision", "f1", "brier"]}))
    print(f"\n[Model B -- +q_t] aggregate ROC-AUC={metricsB.iloc[-1]['roc_auc']:.4f} "
          f"PR-AUC={metricsB.iloc[-1]['pr_auc']:.4f} "
          f"(near h1 PR-AUC {metricsB.iloc[0]['pr_auc']:.4f} vs A {metricsA.iloc[0]['pr_auc']:.4f})")
    print(f"[LR baseline] pooled test PR-AUC={lr_test_ap:.4f}  "
          f"top coefs: " + ", ".join(f"{k}:{v:+.2f}" for k, v in list(lr_coefs.items())[:6]))

    # --- permutation importance for Model A (validation subsample) ---
    rng = np.random.default_rng(RANDOM_STATE)
    vidx = rng.choice(len(val), size=min(4000, len(val)), replace=False)
    perm = permutation_importance(
        modelA, val[FEATURES_A].to_numpy()[vidx], val["flood_on_day"].to_numpy()[vidx],
        scoring="average_precision", n_repeats=4, random_state=RANDOM_STATE)
    imp = dict(sorted(zip(FEATURES_A, perm.importances_mean), key=lambda kv: -kv[1]))
    print("[Model A importance] top: " +
          ", ".join(f"{k}:{v:.3f}" for k, v in list(imp.items())[:6]))

    # =====================================================================
    #  WINDOWED headline model  (flood_within7, antecedent-only, balanced LR)
    # =====================================================================
    win = (pooled.groupby("anchor_date")
           .agg(flood_within7=("flood_on_day", "max"),
                **{c: (c, "first") for c in ANTECEDENT})
           .reset_index())
    wm = split_masks(win["anchor_date"])
    w_fit, w_val, w_test = win[wm["fit"]], win[wm["val"]], win[wm["test"]]
    w_scaler = StandardScaler().fit(w_fit[FEATURES_WINDOWED].to_numpy())
    w_lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    w_lr.fit(w_scaler.transform(w_fit[FEATURES_WINDOWED].to_numpy()),
             w_fit["flood_within7"].to_numpy())
    w_val_score = w_lr.predict_proba(w_scaler.transform(w_val[FEATURES_WINDOWED].to_numpy()))[:, 1]
    w_cal = IsotonicRegression(out_of_bounds="clip")
    w_cal.fit(w_val_score, w_val["flood_within7"].to_numpy())
    w_val_cal = w_cal.predict(w_val_score)
    w_threshold = choose_threshold(w_val_cal, w_val["flood_within7"].to_numpy(), TARGET_RECALL)
    w_test_score = w_lr.predict_proba(w_scaler.transform(w_test[FEATURES_WINDOWED].to_numpy()))[:, 1]
    w_test_cal = w_cal.predict(w_test_score)
    yw = w_test["flood_within7"].to_numpy()
    w_pred = (w_test_cal >= w_threshold).astype(int)
    windowed_metrics = {
        "base_rate": float(yw.mean()),
        "roc_auc": float(roc_auc_score(yw, w_test_cal)),
        "pr_auc": float(average_precision_score(yw, w_test_cal)),
        "recall": float(recall_score(yw, w_pred, zero_division=0)),
        "precision": float(precision_score(yw, w_pred, zero_division=0)),
        "brier": float(brier_score_loss(yw, w_test_cal)),
        "threshold": w_threshold,
    }
    print(f"\n[Windowed headline] flood_within7 base rate {windowed_metrics['base_rate']:.3%} | "
          f"ROC-AUC {windowed_metrics['roc_auc']:.3f} | PR-AUC {windowed_metrics['pr_auc']:.3f} | "
          f"recall {windowed_metrics['recall']:.2f} precision {windowed_metrics['precision']:.2f}")

    # =====================================================================
    #  §8 forecast-degradation stress test (bounds the hindcast optimism)
    # =====================================================================
    stress = []
    for h in HORIZONS:
        mh = test["horizon"].to_numpy() == h
        Xh = test[FEATURES_A].to_numpy()[mh].copy()
        # multiplicative lognormal noise on the forecast-block rain, growing with h
        sigma = 0.15 * h
        for j, f in enumerate(FEATURES_A):
            if f.startswith("fc_rain"):
                noise = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=mh.sum())
                Xh[:, j] = Xh[:, j] * noise
        s = modelA.predict_proba(Xh)[:, 1]
        c = calsA[h].predict(s)
        yh = test["flood_on_day"].to_numpy()[mh]
        stress.append({"horizon": h,
                       "pr_auc_perfect": float(metricsA.iloc[h - 1]["pr_auc"]),
                       "pr_auc_degraded": float(average_precision_score(yh, c))})
    stress_df = pd.DataFrame(stress)
    print("\n[stress] forecast-degradation PR-AUC (perfect vs noisy QPF):")
    print(stress_df.to_string(index=False, formatters={
        "pr_auc_perfect": "{:.3f}".format, "pr_auc_degraded": "{:.3f}".format}))

    # =====================================================================
    #  Figures
    # =====================================================================
    _fig_skill_gradient(metricsA)
    _fig_reliability(test_calA, test["flood_on_day"].to_numpy(), test["horizon"].to_numpy())
    _fig_windowed(yw, w_test_cal)
    _fig_stress(stress_df)
    _fig_importance(lr_coefs, imp)
    print(f"[figures] saved 5 figures to {FIGURES}")

    # =====================================================================
    #  Persist bundles + test predictions
    # =====================================================================
    feature_groups = {"antecedent": ANTECEDENT, "forecast": FORECAST, "horizon": HORIZON}
    meta_common = {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "random_state": RANDOM_STATE,
        "ewma_alpha": EWMA_ALPHA, "q_ewma_alpha": Q_EWMA_ALPHA,
        "fit_period": [str(fit_days.min().date()), str(fit_days.max().date())],
        "validation_period": [str(val_days.min().date()), str(val_days.max().date())],
        "test_period": [str(test_days.min().date()), str(test_days.max().date())],
        "embargo": [str(EMBARGO_START.date()), str(EMBARGO_END.date())],
        "horizons": HORIZONS, "target_recall": TARGET_RECALL,
        "calibration_method": "isotonic per horizon",
        "cv_scheme": "anchor-day expanding-window time folds (average_precision)",
        "cv_scores": cv_scores, "chosen_config": best_name, "hgb_params": best_params,
        "per_horizon_base_rate": {int(r.horizon): r.base_rate
                                  for r in metricsA.itertuples() if r.horizon != "ALL"},
        "one_day_model_roc_auc": ONE_DAY_MODEL_ROC_AUC,
        "windowed_model_roc_auc": windowed_metrics["roc_auc"],
        "lr_baseline_pooled_pr_auc": lr_test_ap,
    }

    def horizon_metric_dict(mdf):
        return {int(r.horizon): {"roc_auc": r.roc_auc, "pr_auc": r.pr_auc,
                                 "recall": r.recall, "precision": r.precision, "brier": r.brier}
                for r in mdf.itertuples() if r.horizon != "ALL"}

    bundleA = {
        "model": modelA, "calibrators": calsA, "thresholds": thresholdsA,
        "feature_groups": feature_groups, "features": FEATURES_A,
        "label_name": "flood_on_day",
        "flood_threshold_cfs": threshold, "flood_threshold_mm_day": thr_mm,
        "flood_threshold_m3s": thr_m3s,
        "basin": BASIN,
        "metadata": {**meta_common, "model": "A (rain-only, pooled HGB)",
                     "requires_usgs": False,
                     "per_horizon_test_metrics": horizon_metric_dict(metricsA)},
    }
    bundleB = {
        "model": modelB, "calibrators": calsB, "thresholds": thresholdsB,
        "feature_groups": {**feature_groups, "discharge": ["q_t", "q_ewma_3"]},
        "features": FEATURES_B, "label_name": "flood_on_day",
        "flood_threshold_cfs": threshold, "flood_threshold_mm_day": thr_mm,
        "flood_threshold_m3s": thr_m3s,
        "basin": BASIN,
        "metadata": {**meta_common, "model": "B (rain + day-t discharge)",
                     "requires_usgs": True,
                     "per_horizon_test_metrics": horizon_metric_dict(metricsB)},
    }
    windowed_bundle = {
        "model": w_lr, "scaler": w_scaler, "calibrator": w_cal,
        "threshold": w_threshold, "features": FEATURES_WINDOWED,
        "label_name": "flood_within7",
        "flood_threshold_cfs": threshold, "flood_threshold_mm_day": thr_mm,
        "flood_threshold_m3s": thr_m3s,
        "basin": BASIN,
        "metadata": {**meta_common, "model": "windowed headline (balanced LR + isotonic)",
                     "metrics": windowed_metrics},
    }
    joblib.dump(bundleA, MODELS / "flood_daybyday_model.joblib")
    joblib.dump(bundleB, MODELS / "flood_daybyday_model_B.joblib")
    joblib.dump(windowed_bundle, MODELS / "flood_windowed_model.joblib")

    # test-prediction CSVs
    outA = test[["anchor_date", "target_date", "horizon", "flood_on_day", *FEATURES_A]].copy()
    outA.insert(4, "p_h", test_calA)
    outA.to_csv(DATA / "test_predictions.csv", index=False)
    outW = w_test[["anchor_date", "flood_within7", *FEATURES_WINDOWED]].copy()
    outW.insert(2, "p_window", w_test_cal)
    outW.to_csv(DATA / "test_predictions_windowed.csv", index=False)

    # save a JSON metrics summary for the notebook/app "how good" tab
    summary = {
        "flood_threshold_cfs": threshold, "flood_threshold_mm_day": thr_mm,
        "flood_threshold_m3s": thr_m3s,
        "per_day_base_rate_train": pr_train, "per_day_base_rate_test": pr_test,
        "pooled_rows": int(len(pooled)), "anchor_days": int(n_anchor),
        "metricsA": metricsA.to_dict(orient="records"),
        "metricsB": metricsB.to_dict(orient="records"),
        "windowed": windowed_metrics, "stress": stress_df.to_dict(orient="records"),
        "lr_coefs": {k: float(v) for k, v in lr_coefs.items()},
        "perm_importance": {k: float(v) for k, v in imp.items()},
        "cv_scores": cv_scores,
        "one_day_model_roc_auc": ONE_DAY_MODEL_ROC_AUC,
        "old_windowed_roc_auc": OLD_WINDOWED_ROC_AUC,
    }
    (DATA / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # =====================================================================
    #  Round-trip: reload every bundle, re-score test, assert allclose
    # =====================================================================
    rb = joblib.load(MODELS / "flood_daybyday_model.joblib")
    rs = rb["model"].predict_proba(test[rb["features"]].to_numpy())[:, 1]
    rc = apply_calibration(rs, test["horizon"].to_numpy(), rb["calibrators"])
    assert np.allclose(rc, test_calA, atol=1e-10), "round-trip drift in Model A"
    rbB = joblib.load(MODELS / "flood_daybyday_model_B.joblib")
    rsB = rbB["model"].predict_proba(test[rbB["features"]].to_numpy())[:, 1]
    rcB = apply_calibration(rsB, test["horizon"].to_numpy(), rbB["calibrators"])
    assert np.allclose(rcB, test_calB, atol=1e-10), "round-trip drift in Model B"
    assert not np.isnan(test_calA).any(), "NaNs in calibrated predictions"
    print("[roundtrip] reload + re-score matches in-run predictions (atol 1e-10)")

    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"threshold: {threshold:.1f} cfs = {thr_mm:.2f} mm/day")
    print(f"per-day base rate: train {pr_train:.3%} / test {pr_test:.3%}")
    print(f"pooled rows: {len(pooled)} ({n_anchor} anchor days x 7)")
    print(f"anchor-day CV confirmed (no day straddles a split); embargo dropped 7 days")
    print(f"Model A aggregate: ROC-AUC {metricsA.iloc[-1]['roc_auc']:.3f} "
          f"PR-AUC {metricsA.iloc[-1]['pr_auc']:.3f}")
    print(f"  near h1: ROC-AUC {metricsA.iloc[0]['roc_auc']:.3f} "
          f"PR-AUC {metricsA.iloc[0]['pr_auc']:.3f}  |  "
          f"far h7: ROC-AUC {metricsA.iloc[6]['roc_auc']:.3f} "
          f"PR-AUC {metricsA.iloc[6]['pr_auc']:.3f}")
    print(f"Model B near h1 PR-AUC {metricsB.iloc[0]['pr_auc']:.3f} (A: {metricsA.iloc[0]['pr_auc']:.3f})")
    print(f"windowed headline: ROC-AUC {windowed_metrics['roc_auc']:.3f} "
          f"(vs 1-day model 0.943, old windowed ~0.79)")
    print(f"default served model: B when USGS up, else A")
    print("bundles + CSVs + figures written. ALL ASSERTIONS PASSED.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_skill_gradient(metricsA: pd.DataFrame) -> None:
    d = metricsA[metricsA["horizon"] != "ALL"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(d["horizon"], d["roc_auc"], "o-", label="ROC-AUC", color="#2166ac")
    ax.plot(d["horizon"], d["pr_auc"], "s-", label="PR-AUC (avg precision)", color="#b2182b")
    ax.axhline(0.943, ls="--", color="#2166ac", alpha=0.5, label="1-day model ROC-AUC 0.943")
    for _, r in d.iterrows():
        ax.axhline(r["base_rate"], xmin=0, xmax=0, alpha=0)  # keep autoscale sane
    ax.set_xlabel("lead day (t+h)"); ax.set_ylabel("score")
    ax.set_title("Skill gradient: forecast sharpness degrades with lead day")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "skill_gradient.png", dpi=130); plt.close(fig)


def _fig_reliability(cal, y, horizon) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    for h in (1, 3, 5, 7):
        mh = horizon == h
        ph, yh = cal[mh], y[mh]
        bins = np.linspace(0, min(1.0, max(0.2, ph.max())), 8)
        idx = np.digitize(ph, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            sel = idx == b
            if sel.sum() >= 20:
                xs.append(ph[sel].mean()); ys.append(yh[sel].mean())
        brier = brier_score_loss(yh, ph)
        ax.plot(xs, ys, "o-", label=f"t+{h}  (Brier {brier:.3f})", alpha=0.8)
    ax.set_xlabel("mean predicted P(flood)"); ax.set_ylabel("observed flood freq")
    ax.set_title("Per-horizon reliability (calibrated)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "reliability_per_horizon.png", dpi=130); plt.close(fig)


def _fig_windowed(y, cal) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(cal, bins) - 1
    xs, ys = [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() >= 20:
            xs.append(cal[sel].mean()); ys.append(y[sel].mean())
    ax.plot(xs, ys, "o-", color="#762a83", label=f"windowed (Brier {brier_score_loss(y, cal):.3f})")
    ax.set_xlabel("mean predicted P(>=1 flood day in 7)"); ax.set_ylabel("observed freq")
    ax.set_title("Windowed headline model reliability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "windowed_calibration.png", dpi=130); plt.close(fig)


def _fig_stress(stress_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(stress_df["horizon"], stress_df["pr_auc_perfect"], "s-",
            label="perfect-prog (held-out test)", color="#1a9850")
    ax.plot(stress_df["horizon"], stress_df["pr_auc_degraded"], "o--",
            label="noisy QPF (stress test)", color="#d73027")
    ax.set_xlabel("lead day (t+h)"); ax.set_ylabel("PR-AUC")
    ax.set_title("Forecast-degradation stress test bounds the hindcast optimism")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "forecast_degradation.png", dpi=130); plt.close(fig)


def _fig_importance(lr_coefs: dict, perm: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    lk = list(lr_coefs.items())[:10][::-1]
    axes[0].barh([k for k, _ in lk], [v for _, v in lk], color="#4575b4")
    axes[0].set_title("LR standardized coefficients (top 10)"); axes[0].grid(alpha=0.3)
    pk = list(perm.items())[:10][::-1]
    axes[1].barh([k for k, _ in pk], [v for _, v in pk], color="#b2182b")
    axes[1].set_title("Model A permutation importance (top 10)"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "feature_importance.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
